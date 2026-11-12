"""
JO-PT : 오픈소스 LLM 채팅 프로그램 (데이터처리개론 과제)
- 챗봇: JOBOT / 모델: EXAONE 3.5 2.4B (Ollama로 내 PC에서 실행)
- 실행: py -3.13 -m streamlit run chat.py --theme.base light
"""
import base64
import json
import os
import re
import threading
import time
import requests
import streamlit as st

# ======================= 여기만 바꾸면 됩니다 =======================
PAGE_NAME = "JO-PT"                     # 큰 페이지 이름
BOT_NAME = "JOBOT"                      # 챗봇 이름
BOT_PHOTO = "jobot.png"                 # 챗봇 프로필 사진 (같은 폴더)
USER_AVATAR = "🧑‍🎓"                    # 내 아이콘
MODEL = "exaone3.5:2.4b"               # DeepSeek: "deepseek-r1:7b" 등
PERSONA = "조선해양공학과 학생을 돕는 친절한 한국어 AI 도우미"
# ===================================================================

# 브라우저 탭 이름·아이콘 (st 명령 중 가장 먼저 실행)
st.set_page_config(page_title=PAGE_NAME, page_icon="🚢", layout="centered")

OLLAMA_URL = "http://localhost:11434"
HERE = os.path.dirname(os.path.abspath(__file__))
PHOTO_PATH = os.path.join(HERE, BOT_PHOTO)
HAS_PHOTO = os.path.exists(PHOTO_PATH)
BOT_AVATAR = PHOTO_PATH if HAS_PHOTO else "🤖"

SYSTEM_PROMPT = f"""너의 이름은 {BOT_NAME}이다. 너는 AI이고, {PERSONA}이다.
[규칙]
1. 너의 이름은 오직 {BOT_NAME}이다. 절대로 다른 이름을 네 이름이라고 말하지 않는다.
2. 대화 상대는 '사용자'이다. 사용자의 이름은 사용자의 것이지 너의 이름이 아니다.
3. 대화 기록에 있는 내용을 근거로 답하고, 모르는 내용은 지어내지 않는다.
4. 한국어로 1~3문장, 짧고 정확하게 답한다. 이모지는 쓰지 않는다.
5. 너는 사용자가 만든 과제 프로그램과 연결되어 있다.
   - 과제1: 어떤 데이터든 X 한 개로 Y를 예측하는 선형 회귀 + 2차 다항 회귀 (기본 예시: 풍속 → 유의파고)
   - 과제2: 어떤 데이터든 여러 입력으로 출력을 예측하는 인공신경망, 선형회귀와 비교 (기본 예시: 선박 길이·너비·깊이 → 총톤수)
   [과제 프로그램 실행 결과]가 주어지면 그 숫자를 그대로 근거로 해석하고, 숫자를 바꾸거나 지어내지 않는다."""

GEN_OPTIONS = {"temperature": 0.3, "top_p": 0.9, "repeat_penalty": 1.1}   # 엉뚱한 말 줄이기

# ================================================================
#  과제 1·2 프로그램 연결 (assignments.py)
# ================================================================
try:
    import assignments as A
    ASSIGN_ERR = None
except Exception as _e:          # 과제 파일이 없거나 오류가 있을 때
    A, ASSIGN_ERR = None, str(_e)

ACTION = r"실행|돌려|분석해|분석\s*결과|그래프|결과\s*(보여|알려)|학습시|학습해|보여\s*줘"
PAST = r"아까|전에|이전|뭐였|얼마였|몇이었|기억"
NUM = r"-?\d+(?:\.\d+)?"


def _nums(q):
    q = re.sub(r"과제\s*\d|\d+\s*차|\d+\s*번", " ", q)          # '과제1', '2차' 같은 숫자는 제외
    return [float(x) for x in re.findall(NUM, q)]


def _after(q, words):
    m = re.search(rf"(?:{words})\D{{0,6}}?({NUM})", q)
    return float(m.group(1)) if m else None


def _has(q, names):
    """문장에 열 이름(별명 포함)이 들어 있는지"""
    return any(al in q for n in names for al in A.aliases(n)) if A else False


def _alias_re(name):
    return "|".join(re.escape(a) for a in sorted(A.aliases(name), key=len, reverse=True))


HP_PATTERNS = {   # 채팅 문장에서 하이퍼파라미터 읽기: "은닉 노드 10개, 학습률 0.03, 에포크 200, 배치 64"
    "hidden": r"(?:은닉\s*(?:층|노드)?|노드)\D{0,6}?(\d+)",
    "epochs": r"(?:에포크|에폭|epochs?)\D{0,4}?(\d+)",
    "lr": r"(?:학습률|러닝\s*레이트|learning\s*rate|lr)\D{0,4}?(\d*\.\d+|\d+)",
    "batch": r"(?:배치(?:\s*크기|\s*사이즈)?|batch(?:\s*size)?)\D{0,4}?(\d+)",
}


def parse_hp(q):
    hp = {}
    for k, pat in HP_PATTERNS.items():
        m = re.search(pat, q, re.I)
        if m:
            v = float(m.group(1))
            hp[k] = v if k == "lr" else int(v)
    return hp


def detect_tasks(q, a1cols, a2cols):
    """채팅 문장에서 과제 실행/예측 요청을 찾음
    a1cols = (X열, Y열),  a2cols = ([입력 열들], 출력 열)
    → [("a1",), ("a2",), ("p1", x), ("p2", [값들])]"""
    tasks, past, nums = [], re.search(PAST, q), _nums(q)
    hp = parse_hp(q)
    if hp and not past and re.search(r"신경망|ANN|과제\s*2|학습", q, re.I):
        return [("a2", hp)]                          # "학습률 0.03으로 신경망 돌려줘"
    x1, y1 = a1cols
    x2s, y2 = a2cols
    nn_word = re.search(r"신경망|ANN|과제\s*2", q, re.I)
    # 과제2 예측: 입력 열 값이 모두 있을 때
    if not past and len(nums) >= len(x2s) and (_has(q, x2s) or _has(q, [y2])) and \
            (len(x2s) >= 2 or nn_word or not _has(q, [x1])):
        vals = [_after(q, _alias_re(c)) for c in x2s]
        if None in vals:
            vals = nums[:len(x2s)]
        return [("p2", vals)]
    # 과제1 예측: X 값 하나
    if not past and nums and (_has(q, [x1]) or _has(q, [y1])):
        x = _after(q, _alias_re(x1))
        return [("p1", nums[0] if x is None else x)]
    if past or not re.search(ACTION, q):
        return []
    both = re.search(r"과제\s*(전부|다|모두|둘\s*다)|1\s*(,|이랑|하고|와|과|랑)\s*2", q)
    if both or re.search(r"과제\s*1|회귀\s*분석|회귀\s*모델", q) or (_has(q, [x1, y1]) and not nn_word):
        tasks.append(("a1",))
    if both or nn_word or _has(q, [y2]):
        tasks.append(("a2", {}))
    out = []
    for t in tasks:
        if t not in out:
            out.append(t)
    return out

# ================================================================
#  기억(Memory) — 교수님이 말씀하신 세 가지 방법을 모두 구현 (사이드바에서 켜고 끄기)
#   ① 전체 기억 : 지금까지 나눈 대화를 매번 통째로 LLM에 전달 (memory.json)
#   ② 메모 기억 : 대화가 끝날 때마다 JOBOT이 중요한 내용을 메모로 정리해 두고,
#                 다음 질문 때 그 메모를 먼저 읽음 (memo.json)  ← "기록해놓고 기억"
#   ③ RAG      : 모든 질문·답변을 벡터로 저장(memory_rag.json)하고, 새 질문과 관련된
#                 과거 대화를 '검색'해서 전달 → ①에 다 못 넣은 오래된 대화도 기억
# ================================================================
HISTORY_FILE = os.path.join(HERE, "memory.json")
RAG_FILE = os.path.join(HERE, "memory_rag.json")
MEMO_FILE = os.path.join(HERE, "memo.json")
EMBED_MODEL = "bge-m3"     # 문장 의미를 숫자 벡터로 바꾸는 임베딩 모델 (ollama pull bge-m3)
NUM_CTX = 16384            # LLM이 한 번에 읽을 수 있는 길이(토큰). 기본값(2048~4096)보다 크게
HISTORY_BUDGET = 12000     # ① 전체 기억에 넣을 대화 최대 글자 수 (넘치면 오래된 것부터 ③이 담당)
SHORT_CONTEXT = 6          # ①을 끈 경우 최근 대화 몇 개만 전달
TOP_K = 4                  # ③ 검색해서 가져올 과거 대화 수


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def load_history():
    return _load(HISTORY_FILE)


def save_history(msgs):
    _save(HISTORY_FILE, msgs)


PROFILE_FILE = os.path.join(HERE, "profile.json")
PROFILE_PATTERNS = {   # "내 ○○은 △△야" 형태의 말에서 사용자 정보를 뽑아 따로 메모
    "이름": r"(?:내|제)\s*이름은\s*([가-힣A-Za-z]{2,10}?)(?:이야|야|입니다|이에요|예요|이고|고|임|라고|\s|[.!,]|$)",
    "전공": r"(?:(?:내|제)\s*)?전공은\s*([가-힣A-Za-z ]{2,20}?)(?:이야|야|입니다|이에요|예요|이고|고|임|\s*[.!,]|$)",
    "학교": r"(?:나는|난|저는)?\s*([가-힣]{2,15}대학교|[가-힣]{2,10}대)\s*(?:다녀|에 다녀|학생)",
}


def load_profile():
    try:
        with open(PROFILE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def update_profile(text):
    prof = load_profile()
    changed = False
    for key, pat in PROFILE_PATTERNS.items():
        m = re.search(pat, text)
        if m and prof.get(key) != m.group(1).strip():
            prof[key] = m.group(1).strip()
            changed = True
    if changed:
        _save(PROFILE_FILE, prof)
    return prof


BOT_NAME_Q = re.compile(r"(너|니|네|넌|당신|너는|너의)\s*(이름|누구|정체|뭐야|이 누구)|^\s*(이름이?\s*뭐|누구세요|누구야)")
USER_NAME_Q = re.compile(r"(내|제|나의|저의)\s*이름")


def rule_answer(q):
    """이름 질문은 LLM이 헷갈리지 않도록 프로그램이 직접 답함. 해당 없으면 None"""
    prof = load_profile()
    if USER_NAME_Q.search(q) and re.search(r"뭐|무엇|알아|기억|누구|맞|말해", q) and "은" not in q.split("이름")[1][:2]:
        if "이름" in prof:
            return f"{prof['이름']}님이라고 알려주셨어요."
        return "아직 이름을 알려주지 않으셨어요. 이름이 어떻게 되세요?"
    if BOT_NAME_Q.search(q) and len(q) <= 25:
        return f"저는 {BOT_NAME}예요. {PERSONA}랍니다. 궁금한 거 있으면 편하게 물어보세요."
    return None


def embed(text):
    """임베딩 모델로 문장을 벡터로 변환. 모델이 없으면 None (→ 키워드 검색만 사용)"""
    try:
        r = requests.post(f"{OLLAMA_URL}/api/embed",
                          json={"model": EMBED_MODEL, "input": [text]}, timeout=60)
        r.raise_for_status()
        return r.json()["embeddings"][0]
    except Exception:
        return None


def _grams(t):
    """키워드 검색용: 글자 2개씩 묶음 (예: '선형회귀' → 선형, 형회, 회귀)"""
    t = "".join(ch for ch in t.lower() if ch.isalnum())
    stop = {"뭐였", "였지", "얼마", "아까", "해줘", "알려", "려줘", "뭐야", "물어", "어봤", "봤던"}
    return {t[i:i + 2] for i in range(len(t) - 1)} - stop


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-9)


def rag_add(question, answer, idx):
    """질문·답변 한 쌍을 RAG 저장소에 추가"""
    store = _load(RAG_FILE)
    text = f"사용자가 말함: {question}\n{BOT_NAME}가 답함: {answer[:400]}"
    store.append({"idx": idx, "time": time.strftime("%Y-%m-%d %H:%M"),
                  "text": text, "vec": embed(text)})
    _save(RAG_FILE, store)


def rag_rebuild_if_needed(msgs):
    """예전 버전에서 저장된 대화가 있으면 RAG 저장소를 처음 한 번 만들어 줌"""
    if _load(RAG_FILE) or len(msgs) < 2:
        return
    for i in range(len(msgs) - 1):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            rag_add(msgs[i]["content"], msgs[i + 1]["content"], i)


def rag_search(query, before_idx):
    """새 질문과 관련 있는 과거 대화 검색 (의미 유사도 70% + 키워드 30%)
    before_idx 이후 대화는 이미 ①로 전달되므로 제외"""
    cands = [x for x in _load(RAG_FILE) if x["idx"] < before_idx]
    if not cands:
        return []
    qv, qg = embed(query), _grams(query)
    hits = []
    for x in cands:
        lex = len(qg & _grams(x["text"])) / (len(qg) + 1e-9)
        if qv is not None and x.get("vec") is not None:
            score = 0.7 * _cos(qv, x["vec"]) + 0.3 * lex
        else:
            score = lex
        hits.append((score, x))
    hits.sort(key=lambda t: -t[0])
    return [dict(x, score=round(sc, 3)) for sc, x in hits[:TOP_K] if sc > 0.15]


def history_start(msgs, full):
    """LLM에 그대로 넣을 대화의 시작 위치. ①이 켜져 있으면 글자 수 한도 안에서 최대한 많이"""
    if not full:
        return max(0, len(msgs) - SHORT_CONTEXT)
    total, start = 0, len(msgs)
    for i in range(len(msgs) - 1, -1, -1):
        total += len(msgs[i]["content"])
        if total > HISTORY_BUDGET:
            break
        start = i
    return start


# ---------- ② 메모 기억
def load_memo():
    m = _load(MEMO_FILE)
    return m.get("memo", "") if isinstance(m, dict) else ""


def _update_memo(question, answer):
    old = load_memo() or "(없음)"
    prompt = (f"너는 {BOT_NAME}의 기억 정리 담당이다. 아래 [기존 메모]에 [이번 대화]에서 새로 알게 된 "
              "중요한 내용을 반영해서 메모를 업데이트해라.\n"
              "- 사용자에 대한 사실(이름, 전공, 관심사, 계획 등)과 사용자가 물어본 주제·핵심 답을 적는다.\n"
              "- 한 줄에 하나씩 '- '로 시작, 최대 15줄, 중복은 합친다. 인사말 같은 사소한 건 뺀다.\n"
              f"- 사용자 정보와 {BOT_NAME}(AI) 정보를 섞지 않는다.\n"
              "- 메모 내용만 출력한다.\n\n"
              f"[기존 메모]\n{old}\n\n[이번 대화]\n사용자: {question}\n{BOT_NAME}: {answer[:500]}")
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat",
                          json={"model": MODEL, "stream": False, "keep_alive": "30m",
                                "options": {"temperature": 0.1, "num_ctx": 4096},
                                "messages": [{"role": "user", "content": prompt}]},
                          timeout=600)
        memo = r.json()["message"]["content"].split("</think>")[-1].strip()
        if memo:
            _save(MEMO_FILE, {"memo": memo, "time": time.strftime("%Y-%m-%d %H:%M")})
    except Exception:
        pass


def update_memo_background(question, answer):
    """답변을 보여준 뒤 뒤에서 메모 정리 (기다리지 않아도 됨)"""
    threading.Thread(target=_update_memo, args=(question, answer), daemon=True).start()


def build_context(msgs, start, hits, memo, tool_result=""):
    """LLM에게 보낼 메시지 = 시스템 프롬프트 + ② 메모 + ③ 검색된 과거 기록 + ① 대화"""
    ctx = [{"role": "system", "content": SYSTEM_PROMPT}]
    if memo:
        ctx.append({"role": "system", "content":
                    f"[{BOT_NAME}의 기억 메모 - 지금까지 대화에서 정리해 둔 내용]\n{memo}"})
    if hits:
        past = "\n\n".join(f"({h['time']})\n{h['text']}" for h in hits)
        ctx.append({"role": "system", "content":
                    "[과거 대화 기록 - 지금 질문과 관련된 예전 대화]\n" + past +
                    "\n\n사용자가 예전에 물어본 내용이나 알려준 정보를 물으면 위 기록을 근거로 답해."})
    ctx += [{"role": m["role"], "content": m["content"]} for m in msgs[start:]]
    prof = load_profile()
    if prof:
        info = ", ".join(f"{k}은(는) {v}" for k, v in prof.items())
        ctx.insert(len(ctx) - 1, {"role": "system", "content":
                   f"참고: 대화 상대(사용자)의 {info}이다. 이것은 사용자의 정보이고, "
                   f"너의 이름은 {BOT_NAME}이다."})
    if tool_result:
        ctx.insert(len(ctx) - 1, {"role": "system", "content":
                   "[과제 프로그램 실행 결과] (사용자의 과제 코드가 방금 계산한 값)\n" + tool_result +
                   "\n\n위 결과의 숫자를 그대로 사용해서 무엇을 의미하는지 2~4문장으로 해석해줘."})
    return ctx


def photo_html(size):
    if not HAS_PHOTO:
        return f'<div class="ph" style="width:{size}px;height:{size}px;font-size:{size//2}px">🤖</div>'
    b64 = base64.b64encode(open(PHOTO_PATH, "rb").read()).decode()
    return f'<img class="ph" src="data:image/png;base64,{b64}" style="width:{size}px;height:{size}px">'


# ------------------------------------------------------------ 조선소 배경 그림 (SVG)
# 골리앗 크레인 + LNG 운반선(모스형 구형 화물창) + 바다 물결
SCENE_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 300' preserveAspectRatio='xMidYMax slice'>
 <g fill='#134074' opacity='.18'>
  <rect x='70' y='30' width='420' height='22' rx='3'/>
  <rect x='95' y='52' width='18' height='200'/><rect x='447' y='52' width='18' height='200'/>
  <path d='M95 252 h70 v8 h-70z M395 252 h70 v8 h-70z'/>
  <rect x='250' y='52' width='30' height='16'/><rect x='262' y='68' width='3' height='60'/>
  <rect x='252' y='128' width='22' height='8'/>
  <path d='M113 60 L447 230 M113 230 L447 60' stroke='#134074' stroke-width='3'/>
 </g>
 <g opacity='.92'>
  <rect x='540' y='112' width='46' height='58' rx='2' fill='#e9eef3' stroke='#0b2545' stroke-width='2'/>
  <rect x='548' y='120' width='30' height='8' fill='#0b2545' opacity='.7'/>
  <rect x='552' y='88' width='16' height='26' fill='#c0392b'/>
  <rect x='552' y='84' width='16' height='6' fill='#0b2545'/>
  <path d='M676 170 A50 50 0 0 1 776 170 Z M792 170 A50 50 0 0 1 892 170 Z M908 170 A50 50 0 0 1 1008 170 Z M1024 170 A50 50 0 0 1 1124 170 Z'
        fill='#f4f7fa' stroke='#0b2545' stroke-width='2'/>
  <path d='M676 170 h448' stroke='#9fb3c8' stroke-width='3'/>
  <path d='M520 168 L1190 168 Q1232 168 1222 186 L1188 222 L552 222 L522 196 Z' fill='#0b2545'/>
  <path d='M534 205 L1205 205 L1188 222 L552 222 Z' fill='#b23a3a'/>
  <path d='M520 168 L1190 168' stroke='#1b8a9a' stroke-width='3'/>
 </g>
 <path d='M0 215 Q100 200 200 215 T400 215 T600 215 T800 215 T1000 215 T1200 215 T1400 215 T1600 215 V300 H0Z' fill='#7cc3d6' opacity='.55'/>
 <path d='M0 235 Q120 220 240 235 T480 235 T720 235 T960 235 T1200 235 T1440 235 T1680 235 V300 H0Z' fill='#3d8fb0' opacity='.6'/>
 <path d='M0 258 Q150 245 300 258 T600 258 T900 258 T1200 258 T1500 258 T1800 258 V300 H0Z' fill='#134074' opacity='.75'/>
</svg>"""
SCENE_B64 = base64.b64encode(SCENE_SVG.encode("utf-8")).decode()


# ------------------------------------------------------------ 디자인
# 모든 글자색을 직접 지정 → 컴퓨터가 다크 모드여도 글자가 숨지 않음
INK = "#13253a"      # 기본 글자색 (진한 남색)
st.markdown(f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
html, body, .stApp, button, input, textarea {{
    font-family: 'Pretendard', -apple-system, 'Malgun Gothic', sans-serif !important;
}}
/* 배경: 설계 도면 격자 + 하늘→바다 그라데이션 */
.stApp {{
    background:
      linear-gradient(rgba(19,64,116,.06) 1px, transparent 1px) 0 0 / 32px 32px,
      linear-gradient(90deg, rgba(19,64,116,.06) 1px, transparent 1px) 0 0 / 32px 32px,
      linear-gradient(180deg, #dcecf8 0%, #eef6fc 45%, #f6fbff 100%) !important;
    color-scheme: light;
}}
/* 화면 아래: 크레인 + LNG 운반선 + 바다 */
.stApp::before {{
    content: ""; position: fixed; left: 0; right: 0; bottom: 0; height: 240px;
    background: url("data:image/svg+xml;base64,{SCENE_B64}") center bottom / cover no-repeat;
    pointer-events: none; z-index: 0;
}}
[data-testid="stMain"], section.main {{ position: relative; z-index: 1; }}
[data-testid="stMainBlockContainer"], .block-container {{ padding-bottom: 260px !important; }}
.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3,
[data-testid="stMarkdownContainer"] {{ color: {INK}; }}
header[data-testid="stHeader"] {{ background: transparent !important; }}
/* Streamlit 표시 숨기기: Deploy 버튼·메뉴·하단 문구 */
#MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"], [data-testid="stMainMenu"],
[data-testid="stStatusWidget"], [data-testid="stDecoration"] {{ display: none !important; }}

/* ---- 상단 배너 ---- */
.hero {{
    background: linear-gradient(135deg, #0b2545 0%, #134074 55%, #1b8a9a 100%);
    border-radius: 24px; padding: 26px 28px; margin-bottom: 18px;
    box-shadow: 0 12px 32px rgba(11,37,69,.28);
    display: flex; align-items: center; gap: 22px; position: relative; overflow: hidden;
}}
.hero::before {{   /* 배너 안 선박 선도(도면) 느낌의 곡선 */
    content: ""; position: absolute; right: -40px; top: -30px; width: 340px; height: 220px;
    background:
      radial-gradient(ellipse 160px 90px at 70% 100%, transparent 62%, rgba(255,255,255,.10) 63%, transparent 65%),
      radial-gradient(ellipse 200px 120px at 70% 100%, transparent 62%, rgba(255,255,255,.08) 63%, transparent 65%),
      radial-gradient(ellipse 240px 150px at 70% 100%, transparent 62%, rgba(255,255,255,.06) 63%, transparent 65%);
    pointer-events: none;
}}
.hero > * {{ position: relative; }}
.hero, .hero * {{ color: #ffffff !important; }}
.hero .ph {{ border-radius: 50%; border: 3px solid rgba(255,255,255,.85);
             object-fit: cover; flex-shrink: 0; box-shadow: 0 6px 18px rgba(0,0,0,.25); }}
.hero .title {{ font-size: 2.4rem; font-weight: 800; letter-spacing: -1px; line-height: 1.1; }}
.hero .sub {{ opacity: .9; margin-top: 4px; font-size: .95rem; }}
.hero .badges {{ margin-top: 12px; display: flex; gap: 6px; flex-wrap: wrap; }}
.badge {{ background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.3);
          padding: 3px 11px; border-radius: 999px; font-size: .78rem; }}
.badge.ok  {{ background: rgba(46,204,113,.3); border-color: rgba(46,204,113,.7); }}
.badge.err {{ background: rgba(231,76,60,.35); border-color: rgba(231,76,60,.8); }}

/* ---- 말풍선 ---- */
[data-testid="stChatMessage"] {{
    background: #ffffff !important; border: 1px solid #dbe8f3; border-radius: 18px;
    padding: 14px 16px; margin-bottom: 10px; box-shadow: 0 2px 10px rgba(19,64,116,.07);
}}
[data-testid="stChatMessage"] * {{ color: {INK} !important; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {{
    background: #e3f0fa !important; border-color: #c6dcef;
}}
[data-testid="stChatMessage"] img {{ border-radius: 50%; object-fit: cover; }}
[data-testid="stChatMessage"] code {{ background: #eef4f9 !important; }}

/* ---- 입력창 ---- */
[data-testid="stBottom"], [data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] {{ background: transparent !important; }}
[data-testid="stChatInput"], [data-testid="stChatInput"] > div {{
    background: #ffffff !important; border-radius: 16px !important;
}}
[data-testid="stChatInput"] > div {{ border: 2px solid #1b8a9a !important;
    box-shadow: 0 4px 16px rgba(27,138,154,.18); }}
[data-testid="stChatInput"] textarea {{ background: #ffffff !important; color: {INK} !important;
    -webkit-text-fill-color: {INK} !important; font-size: 1rem; }}
[data-testid="stChatInput"] textarea::placeholder {{ color: #7b8ea3 !important;
    -webkit-text-fill-color: #7b8ea3 !important; }}
[data-testid="stChatInput"] button svg {{ fill: #1b8a9a !important; color: #1b8a9a !important; }}

/* ---- 추천 질문 버튼 ---- */
.main div.stButton > button, [data-testid="stMain"] div.stButton > button {{
    border-radius: 14px; border: 1px solid #c6dcef; background: #ffffff !important;
    padding: 10px 12px; width: 100%; transition: all .15s;
}}
.main div.stButton > button *, [data-testid="stMain"] div.stButton > button * {{
    color: #134074 !important; font-weight: 500;
}}
[data-testid="stMain"] div.stButton > button:hover {{ border-color: #1b8a9a;
    transform: translateY(-1px); box-shadow: 0 4px 12px rgba(27,138,154,.18); }}

/* ---- 사이드바 ---- */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{ background: #0b2545 !important; }}
[data-testid="stSidebar"] * {{ color: #e8f1f8 !important; }}
[data-testid="stSidebar"] div.stButton > button {{
    background: rgba(255,255,255,.1) !important; border: 1px solid rgba(255,255,255,.3);
    border-radius: 12px; width: 100%;
}}
[data-testid="stSidebar"] .ph {{ border-radius: 50%; object-fit: cover;
    border: 2px solid rgba(255,255,255,.7); display: block; margin: 0 auto 8px; }}

/* 사이드바 펼침 상자(기억 방식, 과제 데이터) = 흰 카드 + 진한 글자 */
[data-testid="stSidebar"] [data-testid="stExpander"] {{ background: #ffffff !important;
    border: 1px solid #c6dcef !important; border-radius: 12px; margin-bottom: 10px; }}
[data-testid="stSidebar"] [data-testid="stExpander"] details,
[data-testid="stSidebar"] [data-testid="stExpander"] summary {{ background: #ffffff !important; border-radius: 12px; }}
[data-testid="stSidebar"] [data-testid="stExpander"] * {{ color: {INK} !important; }}
[data-testid="stSidebar"] [data-testid="stExpander"] svg {{ fill: {INK} !important; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{ background: #eef6fc !important;
    border: 1px dashed #1b8a9a !important; }}
[data-testid="stSidebar"] [data-testid="stExpander"] button {{ background: #ffffff !important;
    border: 1px solid #c6dcef !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{ background: #ffffff !important;
    border-color: #c6dcef !important; }}
[data-testid="stSidebar"] [data-baseweb="tag"] {{ background: #1b8a9a !important; }}
[data-testid="stSidebar"] [data-baseweb="tag"] * {{ color: #ffffff !important; fill: #ffffff !important; }}
[data-baseweb="popover"] li, [data-baseweb="popover"] li * {{ color: {INK} !important; }}

/* ---- 첫 화면 ---- */
.welcome {{ text-align: center; margin: 22px 0 14px; }}
.welcome, .welcome * {{ color: #4a6078 !important; }}
.welcome .ph {{ border-radius: 50%; object-fit: cover; border: 3px solid #1b8a9a;
                display: block; margin: 0 auto 10px; }}
.welcome b {{ color: #134074 !important; }}


/* 펼침 상자(과거 기록) */
[data-testid="stExpander"] {{ background: #ffffff !important; border-radius: 12px; }}
[data-testid="stExpander"] * {{ color: {INK} !important; }}

/* 오류 메시지 */
[data-testid="stAlert"] * {{ color: #7a1d14 !important; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------ Ollama 연결 확인
try:
    models = [m["name"] for m in requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()["models"]]
    status, connected = '<span class="badge ok">● 연결됨</span>', True
except Exception:
    models, connected = [], False
    status = '<span class="badge err">● Ollama 꺼져 있음</span>'

st.markdown(f"""
<div class="hero">
  {photo_html(92)}
  <div>
    <div class="title">{PAGE_NAME}</div>
    <div class="sub">{BOT_NAME} · {PERSONA}</div>
    <div class="badges">
      {status}
      <span class="badge">🧠 {MODEL}</span>
      <span class="badge">💻 내 PC에서 실행</span>
      <span class="badge">🔒 인터넷 불필요</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not connected:
    st.error("Ollama가 실행 중이 아닙니다. 시작 메뉴에서 Ollama를 켜고 F5로 새로고침하세요.")
    st.stop()


# ------------------------------------------------------------ 대화 기록 불러오기
if "messages" not in st.session_state:
    st.session_state.messages = load_history()       # 저장된 대화 불러오기
    rag_rebuild_if_needed(st.session_state.messages)
msgs = st.session_state.messages

UPLOAD_DIR = os.path.join(HERE, "data", "uploads")


@st.cache_resource(show_spinner=False)
def table_info(path, mtime):
    h, rows = A.read_table(path)
    return h, A.numeric_columns(h, rows), len(rows)


def uploaded_file(key, label):
    """파일 올리기 → data/uploads 에 저장하고 경로 반환 (과제별로 따로 기억)"""
    up = st.file_uploader(label, type=["csv", "xlsx"], key=f"up_{key}")
    if up is not None and st.session_state.get(f"{key}_name") != up.name:
        folder = os.path.join(UPLOAD_DIR, key)            # 과제1·2 파일을 폴더로 구분
        os.makedirs(folder, exist_ok=True)
        dest = os.path.join(folder, os.path.basename(up.name))
        with open(dest, "wb") as f:
            f.write(up.getbuffer())
        st.session_state[f"{key}_name"] = up.name
        st.session_state[f"{key}_path"] = dest
    path = st.session_state.get(f"{key}_path")
    return path if path and os.path.exists(path) else None


def reset_upload(key):
    for k in (f"{key}_path", f"{key}_name"):
        st.session_state.pop(k, None)
    st.rerun()


def valid_table(path):
    """올린 파일을 읽어서 (숫자 열 목록, 행 수). 쓸 수 없으면 None"""
    if not path:
        return None
    try:
        _, nums, nrows = table_info(path, os.path.getmtime(path))
    except Exception as e:
        st.error(f"파일을 읽지 못했어요: {e}")
        return None
    if len(nums) < 2:
        st.warning("숫자로 된 열이 2개 이상 있어야 해요. 기본 예시 데이터를 쓸게요.")
        return None
    return nums, nrows


# ------------------------------------------------------------ 사이드바
with st.sidebar:
    st.markdown(photo_html(110), unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center;margin:0'>{BOT_NAME}</h2>"
                f"<p style='text-align:center;opacity:.75;margin-top:2px'>{PAGE_NAME} · 데이터처리개론</p>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🧠 기억 방식 설정"):
        use_full = st.toggle("① 전체 기억", value=True, help="나눈 대화를 매번 통째로 전달")
        use_memo = st.toggle("② 메모 기억", value=True, help="중요한 내용을 메모로 정리해 두고 매번 읽기")
        use_rag = st.toggle("③ RAG 검색", value=True, help="전체 기록에서 관련 대화를 찾아서 전달")
    # ---------- 과제 1·2를 따로 설정 (각자 파일 올리기 + 열 고르기)
    a1_path = a2_path = None                 # None = 기본 예시 데이터
    a1cols = a2cols = None
    side_hp = {}
    if A is not None:
        # ===== 과제1 · 회귀분석
        with st.expander("📈 과제1 · 회귀분석"):
            a1_path = uploaded_file("a1", "데이터 파일 (CSV · xlsx)")
            info = valid_table(a1_path)
            if info:
                nums, nrows = info
                base = os.path.basename(a1_path)
                st.caption(f"📄 {base} · {nrows:,}행")
                x1 = st.selectbox("X (독립변수)", nums, index=0, key=f"a1x_{base}")
                y1 = st.selectbox("Y (종속변수)", [c for c in nums if c != x1], index=0, key=f"a1y_{base}_{x1}")
                a1cols = (x1, y1)
                if st.button("↩ 기본 예시로", key="reset_a1"):
                    reset_upload("a1")
            else:
                a1_path = None
                st.caption("기본 예시: 풍속(m/s) → 유의파고(m), 60개")
        # ===== 과제2 · 인공신경망
        with st.expander("🧠 과제2 · 인공신경망"):
            a2_path = uploaded_file("a2", "데이터 파일 (CSV · xlsx)")
            info = valid_table(a2_path)
            if info:
                nums, nrows = info
                base = os.path.basename(a2_path)
                st.caption(f"📄 {base} · {nrows:,}행")
                y2 = st.selectbox("예측할 열 (출력)", nums, index=len(nums) - 1, key=f"a2y_{base}")
                opts = [c for c in nums if c != y2]
                x2 = st.multiselect("입력 열", opts, default=opts, key=f"a2x_{base}_{y2}")
                if not x2:
                    st.warning("입력 열을 1개 이상 골라 주세요.")
                    x2 = opts
                a2cols = (x2, y2)
                if st.button("↩ 기본 예시로", key="reset_a2"):
                    reset_upload("a2")
            else:
                a2_path = None
                st.caption("기본 예시: 울산항 선박 길이·너비·깊이 → 총톤수, 54,368척")
            st.markdown("**신경망 설정**")
            auto_hp = st.toggle("기본값 사용", value=True,
                                help="선박 예시: 은닉 6·60 epoch·배치 256 / 내 데이터: 은닉 8·100 epoch·배치 64 (학습률 0.01)")
            if not auto_hp:
                d = A.GENERAL if a2_path else A.SHIP
                side_hp["hidden"] = st.slider("은닉층 노드 개수", 1, 16, d["hidden"])
                side_hp["epochs"] = st.number_input("epoch (반복 학습 횟수)", 1, 3000, d["epochs"], step=10)
                side_hp["lr"] = st.select_slider("학습률", [0.001, 0.003, 0.01, 0.03, 0.05, 0.1], value=d["lr"])
                side_hp["batch"] = st.select_slider("배치 크기", [16, 32, 64, 128, 256, 512], value=d["batch"])
            st.caption("채팅으로도 바꿀 수 있어요: \"은닉 노드 10개, 학습률 0.03, 에포크 200으로 신경망 돌려줘\"")
        if a1cols is None:                                    # 기본 예시 데이터의 열 이름
            _, n1, _ = table_info(A.A1_FILE, os.path.getmtime(A.A1_FILE))
            a1cols = (n1[0], n1[1])
        if a2cols is None:
            a2cols = (A.SHIP["x_cols"], A.SHIP["y_col"])
    if st.button("🧹 기억 전체 삭제"):
        st.session_state.messages = []
        for f, empty in ((HISTORY_FILE, []), (RAG_FILE, []), (PROFILE_FILE, {}), (MEMO_FILE, {})):
            _save(f, empty)
        st.rerun()


@st.cache_resource(show_spinner=False)
def get_a1(path, x_col, y_col, mtime):
    return A.train_assignment1(path, x_col, y_col)


@st.cache_resource(show_spinner=False)
def get_a2(path, x_cols, y_col, mtime, hp_items=()):
    hp = dict(hp_items)
    return A.train_assignment2(path, list(x_cols) if x_cols else None, y_col,
                               hp.get("hidden"), hp.get("epochs"), hp.get("batch"), hp.get("lr"))


def run_tasks(tasks):
    """과제 프로그램 실행 → [(요약 문자열, 그래프 경로 또는 None)]"""
    c1 = (a1_path, a1cols[0], a1cols[1]) if a1_path else (A.A1_FILE, None, None)
    c2 = (a2_path, tuple(a2cols[0]), a2cols[1]) if a2_path else (A.A2_FILE, None, None)
    out = []
    for t in tasks:
        if t[0] in ("a1", "p1"):
            with st.spinner("과제1 회귀분석 실행 중..."):
                res = get_a1(*c1, os.path.getmtime(c1[0]))
            out.append((res["summary"], res["image"]) if t[0] == "a1"
                       else (A.predict_assignment1(res, t[1]), None))
        else:
            hp = {**side_hp, **(t[1] if t[0] == "a2" else {})}   # 채팅에서 말한 값 > 사이드바 값
            with st.spinner("과제2 인공신경망 학습 중... (처음 한 번은 시간이 좀 걸려요)"):
                res = get_a2(*c2, os.path.getmtime(c2[0]), tuple(sorted(hp.items())))
            out.append((res["summary"], res["image"]) if t[0] == "a2"
                       else (A.predict_assignment2(res, t[1]), None))
    return out


def show_graph(path):
    """과제 그래프(SVG가 담긴 html 파일)를 채팅에 표시"""
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)


def avatar(role):
    return BOT_AVATAR if role == "assistant" else USER_AVATAR


for m in msgs:
    with st.chat_message(m["role"], avatar=avatar(m["role"])):
        st.markdown(m["content"])
        for img in m.get("images", []):
            show_graph(os.path.join(HERE, img))

# ------------------------------------------------------------ 첫 화면: 추천 질문
picked = None
if len(msgs) == 0:
    st.markdown(f"""<div class="welcome">{photo_html(84)}
        안녕하세요! 저는 <b>{BOT_NAME}</b>예요. 무엇이든 물어보세요.<br>
        우리가 나눈 대화는 전부 기억해 둘게요.</div>""", unsafe_allow_html=True)
    suggestions = (["과제1 회귀분석 실행해줘", "과제2 인공신경망 실행해줘",
                    "풍속 12m/s면 유의파고는 얼마야?", "길이 200 너비 32 깊이 18이면 총톤수는?"]
                   if not (a1_path or a2_path) else
                   ["과제1 회귀분석 실행해줘", "과제2 인공신경망 실행해줘",
                    "과제 둘 다 실행해줘", "선형회귀랑 인공신경망 차이가 뭐야?"])
    cols = st.columns(2)
    for i, q in enumerate(suggestions):
        if cols[i % 2].button(q, key=f"sug{i}"):
            picked = q

# ------------------------------------------------------------ 질문 → 기억 검색 → 답변
prompt = st.chat_input(f"{BOT_NAME}에게 메시지 보내기") or picked
if prompt:
    q_idx = len(msgs)
    update_profile(prompt)                            # 사용자 이름·전공 메모
    msgs.append({"role": "user", "content": prompt})
    start = history_start(msgs, use_full)             # ① 그대로 넣을 대화 범위
    hits = rag_search(prompt, start) if use_rag else []   # ③ 범위 밖 과거 대화 검색
    memo = load_memo() if use_memo else ""            # ② 정리해 둔 메모
    save_history(msgs)
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        tasks = detect_tasks(prompt, a1cols, a2cols) if (A and not rule_answer(prompt)) else []
        tool_text, images = "", []
        task_error = ""
        if tasks and A is None:
            task_error = f"과제 프로그램을 불러오지 못했어요: {ASSIGN_ERR}"
        elif tasks:
            try:
                results = run_tasks(tasks)
            except Exception as e:
                task_error = f"과제 프로그램 실행 중 오류가 났어요: {e}"
        if task_error:                               # 실패하면 LLM이 지어내지 않도록 여기서 끝냄
            st.error(task_error)
            msgs.append({"role": "assistant", "content": "⚠️ " + task_error})
            save_history(msgs)
            st.stop()
        if tasks:
            for summ, img in results:                # 내 과제 코드 실행 결과
                st.markdown(summ)
                if img:
                    show_graph(img)
                    images.append(os.path.relpath(img, HERE))
                tool_text += summ + "\n\n"
        box = st.empty()
        box.markdown(f"⏳ {BOT_NAME}이(가) 생각 중이에요... (첫 질문은 모델을 불러오느라 30초 이상 걸릴 수 있어요)")
        answer, thinking = "", False
        fixed = rule_answer(prompt)
        if fixed:                                       # 이름 질문 → 프로그램이 바로 답함
            box.markdown(fixed)
            answer = fixed
        try:
            if fixed:
                raise StopIteration
            # 글자가 만들어지는 대로 바로 화면에 표시 (ChatGPT 방식)
            with requests.post(f"{OLLAMA_URL}/api/chat",
                               json={"model": MODEL,
                                     "messages": build_context(msgs, start, hits, memo, tool_text),
                                     "stream": True,
                                     "options": {**GEN_OPTIONS, "num_ctx": NUM_CTX},
                                     "keep_alive": "30m"},   # 모델을 30분간 메모리에 유지
                               stream=True, timeout=600) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data:                      # Ollama가 보낸 오류
                        raise RuntimeError(data["error"])
                    piece = data.get("message", {}).get("content", "")
                    # DeepSeek-R1의 생각 과정(<think>...</think>)은 숨김
                    if "<think>" in piece:
                        thinking = True
                    if thinking:
                        if "</think>" in piece:
                            thinking = False
                            piece = piece.split("</think>")[-1]
                        else:
                            continue
                    answer += piece
                    box.markdown(answer + " ▌")
            box.markdown(answer or "(빈 답변이 왔어요. 다시 질문해 주세요.)")
        except StopIteration:
            pass
        except Exception as e:
            box.empty()
            st.error(f"답변 오류: {e}\n\n→ cmd에서 `ollama run {MODEL}` 로 대화가 되는지 확인해 주세요.")
            msgs.pop()                                   # 실패한 질문은 기록에서 제거
            save_history(msgs)
            st.stop()
        if not fixed and (memo or hits or start > 0):
            with st.expander(f"🔎 {BOT_NAME}이(가) 참고한 기억"):
                st.markdown(f"**① 전체 기억** · 대화 {len(msgs) - start}개 메시지 전달"
                            + (f" (오래된 {start}개는 ③이 담당)" if start else ""))
                if memo:
                    st.markdown("**② 메모 기억**")
                    st.caption(memo)
                for h in hits:
                    st.markdown(f"**③ RAG 검색** · 유사도 {h['score']} · {h['time']}")
                    st.caption(h["text"])

    if tool_text:                                         # 과제 결과도 대화 기록·기억에 저장
        answer = tool_text + answer
    msgs.append({"role": "assistant", "content": answer, **({"images": images} if images else {})})
    save_history(msgs)
    rag_add(prompt, answer, q_idx)                       # ③ 이번 질문·답변도 검색용으로 저장
    if use_memo and not fixed:
        update_memo_background(prompt, answer)           # ② 메모는 뒤에서 정리
    if picked or len(msgs) == 2:      # 첫 대화 뒤에는 환영 문구·추천 버튼 정리
        st.rerun()
