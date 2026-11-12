#! python3
"""
JO-PT 실행기 — 더블클릭(윈도우) 또는 `python3 JO-PT_run.py`(맥)로 실행하세요.

처음 실행하면 필요한 것을 자동으로 준비합니다.
  1) 파이썬 패키지(streamlit, requests) 설치
  2) Ollama가 켜져 있는지 확인하고, 꺼져 있으면 켜기
  3) LLM 모델(EXAONE 3.5 2.4B, 약 1.6GB)이 없으면 내려받기
  4) 채팅 페이지 JO-PT 실행 → 브라우저가 열림
윈도우에서는 바탕화면 'JO-PT 실행' 파일과 cmd 명령어 'jopt'도 만들어져서 다음부터 더 쉽게 켤 수 있어요.

미리 설치해야 하는 것은 두 가지뿐입니다.
  - Python 3.10 ~ 3.13 : https://www.python.org/downloads  (윈도우는 설치할 때 "Add python.exe to PATH" 체크)
  - Ollama             : https://ollama.com/download
"""
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "exaone3.5:2.4b"
OLLAMA_URL = "http://localhost:11434"
PACKAGES = ["streamlit", "requests"]


def say(msg):
    print(msg, flush=True)


def finish(code=0):
    input("\n창을 닫으려면 Enter를 누르세요.")
    sys.exit(code)


# ------------------------------------------------------------ 0. 알맞은 파이썬으로 실행
def pick_python():
    """너무 최신 파이썬(3.14 이상)은 패키지가 아직 지원하지 않을 수 있어서, 3.13이 있으면 그걸로 다시 실행"""
    if sys.version_info < (3, 10):
        say(f"Python {sys.version.split()[0]}은 너무 오래된 버전이에요. 3.10 ~ 3.13을 설치해 주세요.")
        say("→ https://www.python.org/downloads")
        finish(1)
    if sys.version_info >= (3, 14) and os.environ.get("JOPT_CHILD") != "1":
        for cmd in (["py", "-3.13"], ["py", "-3.12"], ["python3.13"], ["python3.12"]):
            try:
                if subprocess.run(cmd + ["-c", "pass"], capture_output=True).returncode == 0:
                    env = dict(os.environ, JOPT_CHILD="1")
                    sys.exit(subprocess.run(cmd + [os.path.abspath(__file__)], env=env).returncode)
            except FileNotFoundError:
                continue


# ------------------------------------------------------------ 1. 패키지 설치
def ensure_packages():
    missing = []
    for pkg in PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        say(f"[1/4] 파이썬 패키지 설치 중: {', '.join(missing)} (처음 한 번, 1~2분)")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--user", *missing])
        if r.returncode != 0:
            say("패키지 설치에 실패했어요. 인터넷 연결을 확인하고 다시 실행해 주세요.")
            finish(1)
    else:
        say("[1/4] 파이썬 패키지 확인 완료")


# ------------------------------------------------------------ 2. Ollama 켜기
def ollama_alive():
    try:
        urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=2)
        return True
    except Exception:
        return False


def find_ollama():
    exe = shutil.which("ollama")
    if exe:
        return exe
    for p in (os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
              "/usr/local/bin/ollama", "/opt/homebrew/bin/ollama",
              "/Applications/Ollama.app/Contents/Resources/ollama"):
        if os.path.exists(p):
            return p
    return None


def ensure_ollama():
    if ollama_alive():
        say("[2/4] Ollama 실행 확인 완료")
        return find_ollama()
    exe = find_ollama()
    if not exe:
        say("[2/4] Ollama가 설치되어 있지 않아요.")
        say("      설치 페이지를 열게요 → https://ollama.com/download")
        say("      설치한 다음 이 파일을 다시 실행해 주세요.")
        webbrowser.open("https://ollama.com/download")
        finish(1)
    say("[2/4] Ollama 켜는 중...")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    for _ in range(30):
        time.sleep(1)
        if ollama_alive():
            say("      Ollama 실행 완료")
            return exe
    say("Ollama를 켜지 못했어요. 시작 메뉴(맥은 응용 프로그램)에서 Ollama를 직접 켠 다음 다시 실행해 주세요.")
    finish(1)


# ------------------------------------------------------------ 3. 모델 내려받기
def ensure_model(exe):
    import json
    try:
        tags = json.load(urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=5))
        names = [m["name"] for m in tags.get("models", [])]
    except Exception:
        names = []
    if any(n.startswith(MODEL) for n in names):
        say(f"[3/4] LLM 모델({MODEL}) 확인 완료")
        return
    say(f"[3/4] LLM 모델({MODEL}, 약 1.6GB) 내려받는 중... 처음 한 번만, 인터넷 속도에 따라 몇 분 걸려요")
    try:                                                     # Ollama API로 내려받기 (진행률 표시)
        req = urllib.request.Request(OLLAMA_URL + "/api/pull", method="POST",
                                     data=json.dumps({"name": MODEL, "stream": True}).encode(),
                                     headers={"Content-Type": "application/json"})
        last = -1
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                d = json.loads(line or b"{}")
                if d.get("error"):
                    raise RuntimeError(d["error"])
                if d.get("total") and d.get("completed"):
                    pct = int(d["completed"] / d["total"] * 100)
                    if pct != last and pct % 5 == 0:
                        say(f"      {pct}%")
                        last = pct
        say("      모델 준비 완료")
    except Exception as e:
        say(f"모델을 내려받지 못했어요: {e}")
        say(f"cmd(맥은 터미널)에서 직접 받아도 돼요 →  ollama pull {MODEL}")
        finish(1)


# ------------------------------------------------------------ 윈도우 바로가기 (다음부터 쉽게)
def make_shortcuts():
    cmd_text = (
        "@echo off\r\n"
        "title JO-PT\r\n"
        f'cd /d "{HERE}"\r\n'
        'if not exist chat.py (echo chat.py not found. Run JO-PT_run.py again from the unzipped JO-PT folder. & pause & exit /b)\r\n'
        f'"{sys.executable}" "{os.path.join(HERE, "JO-PT_run.py")}"\r\n'
    )
    made = []
    targets = []
    apps = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps")
    if os.path.isdir(apps):                                   # 이 폴더는 원래 PATH에 들어 있음
        targets.append((os.path.join(apps, "jopt.cmd"), "cmd 명령어 jopt"))
    for desk in (os.path.join(os.environ.get("USERPROFILE", ""), "Desktop"),
                 os.path.join(os.environ.get("OneDrive", ""), "Desktop"),
                 os.path.join(os.environ.get("OneDrive", ""), "바탕 화면")):
        if os.path.isdir(desk):
            targets.append((os.path.join(desk, "JO-PT 실행.cmd"), "바탕화면 'JO-PT 실행'"))
            break
    for path, name in targets:
        try:
            old = open(path, encoding="mbcs").read() if os.path.exists(path) else None
            if old != cmd_text:
                with open(path, "w", encoding="mbcs", newline="") as f:
                    f.write(cmd_text)
                made.append(name)
        except Exception:
            pass
    return made


# ------------------------------------------------------------ 실행
if __name__ == "__main__":
    os.chdir(HERE)
    if not os.path.exists(os.path.join(HERE, "chat.py")):      # zip 안에서 바로 실행한 경우 등
        say("chat.py를 찾을 수 없어요.\n")
        say("→ zip 파일을 먼저 '압축 풀기' 한 다음, 풀린 폴더 안의 JO-PT_run.py를 실행해 주세요.")
        say(f"   (지금 실행된 위치: {HERE})")
        finish(1)
    pick_python()
    say("JO-PT 준비 중...\n")
    ensure_packages()
    exe = ensure_ollama()
    ensure_model(exe)
    if os.name == "nt":
        made = make_shortcuts()
        if made:
            say("      다음부터는 이렇게 켤 수 있어요: " + ", ".join(made))
    say("[4/4] JO-PT 실행! 브라우저가 열립니다. (이 창을 닫으면 JO-PT도 꺼져요)\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "chat.py", "--theme.base", "light"])
    finish(0)
