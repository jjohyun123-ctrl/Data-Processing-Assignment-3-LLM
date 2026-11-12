#! python3.13
"""
JO-PT 실행기 — 이 파일을 더블클릭하면 채팅창이 열립니다.
처음 한 번 실행하면 아래 두 가지가 자동으로 만들어져서, 다음부터는 더 쉽게 켤 수 있어요.
  1) 바탕화면의 'JO-PT 실행' 파일  → 더블클릭
  2) cmd 명령어 'jopt'            → 아무 폴더에서나 jopt 입력
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CMD_TEXT = (
    "@echo off\r\n"
    "title JO-PT\r\n"
    f'cd /d "{HERE}"\r\n'
    'if not exist chat.py (echo chat.py not found. Run JO-PT_run.py again from the unzipped JO-PT folder. & pause & exit /b)\r\n'
    "echo JO-PT starting... (close this window to stop)\r\n"
    "py -3.13 -m streamlit run chat.py --theme.base light\r\n"
    "pause\r\n"
)


def make_shortcuts():
    made = []
    targets = []
    apps = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps")
    if os.path.isdir(apps):                                  # 이 폴더는 원래 PATH에 들어 있음
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
            if old != CMD_TEXT:
                with open(path, "w", encoding="mbcs", newline="") as f:
                    f.write(CMD_TEXT)
                made.append(name)
        except Exception:
            pass
    return made


if __name__ == "__main__":
    os.chdir(HERE)
    if not os.path.exists(os.path.join(HERE, "chat.py")):      # zip 안에서 바로 실행한 경우 등
        print("chat.py를 찾을 수 없어요.\n")
        print("→ zip 파일을 먼저 '압축 풀기' 한 다음, 풀린 폴더 안의 JO-PT_run.py를 더블클릭해 주세요.")
        print(f"   (지금 실행된 위치: {HERE})")
        input("\n종료하려면 Enter를 누르세요.")
        sys.exit()
    if os.name == "nt":
        made = make_shortcuts()
        if made:
            print("다음부터는 이렇게 켤 수 있어요:", ", ".join(made))
    print("JO-PT 실행 중... 브라우저가 열립니다. (이 창을 닫으면 채팅도 꺼져요)\n")
    cmd = ["-m", "streamlit", "run", "chat.py", "--theme.base", "light"]
    try:
        import streamlit  # noqa: F401
        subprocess.run([sys.executable] + cmd)
    except ImportError:
        subprocess.run(["py", "-3.13"] + cmd)
    input("\n종료하려면 Enter를 누르세요.")
