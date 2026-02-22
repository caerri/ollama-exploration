import os
import platform
import shutil
import subprocess
import sys


def check_ollama() -> str:
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        return "NOT INSTALLED"

    try:
        result = subprocess.run(
            ["ollama", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        version = (result.stdout or result.stderr).strip()
        return f"INSTALLED ({version or 'version unknown'})"
    except Exception:
        return "INSTALLED (version check failed)"


def main() -> None:
    print("=" * 40)
    print("SYSTEM DETECTION REPORT")
    print("=" * 40)

    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Ollama Status: {check_ollama()}")

    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    print(f"Anthropic API Key Present: {'YES' if has_key else 'NO'}")


if __name__ == "__main__":
    main()
