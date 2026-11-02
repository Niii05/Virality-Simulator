"""Run a safe environment and Gemini connectivity preflight."""

import sys
from pathlib import Path
import shutil
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def report(label, passed, detail=""):
    state = "PASS" if passed else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"[{state}] {label}{suffix}")
    return passed


def main():
    checks_passed = True

    version_ok = sys.version_info >= (3, 10)
    checks_passed &= report(
        "Python version >= 3.10",
        version_ok,
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )

    expected_venv = (PROJECT_ROOT / "venv").resolve()
    active_venv = sys.prefix != sys.base_prefix and Path(sys.prefix).resolve() == expected_venv
    checks_passed &= report("Project .\\venv active", active_venv)

    import_results = {}
    for module in ("google.genai", "faster_whisper", "dotenv"):
        try:
            __import__(module)
        except Exception as error:
            import_results[module] = False
            checks_passed &= report(f"Import {module}", False, str(error))
        else:
            import_results[module] = True
            checks_passed &= report(f"Import {module}", True)

    tools_ok = True
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool) is not None
        tools_ok &= found
        checks_passed &= report(f"{tool} on PATH", found)
    if not tools_ok:
        print("Install hint: winget install Gyan.FFmpeg, then restart VS Code")

    env_path = PROJECT_ROOT / ".env"
    env_exists = env_path.is_file()
    checks_passed &= report(".env exists", env_exists)
    env_status = {"GEMINI_API_KEY": "empty", "GEMINI_MODEL": "empty"}
    if env_exists:
        try:
            from dotenv import dotenv_values

            parsed_env = dotenv_values(env_path)
            for name in env_status:
                if parsed_env.get(name):
                    env_status[name] = "set"
        except Exception:
            pass
    for name, state in env_status.items():
        non_empty = state == "set"
        checks_passed &= report(name, non_empty, state)

    ignore_result = subprocess.run(
        ["git", "check-ignore", "--quiet", ".env"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    ignored = ignore_result.returncode == 0
    checks_passed &= report("git check-ignore confirms .env is ignored", ignored)

    if all(import_results.values()):
        try:
            from core.llm import _timeout_seconds, ask

            ask("Reply with only OK.", retries=1)
        except Exception as error:
            if "timed out" in str(error).lower() or "timeout" in type(error).__name__.lower():
                detail = (
                    f"Gemini request timed out after {_timeout_seconds():g} seconds. "
                    "Check your network/VPN."
                )
            else:
                detail = str(error)
            checks_passed &= report("Gemini connectivity", False, detail)
        else:
            checks_passed &= report("Gemini connectivity", True, "OK")
    else:
        checks_passed &= report(
            "Gemini connectivity", False, "Required imports are unavailable."
        )

    print("Preflight complete: " + ("PASS" if checks_passed else "FAIL"))
    return 0 if checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
