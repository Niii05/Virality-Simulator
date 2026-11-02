"""Check whether the configured Gemini key and model are reachable."""
from core.llm import ask

if __name__ == "__main__":
    print(ask("Say hello in one line.", retries=1))
