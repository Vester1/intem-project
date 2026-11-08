import os
from dataclasses import dataclass, field
from pathlib import Path


def load_local_env(path):
    """Загружает значения INTEM_* как текст; переменные окружения имеют приоритет. Содержимое .env не исполняется."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key.startswith("INTEM_") or not key.replace("_", "").isalnum():
            raise ValueError("Invalid INTEM configuration in .env")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '\"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    mode: str = "demo"
    db: str = "var/intem.sqlite3"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = field(default="", repr=False)
    model: str = ""
    embedding_model: str = ""
    request_timeout: float = 120
    provider_timeout: float = 45
    max_llm_calls: int = 4
    reasoning_effort: str = "low"
    max_completion_tokens: int = 3000

    @classmethod
    def from_env(cls, env_file=None):
        load_local_env(env_file if env_file is not None else Path(__file__).resolve().parent.parent / ".env")
        settings = cls(mode=os.getenv("INTEM_MODE", "demo"), db=os.getenv("INTEM_DB", "var/intem.sqlite3"),
                       base_url=os.getenv("INTEM_LLM_BASE_URL", "https://api.openai.com/v1"),
                       api_key=os.getenv("INTEM_LLM_API_KEY", ""), model=os.getenv("INTEM_LLM_MODEL", ""),
                       embedding_model=os.getenv("INTEM_EMBEDDING_MODEL", ""),
                       request_timeout=float(os.getenv("INTEM_REQUEST_TIMEOUT", "120")),
                       provider_timeout=float(os.getenv("INTEM_PROVIDER_TIMEOUT", "45")),
                       max_llm_calls=int(os.getenv("INTEM_MAX_LLM_CALLS", "4")),
                       reasoning_effort=os.getenv("INTEM_REASONING_EFFORT", "low"),
                       max_completion_tokens=int(os.getenv("INTEM_MAX_COMPLETION_TOKENS", "3000")))
        if settings.mode not in {"demo", "llm"}:
            raise ValueError("INTEM_MODE must be demo or llm")
        if settings.mode == "llm" and not (settings.api_key and settings.model):
            raise ValueError("LLM mode requires INTEM_LLM_API_KEY and INTEM_LLM_MODEL")
        if (settings.request_timeout <= 0 or settings.provider_timeout <= 0
                or settings.max_llm_calls < 1
                or settings.max_completion_tokens < 1):
            raise ValueError("Timeouts and budgets must be positive")
        if settings.reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("INTEM_REASONING_EFFORT must be low, medium or high")
        return settings
