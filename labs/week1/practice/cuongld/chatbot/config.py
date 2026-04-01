from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


def _get_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    return value if value > 0 else default


@dataclass(frozen=True)
class Settings:
    chatbot_title: str = os.getenv("CHATBOT_TITLE", "Seoul Travel Agent")
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        "You are a specialized Seoul travel planner. Focus on Seoul attractions, itineraries, food, shopping, and practical transit tips.",
    )
    max_history: int = _get_positive_int("MAX_HISTORY", 20)


def get_settings() -> Settings:
    return Settings()
