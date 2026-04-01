from __future__ import annotations

import re

SEOUL_KEYWORDS = {
    "seoul",
    "korea",
    "gangnam",
    "jongno",
    "hongdae",
    "myeongdong",
    "south korea",
    "seoul trip",
}

DAY_PATTERN = r"(\d+)\s*[-]?\s*(day|days)"
ITINERARY_DAY_PATTERN = r"\d+\s*[-]?\s*(day|days)"

INTENT_KEYWORDS = {
    "itinerary": (
        "itinerary",
        "plan",
        "travel plan",
        "1 day",
        "1 days",
        "2 day",
        "2 days",
        "3 day",
        "3 days",
        "4 day",
        "4 days",
        "5 day",
        "5 days",
        "6 day",
        "6 days",
        "7 day",
        "7 days",
    ),
    "food": ("food", "eat", "restaurant", "market", "street food"),
    "night": ("night", "nightlife", "bar", "pub", "club"),
    "family": ("family", "kid", "kids", "children", "child-friendly"),
    "shopping": ("shopping", "cosmetics", "mall", "outlet"),
    "culture": ("history", "culture", "palace", "hanok", "museum"),
}

ITINERARY_HINT_KEYWORDS = (
    "plan",
    "trip",
    "suggest",
    "itinerary",
    "schedule",
)

BUDGET_KEYWORDS = {
    1: ("cheap", "budget", "low cost", "affordable"),
    3: ("luxury", "premium", "high end", "upscale"),
    2: ("mid", "moderate", "mid-range"),
}

INDOOR_KEYWORDS = ("rain", "rainy", "indoor", "inside")
FAMILY_KEYWORDS = ("family", "kid", "kids", "children", "child-friendly")

REFINE_QUERY_SUFFIX = "seoul itinerary food culture shopping"

DAY_EMOJIS = {1: "☀️", 2: "🌤️", 3: "⛅", 4: "🌥️", 5: "☁️", 6: "🌦️", 7: "🌧️"}
TIME_EMOJIS = {
    "Morning": "🌅",
    "Lunch": "🍽️",
    "Afternoon": "☀️",
    "Late Afternoon": "🌤️",
    "Evening": "🌆",
    "Sunset": "🌅",
    "Night": "🌙",
}

MSG_ITINERARY_LIMIT_EXCEEDED = (
    "This planner supports trips of up to 7 days in Seoul. "
    "Please choose a length of 7 days or fewer and try again."
)
MSG_ASK_ITINERARY_DAYS = (
    "Happy to plan that. How many days are you planning to stay in Seoul? "
    "(For example: 2 days or 3 days.)"
)
MSG_QUALITY_GATE_BROAD = "Let me dig a bit deeper to find you the best spots... 🔍"
MSG_CLARIFY = (
    "Tell me a bit more and I'll build the perfect Seoul trip for you! 🗺️\n"
    "For example: how many days, any vibe you're after (food, culture, nightlife), and roughly what budget?"
)
MSG_NO_MATCH = "Hmm, I couldn't find a great match for that — try telling me more about what you're looking for! 🤔"
MSG_OPTIMIZE_BY_DISTRICT = "Would you like me to optimize this by district (Hongdae, Myeongdong, Gangnam)?"

MSG_HUMAN_REVIEW_TITLE = "✨ How does this look?"
MSG_HUMAN_REVIEW_INSTRUCTION = "Type 'ok' if you like it, or share your feedback to improve it"

MSG_FINAL_APPROVED = "Great, glad you like it! 🎉 Let me know if you'd like to tweak anything."
MSG_FINAL_UPDATED_SUFFIX = "Just say the word if you'd like me to try a completely fresh version!"

MSG_OUT_OF_SCOPE = (
    "I only help with Seoul trips. Ask about neighborhoods, food, sights, "
    "shopping, or how to plan your time in Seoul."
)

MSG_PLACES_OUTSIDE_SEOUL = (
    "This planner only covers destinations in Seoul (the city). "
    "Other Korean cities (for example Busan or Jeju) or international destinations are not supported. "
    "Please list areas within Seoul such as Hongdae, Myeongdong, Gangnam, Jongno, or Itaewon."
)

# Multi-word places / regions outside Seoul (substring match, lowercase)
_OFF_SEOUL_PHRASES = (
    "new york",
    "los angeles",
    "san francisco",
    "ho chi minh",
    "hong kong",
    "jeju island",
    "great wall",
    "chiang mai",
)

# Single-token or obvious Latin names; word-boundary match on lowercase ASCII
_OFF_SEOUL_WORDS = frozenset(
    {
        "busan",
        "jeju",
        "daegu",
        "daejeon",
        "gwangju",
        "ulsan",
        "gangneung",
        "gangneong",
        "sokcho",
        "jeonju",
        "tongyeong",
        "pyeongchang",
        "andong",
        "yeosu",
        "mokpo",
        "chuncheon",
        "tokyo",
        "osaka",
        "kyoto",
        "nagoya",
        "fukuoka",
        "sapporo",
        "okinawa",
        "hiroshima",
        "beijing",
        "shanghai",
        "guangzhou",
        "shenzhen",
        "taipei",
        "bangkok",
        "phuket",
        "chiangmai",
        "singapore",
        "manila",
        "jakarta",
        "bali",
        "hanoi",
        "saigon",
        "paris",
        "london",
        "berlin",
        "rome",
        "madrid",
        "amsterdam",
        "sydney",
        "melbourne",
        "auckland",
    }
)

# Common non-Seoul names in Korean / Japanese / Chinese (substring on original text)
_OFF_SEOUL_HANGUL_KANA = (
    "부산",
    "제주",
    "대구",
    "대전",
    "광주",
    "울산",
    "강릉",
    "속초",
    "전주",
    "도쿄",
    "오사카",
    "京都",
    "北京",
    "上海",
)


def places_outside_seoul_gate(places_text: str) -> str | None:
    """Quality gate: block trip focus outside Seoul city. Returns message if blocked, else None."""
    if not places_text or not str(places_text).strip():
        return None
    raw = str(places_text)
    low = raw.lower()
    for phrase in _OFF_SEOUL_PHRASES:
        if phrase in low:
            return MSG_PLACES_OUTSIDE_SEOUL
    for token in sorted(_OFF_SEOUL_WORDS, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", low):
            return MSG_PLACES_OUTSIDE_SEOUL
    for chunk in _OFF_SEOUL_HANGUL_KANA:
        if chunk in raw:
            return MSG_PLACES_OUTSIDE_SEOUL
    return None


# Shared approve tokens for agent "confirm" and LangGraph HITL resume
HITL_APPROVE_KEYWORDS = frozenset(
    {
        "ok",
        "okay",
        "yes",
        "y",
        "confirm",
        "confirmed",
        "go",
        "continue",
        "네",
        "확인",
        "đồng ý",
        "dong y",
        "xac nhan",
        "được",
        "duoc",
    }
)

# "ok"/"okay" often mean "anything goes" during intake; do not treat as premature HITL.
HITL_INTAKE_DEFERRAL_TOKENS = frozenset({"ok", "okay"})


def trip_intake_strong_confirm(text: str) -> bool:
    """True if the message is a HITL-style approval but not a vague deferral (ok/okay)."""
    token = text.strip().lower().rstrip(".!?")
    return token in HITL_APPROVE_KEYWORDS and token not in HITL_INTAKE_DEFERRAL_TOKENS


def topic_outside_seoul_gate(user_text: str) -> str | None:
    """
    Detect explicit non-Seoul trip intent in free-form user text.
    Uses the same heuristics as places gate, but only blocks when the message
    does NOT also mention Seoul.
    """
    if not user_text or not str(user_text).strip():
        return None
    raw = str(user_text)
    low = raw.lower()
    if "seoul" in low:
        return None
    if places_outside_seoul_gate(raw):
        return MSG_OUT_OF_SCOPE
    return None


MSG_TRIP_HITL_TITLE = "Confirm before generating your itinerary"
MSG_TRIP_HITL_INSTRUCTION = (
    "This is the only confirmation step. Reply with confirm, ok, or yes to generate the full plan. "
    "To change days, food, or places, update them in the chat and send a new message so this review appears again."
)
MSG_TRIP_HITL_DENIED = (
    "The itinerary was not generated. Reply with confirm, ok, or yes to try again, "
    "or update your trip details in the chat first."
)

