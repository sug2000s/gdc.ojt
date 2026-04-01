from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from constants import trip_intake_strong_confirm
from graph import LLM


class FoodChainResult(BaseModel):
    """Step 1 — food / dining only."""

    summary: str | None = Field(
        default=None,
        description=(
            "Short food or dining preference in English only (dishes, markets, dietary notes). "
            "Null if the message has no food-related content."
        ),
    )
    vague_ok: bool = Field(
        default=False,
        description="True when the user defers choice (e.g. anything, surprise me, up to you, no preference).",
    )


class PlacesChainResult(BaseModel):
    """Step 2 — places / sights only."""

    summary: str | None = Field(
        default=None,
        description=(
            "Short list of neighborhoods, landmarks, or areas to visit in English only. "
            "Null if the message has no place-related content."
        ),
    )
    vague_ok: bool = Field(
        default=False,
        description="True when the user defers on where to go.",
    )


def _food_to_slot_value(result: FoodChainResult) -> str | None:
    if result.vague_ok:
        return "ANY"
    if result.summary and result.summary.strip():
        return result.summary.strip()
    return None


def _places_to_slot_value(result: PlacesChainResult) -> str | None:
    if result.vague_ok:
        return "ANY"
    if result.summary and result.summary.strip():
        return result.summary.strip()
    return None


def chain_step_detect_food(user_text: str, slot_context: str) -> str | None:
    """Prompt chain step 1: food / dining only (structured output)."""
    if LLM is None:
        return None
    llm = LLM.with_structured_output(FoodChainResult)
    system = (
        "You are step 1 in a two-step Seoul travel intake chain. "
        "Focus ONLY on food: dishes, restaurants, street food, markets to eat at, BBQ, dietary notes. "
        "Write summary in concise English even if the user wrote another language. "
        "Set vague_ok=true if the user has no specific food wish (e.g. anything, surprise me). "
        "Leave summary null if there is no food-related content. "
        "Do not treat sightseeing neighborhoods as food unless they clearly describe where to eat."
    )
    user = f"What we are collecting now: {slot_context}\nUser message:\n{user_text}"
    try:
        parsed = llm.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=user),
            ]
        )
        return _food_to_slot_value(parsed)
    except Exception:
        return None


def chain_step_detect_places(
    user_text: str, slot_context: str, food_step_summary: str | None
) -> str | None:
    """Prompt chain step 2: places / sights only (structured output)."""
    if LLM is None:
        return None
    llm = LLM.with_structured_output(PlacesChainResult)
    food_hint = food_step_summary if food_step_summary else "(none extracted in step 1)"
    system = (
        "You are step 2 in a two-step Seoul travel intake chain. "
        "Use the step 1 food summary only to avoid listing pure food items as places. "
        "Focus ONLY on places: neighborhoods, districts, palaces, museums, parks, shopping areas, landmarks. "
        "Write summary in concise English even if the user wrote another language. "
        "Set vague_ok=true if the user defers on places. "
        "Leave summary null if there is no place-related content. "
        "Do not duplicate food-only items as place names unless they are real districts."
    )
    user = (
        f"What we are collecting now: {slot_context}\n"
        f"Step 1 food summary: {food_hint}\n"
        f"User message:\n{user_text}"
    )
    try:
        parsed = llm.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=user),
            ]
        )
        return _places_to_slot_value(parsed)
    except Exception:
        return None


def run_food_intake_chain_only(user_text: str, pending_slot: str | None) -> str | None:
    """LLM food extraction only (when collecting food — no places step)."""
    if trip_intake_strong_confirm(user_text):
        return None
    return chain_step_detect_food(user_text, pending_slot or "food preference")


def run_places_intake_chain_only(
    user_text: str, pending_slot: str | None, food_hint: str | None
) -> str | None:
    """LLM places extraction only (when collecting places — no food step)."""
    if trip_intake_strong_confirm(user_text):
        return None
    hint = food_hint if (food_hint or "").strip() else None
    return chain_step_detect_places(user_text, pending_slot or "places", hint)

