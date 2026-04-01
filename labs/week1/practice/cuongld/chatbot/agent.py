from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from langgraph.types import Command

from constants import (
    MSG_ITINERARY_LIMIT_EXCEEDED,
    MSG_OUT_OF_SCOPE,
    MSG_PLACES_OUTSIDE_SEOUL,
    MSG_TRIP_HITL_DENIED,
)
from graph import graph

InterruptProfile = Literal["none", "hitl", "pick"]


def _has_slot_value(value: object) -> bool:
    """True if a slot value is meaningfully set (not empty/ANY)."""
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.upper() != "ANY"


def _state_slots(values: dict) -> tuple[bool, bool, bool]:
    """Return (has_days, has_food, has_places) from checkpoint values."""
    has_days = bool(values.get("days"))
    has_food = _has_slot_value(values.get("food"))
    has_places = _has_slot_value(values.get("places"))
    return has_days, has_food, has_places


def _extract_interrupt_payload(result: dict) -> dict | str | None:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    return first.value if hasattr(first, "value") else first


def _next_actions_for_interrupt(payload: dict | str | None) -> list[str]:
    if isinstance(payload, dict):
        return [
            "Reply **confirm**, **ok**, or **yes** to generate the full itinerary.",
            "To edit first, send a new message with updated **days**, **food**, or **places** — the review will show again.",
        ]
    if isinstance(payload, str):
        if "Here are some options:" in payload or "Reply with a number" in payload:
            return [
                "Send a **number** (1–5) to pick from the list, or type a **short preference** in your own words.",
                "If **days** or the other slot are still missing, add them in your next message (see “Still needed” if shown).",
            ]
        return [
            "Reply in the chat using what the message above asks for (number, confirm, or free text).",
        ]
    return []


def _next_actions_for_result(result: dict, state_values: dict) -> list[str]:
    text = (result.get("final_response") or "").strip()
    has_days, has_food, has_places = _state_slots(state_values)
    if result.get("plan_generated"):
        rows = [
            "Refine one day: e.g. *“Make Day 2 slower, fewer museums.”*",
            "Start a **new plan** with days + food + places in one message.",
        ]
        if not has_food:
            rows.append("**Food ideas:** *Suggest food for Seoul*")
        if not has_places:
            rows.append("**Places:** *Suggest places to visit*")
        return rows
    if "Still needed:" in text:
        # Keep generic guidance, but avoid pushing food suggestions when food is already set.
        rows = ["Add **trip length** (e.g. `3 days in Seoul` or `weekend`)."] if not has_days else []
        if not has_food and not has_places:
            rows.append("For **food** or **places**, name specifics or say *anything* to get suggestions.")
        elif not has_food:
            rows.append("Add **food** preferences or ask for **food suggestions**.")
        elif not has_places:
            rows.append("Add **places** (neighborhoods, sights) or ask for **place suggestions**.")
        return rows or [
            "For **food** or **places**, name specifics or say *anything* to get suggestions.",
        ]
    if "Saved your food preference" in text:
        rows: list[str] = []
        if not has_places:
            rows.append("Add **places** (neighborhoods, sights) or ask for **place suggestions**.")
        if not has_days:
            rows.append("Set **how many days** if you have not yet.")
        if has_days and has_food and has_places:
            rows.append("Reply **ok** to open the review screen, then **confirm** to generate the itinerary.")
        return rows or ["Continue with **days**, **food**, and **places** to build an itinerary."]
    if "Saved your place preference" in text:
        rows: list[str] = []
        if not has_food:
            rows.append("Add **food** preferences or ask for **food suggestions**.")
        if not has_days:
            rows.append("Set **how many days** if you have not yet.")
        if has_days and has_food and has_places:
            rows.append("Reply **ok** to open the review screen, then **confirm** to generate the itinerary.")
        return rows or ["Continue with **days**, **food**, and **places** to build an itinerary."]
    if MSG_OUT_OF_SCOPE in text:
        return [
            "Ask about a **Seoul-only** trip: days, food, neighborhoods, or sights.",
            "Example: `4 days Seoul — BBQ, Bukchon, Han River parks`",
        ]
    if MSG_PLACES_OUTSIDE_SEOUL in text:
        return [
            "Replace with **areas inside Seoul** (e.g. Hongdae, Myeongdong, Gangnam, Jongno, Itaewon).",
            "You can still mix **food** + **places** in one follow-up message.",
        ]
    if MSG_ITINERARY_LIMIT_EXCEEDED in text:
        return [
            "Send again with **7 days or fewer** in Seoul.",
        ]
    if MSG_TRIP_HITL_DENIED in text:
        return [
            "Reply **confirm** / **ok** / **yes** to generate, or edit **days / food / places** and send a new message.",
        ]
    if "LLM not configured" in text:
        return [
            "Set **OPENAI_API_KEY** and **OPENAI_MODEL_NAME** in `.env`, then restart the app.",
        ]
    return [
        "Continue with **days**, **food**, and **places** to build an itinerary.",
        "Check the **sidebar** for example prompts and checkpoint tools.",
    ]


@dataclass(frozen=True)
class AgentReply:
    text: str
    next_actions: tuple[str, ...] = ()


@dataclass
class ChatbotAgent:
    system_prompt: str
    thread_id: str = field(default_factory=lambda: str(uuid4()))
    waiting_for_human_review: bool = False
    pending_interrupt_profile: InterruptProfile = "none"

    def _config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def _sync_interrupt_flags(self, result: object) -> None:
        if not isinstance(result, dict):
            self.waiting_for_human_review = False
            self.pending_interrupt_profile = "none"
            return
        payload = _extract_interrupt_payload(result)
        if payload is not None:
            self.waiting_for_human_review = True
            self.pending_interrupt_profile = "hitl" if isinstance(payload, dict) else "pick"
        else:
            self.waiting_for_human_review = False
            self.pending_interrupt_profile = "none"

    def _checkpoint_values(self) -> dict:
        try:
            snap = graph.get_state(self._config())
            values = snap.values
            return values if isinstance(values, dict) else {}
        except Exception:
            return {}

    def get_latest_plan_markdown(self) -> str | None:
        """
        Return the latest generated itinerary markdown from the graph state, if available.
        """
        v = self._checkpoint_values()
        if v.get("plan_generated"):
            text = (v.get("final_response") or "").strip()
            return text or None
        return None

    def get_latest_final_response_markdown(self) -> str | None:
        """
        Return the latest final_response markdown from the graph state (even if plan_generated is False).
        Useful for UI export buttons that should enable as soon as the itinerary text exists.
        """
        v = self._checkpoint_values()
        text = (v.get("final_response") or "").strip()
        return text or None

    def get_latest_itinerary_markdown(self) -> str | None:
        """
        Return the latest itinerary markdown if the graph state's final_response looks like an itinerary.
        This avoids exporting intermediate messages such as "Saved your food preference ...".
        """
        v = self._checkpoint_values()
        text = (v.get("final_response") or "").strip()
        if not text:
            return None
        if "# Your Seoul Itinerary" in text or "## Day " in text:
            return text
        return None

    def get_chat_input_hints(self) -> list[dict[str, str]]:
        """Short copy-paste lines for the chat input (English UI copy)."""
        v = self._checkpoint_values()
        waiting = self.waiting_for_human_review
        profile = self.pending_interrupt_profile

        if waiting:
            if profile == "hitl":
                return [
                    {
                        "paste": "confirm",
                        "description": "Confirm the trip summary to generate the full itinerary.",
                    },
                    {
                        "paste": "ok",
                        "description": "Same as confirm — **ok** or **yes** also work.",
                    },
                ]
            if profile == "pick":
                rows: list[dict[str, str]] = [
                    {
                        "paste": "1",
                        "description": "Pick a number from the list (1–5) or type the option name.",
                    },
                ]
                days = v.get("days")
                food = (v.get("food") or "").strip()
                places = (v.get("places") or "").strip()
                if not days:
                    rows.append({
                        "paste": "3 days in Seoul",
                        "description": "After this pick, add trip length in a follow-up message if it is still missing.",
                    })
                if not food:
                    rows.append({
                        "paste": "Suggest food for Seoul",
                        "description": "Later, if food is still missing, ask for food suggestions.",
                    })
                if not places:
                    rows.append({
                        "paste": "Suggest places to visit in Seoul",
                        "description": "Later, if places are still missing, ask for place suggestions.",
                    })
                return rows
            return [
                {
                    "paste": "ok",
                    "description": "Reply as instructed in the message above.",
                },
            ]

        if v.get("plan_generated"):
            has_days, has_food, has_places = _state_slots(v)
            hints: list[dict[str, str]] = [
                {
                    "paste": "4 days in Seoul — BBQ, Bukchon, Hangang parks",
                    "description": "Example to start a new trip.",
                },
            ]
            if not has_food:
                hints.append({
                    "paste": "Suggest food for Seoul",
                    "description": "Ask for food ideas only.",
                })
            if not has_places:
                hints.append({
                    "paste": "Suggest places to visit in Seoul",
                    "description": "Ask for sightseeing / neighborhood ideas only.",
                })
            return hints

        days = v.get("days")
        food = (v.get("food") or "").strip()
        places = (v.get("places") or "").strip()
        hints: list[dict[str, str]] = []

        if not days:
            hints.append({
                "paste": "3 days in Seoul",
                "description": "Trip length: 1–7 days, or say **weekend**.",
            })
        if not food:
            hints.append({
                "paste": "Suggest food for Seoul",
                "description": "Not sure what to eat? Get a short list of dishes to pick from.",
            })
        if not places:
            hints.append({
                "paste": "Suggest places to visit in Seoul",
                "description": "Not sure where to go? Get a short list of areas or sights to pick from.",
            })
        if days and food and places:
            hints.append({
                "paste": "ok",
                "description": (
                    "Days, food, and places are set — send this to open the review screen, "
                    "then reply **confirm** to generate the itinerary."
                ),
            })
        if hints:
            return hints

        return [
            {
                "paste": "Seoul trip, 3 days — street food and palaces",
                "description": "Example starter message.",
            },
        ]

    def _format_interrupt_message(self, payload: dict | str | None) -> str:
        if isinstance(payload, dict):
            title = payload.get("title", "Review before continuing")
            draft = payload.get("draft", "")
            instruction = payload.get("instruction", "Reply with ok or confirm to continue.")
            return f"{title}\n\n{draft}\n\n{instruction}"
        if isinstance(payload, str):
            return payload
        return "Waiting for your reply to continue."

    def reply(self, user_input: str, history: list[dict]) -> AgentReply:
        if self.waiting_for_human_review:
            result = graph.invoke(Command(resume=user_input), config=self._config())
        else:
            result = graph.invoke({"user_input": user_input}, config=self._config())

        if not isinstance(result, dict):
            self._sync_interrupt_flags(result)
            return AgentReply(
                text="Unexpected response from agent.",
                next_actions=("Try sending your message again.", "Use **Clear chat** in the sidebar if the session seems stuck."),
            )

        self._sync_interrupt_flags(result)
        interrupt_payload = _extract_interrupt_payload(result)
        if interrupt_payload is not None:
            body = self._format_interrupt_message(interrupt_payload)
            actions = _next_actions_for_interrupt(interrupt_payload)
            return AgentReply(text=body, next_actions=tuple(actions))

        body = (
            result.get("final_response", "")
            or "I could not generate a response. Please try again."
        )
        actions = _next_actions_for_result(result, self._checkpoint_values())
        return AgentReply(text=body, next_actions=tuple(actions))

    def get_state_history(self, limit: int = 20) -> list[dict]:
        snapshots = list(graph.get_state_history(self._config(), limit=limit))
        rows: list[dict] = []
        for idx, snapshot in enumerate(snapshots):
            cfg = snapshot.config.get("configurable", {})
            values = snapshot.values or {}
            summary = ""
            if isinstance(values, dict):
                summary = str(values.get("final_response") or values.get("user_input", ""))[:80]
            rows.append({
                "index": str(idx),
                "checkpoint_id": str(cfg.get("checkpoint_id", "")),
                "checkpoint_ns": str(cfg.get("checkpoint_ns", "")),
                "next": ", ".join(snapshot.next) if snapshot.next else "END",
                "summary": summary,
            })
        return rows

    def fork_from_checkpoint(
        self, checkpoint_id: str, checkpoint_ns: str, new_user_input: str
    ) -> str:
        fork_config = graph.update_state(
            {
                "configurable": {
                    "thread_id": self.thread_id,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_ns": checkpoint_ns,
                }
            },
            {"user_input": new_user_input},
            as_node="guardrail",
        )
        result = graph.invoke(None, config=fork_config)
        self._sync_interrupt_flags(result)
        interrupt_payload = _extract_interrupt_payload(result)
        if interrupt_payload is not None:
            return self._format_interrupt_message(interrupt_payload)
        return (
            result.get("final_response", "Fork completed.")
            if isinstance(result, dict)
            else "Fork completed."
        )
