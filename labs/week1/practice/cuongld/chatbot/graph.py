"""
Seoul Travel Agent — LangGraph graph definition.

Concepts demonstrated:
  Ch13 — MessagesState (Reducer: add_messages) + Annotated day_plans (operator.add)
  Ch14 — ToolNode (suggest_llm ↔ tool_executor; router pins food vs place tool on first hop)
  Ch14 — Human-in-the-loop (interrupt) + Time Travel (SQLite checkpointer)
  Ch16.1 — Prompt Chaining (extract_food → extract_places share food hint sequentially)
  Ch16.2 — Gate (guardrail blocks non-Seoul; slot_aggregator validates before HITL)
  Ch16.3 — Routing (router LLM classifies intent → different node paths)
  Ch16.4 — Parallelization (extract_food + extract_places fan-out/fan-in)
  Ch16.5 — Orchestrator-Workers + Send API (orchestrator fans out one day_worker per day)
"""

import ast
import operator
import os
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Send
from pydantic import BaseModel, Field

from constants import (
    DAY_PATTERN,
    HITL_APPROVE_KEYWORDS,
    MSG_ITINERARY_LIMIT_EXCEEDED,
    MSG_OUT_OF_SCOPE,
    MSG_TRIP_HITL_DENIED,
    MSG_TRIP_HITL_INSTRUCTION,
    MSG_TRIP_HITL_TITLE,
    SEOUL_KEYWORDS,
    places_outside_seoul_gate,
)
from tools import get_food_list, get_place_list


load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"))

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "seoul_agent_memory.db"
SQLITE_CONN = sqlite3.connect(DB_PATH, check_same_thread=False)


# ── LLM setup ─────────────────────────────────────────────────────────────────

def _build_llm() -> ChatOpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("OPENAI_MODEL_NAME", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not api_key or not model_name:
        return None
    kwargs: dict = {"api_key": api_key, "model": model_name}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(temperature=0.2, timeout=60, **kwargs)


LLM = _build_llm()

_SUGGESTION_TOOLS = [get_food_list, get_place_list]
# Suggest path: router intent selects get_food_list vs get_place_list (no LLM tool-choice on first hop).


def _to_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return ""


def _normalize(text: str) -> str:
    return text.strip().lower()


# ── State ─────────────────────────────────────────────────────────────────────
# Ch13 — MessagesState provides: messages: Annotated[list[AnyMessage], add_messages]
# Ch13 — day_plans uses operator.add reducer: each day_worker appends its result
class TravelState(MessagesState):
    user_input: str
    in_scope: bool
    intent: str
    days: int | None
    food: str | None
    places: str | None
    pre_plan_approved: bool
    day_plans: Annotated[list[str], operator.add]
    final_response: str
    plan_generated: bool


def _tool_call_name(tc: object) -> str:
    if isinstance(tc, dict):
        return str(tc.get("name") or "")
    return str(getattr(tc, "name", "") or "")


def _tool_call_id(tc: object) -> str:
    if isinstance(tc, dict):
        return str(tc.get("id") or "")
    return str(getattr(tc, "id", "") or "")


def _router_suggest_slot(state: TravelState) -> Literal["food", "places"]:
    """Slot implied by router intent (only suggest_food / suggest_places reach this node)."""
    intent_val = (state.get("intent") or "").strip()
    if intent_val == "suggest_food":
        return "food"
    return "places"


def _slot_from_last_tool_message(state: TravelState) -> Literal["food", "places"]:
    """Infer food vs places from the tool that actually produced the list (most accurate for UI copy)."""
    messages = state.get("messages") or []
    for m in reversed(messages):
        if getattr(m, "type", "") != "tool":
            continue
        tc_id = str(getattr(m, "tool_call_id", "") or "")
        for ai in reversed(messages):
            if getattr(ai, "type", "") != "ai":
                continue
            for tc in getattr(ai, "tool_calls", None) or []:
                if _tool_call_id(tc) != tc_id:
                    continue
                name = _tool_call_name(tc)
                if name == "get_food_list":
                    return "food"
                if name == "get_place_list":
                    return "places"
                return "places"
        break
    return _router_suggest_slot(state)


def _suggest_list_heading(slot: Literal["food", "places"]) -> str:
    """English label so users know if the list is dishes or sightseeing spots."""
    if slot == "food":
        return (
            "**Food (dishes to eat)**\n"
            "_Not neighborhoods or tourist spots — only meals / cuisine._\n"
        )
    return (
        "**Places (sightseeing / areas)**\n"
        "_Not dishes — only where to go (areas, landmarks, streets)._\n"
    )


# ── 1. Guardrail ──────────────────────────────────────────────────────────────

def _in_scope(user_input: str, messages: list) -> bool:
    if any(token in _normalize(user_input) for token in SEOUL_KEYWORDS):
        return True
    for m in messages[-6:]:
        content = _normalize(str(getattr(m, "content", "") or ""))
        if any(token in content for token in SEOUL_KEYWORDS):
            return True
    return False


def node_guardrail(state: TravelState) -> dict:
    user_input = state.get("user_input", "")
    return {
        "in_scope": _in_scope(user_input, state.get("messages", [])),
        "messages": [HumanMessage(content=user_input)],   # add_messages reducer appends
    }


def route_after_guardrail(state: TravelState) -> str:
    return "router" if state.get("in_scope") else "out_of_scope"


# ── 2. Router — Ch16.3 Routing pattern ────────────────────────────────────────

class IntentResult(BaseModel):
    intent: Literal["itinerary", "suggest_food", "suggest_places", "other"] = Field(
        description=(
            "itinerary: user provides trip details or asks for a plan. "
            "suggest_food: user wants food/dining suggestions to pick from. "
            "suggest_places: user wants sightseeing place suggestions. "
            "other: general Seoul question."
        )
    )


def node_router(state: TravelState) -> dict:
    """Ch16.3 Routing — LLM classifies intent; regex extracts days (no LLM cost)."""
    user_input = state.get("user_input", "")
    update: dict = {}

    # Extract days with regex here (not in parallel nodes to avoid write-race)
    if not state.get("days"):
        m = re.search(DAY_PATTERN, _normalize(user_input))
        if m:
            update["days"] = max(1, int(m.group(1)))
        elif "weekend" in _normalize(user_input):
            update["days"] = 2

    if LLM is None:
        update["intent"] = "itinerary"
        return update

    # Fast path: skip LLM for obvious approval keywords
    if _normalize(user_input).rstrip(".!?") in HITL_APPROVE_KEYWORDS:
        update["intent"] = "itinerary"
        return update

    llm = LLM.with_structured_output(IntentResult)
    try:
        result = llm.invoke([
            SystemMessage(
                "Classify the intent of this Seoul travel chatbot user message.\n"
                "itinerary: trip length and/or BOTH food and places (or asks for a full plan).\n"
                "suggest_food: user wants ONLY dishes/meals/what to eat (e.g. BBQ, street food, vegetarian).\n"
                "suggest_places: user wants ONLY where to go / sightseeing / neighborhoods / landmarks.\n"
                "other: general Seoul question without picking food-vs-place suggestion mode.\n"
                "Disambiguation: vague 'suggestions' about eating → suggest_food; about visiting/seeing → suggest_places.\n"
                "Examples — suggest_food: 'gợi ý ăn', '먹을 거 추천', 'what should I eat'.\n"
                "Examples — suggest_places: 'gợi ý chỗ chơi', 'đi đâu', '관광지 추천', 'where to visit'."
            ),
            HumanMessage(user_input),
        ])
        update["intent"] = result.intent if isinstance(result, IntentResult) else "itinerary"
    except Exception:
        update["intent"] = "itinerary"

    return update


def route_after_router(state: TravelState) -> str:
    """Return routing key — mapped to node(s) in add_conditional_edges."""
    intent = state.get("intent", "itinerary")
    if intent in ("suggest_food", "suggest_places"):
        return "suggest"
    return "extract"   # itinerary / other → parallel fan-out


# ── 3. Suggest + ToolNode — Ch14 Tool Use (ReAct loop) ───────────────────────

def node_suggest_llm(state: TravelState) -> dict:
    """
    Ch14 ToolNode + bind_tools.
    First call: LLM decides which tool to call.
    Second call (after ToolNode returns results): format pick list → interrupt for user pick.
    """
    if LLM is None:
        return {"final_response": "LLM not configured.", "plan_generated": False}

    messages = state.get("messages", [])

    # Detect tool results produced THIS turn (after last HumanMessage)
    last_human_idx = max(
        (i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"),
        default=-1,
    )
    current_turn = messages[last_human_idx + 1:]
    tool_results = [m for m in current_turn if getattr(m, "type", "") == "tool"]

    if not tool_results:
        # First hop: tool is chosen by router intent (not the LLM) so lists never swap food/places.
        slot = _router_suggest_slot(state)
        tool_name = "get_food_list" if slot == "food" else "get_place_list"
        call_id = f"sg_{uuid.uuid4().hex[:16]}"
        return {
            "messages": [
                AIMessage(
                    content=f"Requesting {slot} suggestions (router intent).",
                    tool_calls=[
                        {
                            "name": tool_name,
                            "args": {},
                            "id": call_id,
                            "type": "tool_call",
                        }
                    ],
                )
            ],
        }

    # Second call: tool results available — present pick list then interrupt
    raw = tool_results[-1].content
    try:
        items: list[str] = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
    except Exception:
        items = [str(raw)]

    list_slot = _slot_from_last_tool_message(state)
    heading = _suggest_list_heading(list_slot)
    formatted = (
        heading
        + "\nHere are some options:\n"
        + "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))
        + (
            "\n\nHow to pick:\n"
            "- If you asked for **food suggestions**, pick a **dish/restaurant**.\n"
            "- If you asked for **place suggestions**, pick a **sightseeing spot/area**.\n"
            "\nReply with a number (1, 2, 3...) to pick one, or type your preference in your own words."
        )
    )

    pick = interrupt(formatted)

    pick_str = str(pick).strip()
    chosen = (
        items[int(pick_str) - 1]
        if pick_str.isdigit() and 0 < int(pick_str) <= len(items)
        else pick_str
    )

    slot = _slot_from_last_tool_message(state)
    if slot == "food":
        final = f"Saved your food preference: **{chosen}**"
        return {"food": chosen, "final_response": final, "messages": [AIMessage(content=final)]}
    final = f"Saved your place preference: **{chosen}**"
    return {"places": chosen, "final_response": final, "messages": [AIMessage(content=final)]}


# Ch14 ToolNode — executes tool calls made by suggest_llm
tool_executor = ToolNode(_SUGGESTION_TOOLS)


# ── 4. Parallel extraction — Ch16.4 Parallelization ──────────────────────────

def node_extract_food(state: TravelState) -> dict:
    """
    Ch16.4 Parallel branch A — extract food from user message.
    Runs simultaneously with node_extract_places.
    Also owns days extraction to avoid write-race with the sibling node.
    """
    if (state.get("food") or "").strip() or not state.get("user_input") or LLM is None:
        return {}
    from prompt_chain import run_food_intake_chain_only
    food = run_food_intake_chain_only(state["user_input"], "food preference")
    if food and food not in ("ANY", ""):
        return {"food": food}
    return {}


def node_extract_places(state: TravelState) -> dict:
    """
    Runs after extract_food — food is already on state for LLM place extraction hint.
    """
    if (state.get("places") or "").strip() or not state.get("user_input") or LLM is None:
        return {}
    from prompt_chain import run_places_intake_chain_only
    food_hint = (state.get("food") or "").strip() or None
    places = run_places_intake_chain_only(state["user_input"], "places/areas", food_hint)
    if places and places not in ("ANY", ""):
        return {"places": places}
    return {}


def node_slot_aggregator(state: TravelState) -> dict:
    """Fan-in sync point after parallel extraction. Prompts for any missing slot."""
    missing: list[str] = []
    if not state.get("days"):
        missing.append("trip length (e.g. 3 days, weekend)")
    if not (state.get("food") or "").strip():
        missing.append("food preferences — or say *anything* for suggestions")
    if not (state.get("places") or "").strip():
        missing.append("places to visit — or say *anything* for suggestions")
    if missing:
        msg = "Still needed:\n- " + "\n- ".join(missing)
        return {"final_response": msg, "plan_generated": False}
    return {}


def route_after_aggregator(state: TravelState) -> str:
    complete = (
        state.get("days")
        and (state.get("food") or "").strip()
        and (state.get("places") or "").strip()
    )
    return "pre_plan_hitl" if complete else "end"


# ── 5. Human-in-the-loop — Ch14 interrupt ────────────────────────────────────

def node_pre_plan_hitl(state: TravelState) -> dict:
    days = state.get("days")
    food = (state.get("food") or "").strip()
    places = (state.get("places") or "").strip()

    if not days or not food or not places:
        return node_slot_aggregator(state)
    if int(days) > 7:
        return {"final_response": MSG_ITINERARY_LIMIT_EXCEEDED, "plan_generated": False}
    gate = places_outside_seoul_gate(places)
    if gate:
        return {"final_response": gate, "plan_generated": False}

    draft = f"- Days: {days}\n- Food: {food}\n- Places: {places}"
    feedback = interrupt({
        "title": MSG_TRIP_HITL_TITLE,
        "draft": draft,
        "instruction": MSG_TRIP_HITL_INSTRUCTION,
    })
    if _normalize(str(feedback)).rstrip(".!?") in HITL_APPROVE_KEYWORDS:
        return {"pre_plan_approved": True}
    return {"final_response": MSG_TRIP_HITL_DENIED, "plan_generated": False}


def route_after_hitl(state: TravelState) -> str:
    return "orchestrator" if state.get("pre_plan_approved") else "end"


# ── 6. Orchestrator-Workers + Send API — Ch16.5 ───────────────────────────────

def node_orchestrator(state: TravelState) -> dict:
    """Sync point before fan-out; actual Send is in the conditional edge below."""
    return {}


def _send_to_day_workers(state: TravelState) -> list[Send]:
    """Ch16.5 Orchestrator fan-out — one Send per day."""
    days = max(1, min(7, int(state.get("days") or 1)))
    food = state.get("food", "")
    places = state.get("places", "")
    return [
        Send("day_worker", {
            "day_number": i,
            "days": days,
            "food": food,
            "places": places,
            # Required TravelState fields with safe defaults
            "user_input": state.get("user_input", ""),
            "in_scope": True,
            "intent": "itinerary",
            "pre_plan_approved": True,
            "day_plans": [],
            "final_response": "",
            "plan_generated": False,
            "messages": [],
        })
        for i in range(1, days + 1)
    ]


def node_day_worker(state: TravelState) -> dict:
    """Ch16.5 Worker — writes one day's itinerary; runs in parallel with sibling workers."""
    day_num = state.get("day_number", 1)
    days = state.get("days", 1)
    food = state.get("food", "")
    places = state.get("places", "")

    if LLM is None:
        plan = f"## Day {day_num}\n\n**Places:** {places}\n**Food:** {food}\n"
        return {"day_plans": [plan]}

    system = (
        "You are an expert Seoul travel planner. Write ONE day's itinerary in Markdown. "
        "Include morning, afternoon, and evening blocks. "
        "Use real Seoul neighborhood names and suggest specific dishes. "
        "Add brief transit notes (subway line or walking). Be practical and concise."
    )
    user = (
        f"Day {day_num} of a {days}-day Seoul trip.\n"
        f"Places: {places}\n"
        f"Food preferences: {food}\n"
        f"Make Day {day_num} feel distinct from other days."
    )
    try:
        ai_msg = LLM.invoke([SystemMessage(system), HumanMessage(user)])
        body = _to_text(getattr(ai_msg, "content", ""))
        plan = f"## Day {day_num}\n\n{body}"
    except Exception as e:
        plan = f"## Day {day_num}\n\nCould not generate plan ({e})"

    # Ch13 Reducer: return list — operator.add appends to parent state's day_plans
    return {"day_plans": [plan]}


def node_aggregate(state: TravelState) -> dict:
    """Combine all day plans accumulated by operator.add reducer into final response."""
    days = state.get("days", 1)
    plans = state.get("day_plans", [])

    def _day_num(p: str) -> int:
        m = re.search(r"## Day (\d+)", p)
        return int(m.group(1)) if m else 999

    sorted_plans = sorted(plans, key=_day_num)
    header = f"# Your Seoul Itinerary ({days} days)\n\n"
    return {
        "final_response": header + "\n\n---\n\n".join(sorted_plans),
        "plan_generated": True,
    }


# ── Out of scope ──────────────────────────────────────────────────────────────

def node_out_of_scope(_: TravelState) -> dict:
    return {"final_response": MSG_OUT_OF_SCOPE, "plan_generated": False}


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(TravelState)

    builder.add_node("guardrail", node_guardrail)
    builder.add_node("router", node_router)
    builder.add_node("suggest_llm", node_suggest_llm)
    builder.add_node("tool_executor", tool_executor)      
    builder.add_node("extract_food", node_extract_food)
    builder.add_node("extract_places", node_extract_places)
    builder.add_node("slot_aggregator", node_slot_aggregator)
    builder.add_node("pre_plan_hitl", node_pre_plan_hitl)
    builder.add_node("orchestrator", node_orchestrator)
    builder.add_node("day_worker", node_day_worker)
    builder.add_node("aggregate", node_aggregate)
    builder.add_node("out_of_scope", node_out_of_scope)

    # Entry
    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail", route_after_guardrail,
        {"router": "router", "out_of_scope": "out_of_scope"},
    )

    # Ch16.3 Routing — "extract" goes to food first, then places (sequential; LangGraph path map
    # cannot be a list of nodes — parallel fan-out would need Send from the router).
    builder.add_conditional_edges(
        "router", route_after_router,
        {
            "suggest": "suggest_llm",
            "extract": "extract_food",
        },
    )

    # Ch14 ReAct loop: suggest_llm ↔ tool_executor
    builder.add_conditional_edges(
        "suggest_llm", tools_condition,
        {"tools": "tool_executor", END: END},
    )
    builder.add_edge("tool_executor", "suggest_llm")

    # Extraction chain: food → places (places step can use food in state) → aggregator
    builder.add_edge("extract_food", "extract_places")
    builder.add_edge("extract_places", "slot_aggregator")
    builder.add_conditional_edges(
        "slot_aggregator", route_after_aggregator,
        {"pre_plan_hitl": "pre_plan_hitl", "end": END},
    )

    # Ch14 HITL
    builder.add_conditional_edges(
        "pre_plan_hitl", route_after_hitl,
        {"orchestrator": "orchestrator", "end": END},
    )

    # Ch16.5 Orchestrator-Workers via Send API
    builder.add_conditional_edges("orchestrator", _send_to_day_workers)
    builder.add_edge("day_worker", "aggregate")
    builder.add_edge("aggregate", END)
    builder.add_edge("out_of_scope", END)

    return builder.compile(checkpointer=SqliteSaver(SQLITE_CONN))


graph = build_graph()
