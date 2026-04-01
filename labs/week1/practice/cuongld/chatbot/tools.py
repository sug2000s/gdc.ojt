"""Tools for fetching food/places using Kakao Local API."""

from __future__ import annotations

import json
import os
import random
import urllib.parse
import urllib.request
from typing import Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

_KAKAO_BASE = "https://dapi.kakao.com"

# Default to central Seoul (City Hall) to keep suggestions in-scope for this app.
_SEOUL_X = "126.9780"  # longitude
_SEOUL_Y = "37.5665"   # latitude
_SEOUL_RADIUS_M = "10000"


def _kakao_rest_api_key() -> str:
    return (os.getenv("KAKAO_REST_API_KEY") or "").strip()


def _kakao_get(path: str, params: dict[str, str]) -> dict[str, Any]:
    api_key = _kakao_rest_api_key()
    if not api_key:
        # Keep the app usable in lab/dev environments.
        # ToolNode exceptions bubble up to the UI, so prefer a safe empty payload.
        return {"documents": []}

    query = urllib.parse.urlencode(params, doseq=True)
    url = f"{_KAKAO_BASE}{path}?{query}"
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": f"KakaoAK {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = resp.read().decode("utf-8", errors="replace")
    return json.loads(payload)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        s2 = (s or "").strip()
        if not s2 or s2 in seen:
            continue
        seen.add(s2)
        out.append(s2)
    return out


def _build_llm() -> ChatOpenAI | None:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model_name = (os.getenv("OPENAI_MODEL_NAME") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if not api_key or not model_name:
        return None
    kwargs: dict[str, Any] = {"api_key": api_key, "model": model_name, "temperature": 0.0, "timeout": 60}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _translate_items_to_english(items: list[str], item_kind: str) -> list[str]:
    """
    Translate Kakao-returned items to English using the configured OpenAI model.

    - Keeps list length and order.
    - Keeps format "Name — Address" when present.
    - Falls back to original items on any failure.
    """
    llm = _build_llm()
    if llm is None:
        return items

    cleaned = [str(s).strip() for s in items if str(s).strip()]
    if not cleaned:
        return items

    prompt = (
        "Translate the following Korean place/restaurant entries to English.\n"
        f"Entry type: {item_kind}.\n"
        "Rules:\n"
        "- Output MUST be a valid JSON array of strings ONLY.\n"
        "- Keep the same number of items and the same order.\n"
        "- If an entry has the form 'NAME — ADDRESS', translate both parts, keep the ' — ' separator.\n"
        "- Keep well-known proper nouns (e.g. 'COEX', 'DDP') as common English names.\n"
        "- Do NOT add numbering or extra commentary.\n\n"
        f"Input JSON:\n{json.dumps(cleaned, ensure_ascii=False)}"
    )

    try:
        resp = llm.invoke(prompt)
        text = str(getattr(resp, "content", "") or "").strip()
        translated = json.loads(text)
        if isinstance(translated, list) and len(translated) == len(cleaned) and all(
            isinstance(x, str) for x in translated
        ):
            return [x.strip() for x in translated]
    except Exception:
        return items

    return items


@tool
def get_place_list() -> list[str]:
    """
    Return place suggestions from Kakao Local keyword search (query='핫플').

    Equivalent to:
      curl -X GET "https://dapi.kakao.com/v2/local/search/keyword.json?page=1&size=10&sort=accuracy&query=%ED%95%AB%ED%94%8C" \
        -H "Authorization: KakaoAK <REST_API_KEY>"
    """
    if not _kakao_rest_api_key():
        return [
            "Myeongdong — Jung-gu, Seoul",
            "Gyeongbokgung Palace — Jongno-gu, Seoul",
            "Bukchon Hanok Village — Jongno-gu, Seoul",
            "Hongdae — Mapo-gu, Seoul",
            "Gangnam Station — Seocho-gu, Seoul",
            "COEX Mall — Gangnam-gu, Seoul",
            "N Seoul Tower — Yongsan-gu, Seoul",
            "Ikseon-dong Hanok Street — Jongno-gu, Seoul",
            "Cheonggyecheon Stream — Jongno-gu, Seoul",
            "Hangang Park — Seoul",
        ]

    try:
        data = _kakao_get(
        "/v2/local/search/keyword.json",
        {
            "page": "1",
            "size": "10",
            "sort": "accuracy",
            "query": "핫플",
            "x": _SEOUL_X,
            "y": _SEOUL_Y,
            "radius": _SEOUL_RADIUS_M,
        },
    )
    except Exception:
        data = {"documents": []}
    docs = data.get("documents") or []
    results: list[str] = []
    if isinstance(docs, list):
        for d in docs:
            if not isinstance(d, dict):
                continue
            name = str(d.get("place_name") or "").strip()
            addr = str(d.get("address_name") or d.get("road_address_name") or "").strip()
            results.append(f"{name} — {addr}" if addr else name)
    results = _dedupe_keep_order(results)[:10]
    return _translate_items_to_english(results, "places")


@tool
def get_food_list() -> list[str]:
    """Return a random sample of fixed food suggestions (no API call)."""
    food_list = [
        "Korean BBQ (Samgyeopsal) — Mapo / Hongdae area",
        "Gwangjang Market street food — Jongno-gu, Seoul",
        "Bibimbap — Insadong area",
        "Kalguksu (knife-cut noodles) — Myeongdong area",
        "Tteokbokki — Sinchon / Hongdae area",
        "Fried chicken + beer (chimaek) — Hongdae area",
        "Naengmyeon (cold noodles) — Euljiro area",
        "Sundubu-jjigae (soft tofu stew) — City Hall area",
        "Hanjeongsik (Korean set meal) — Insadong area",
        "Cafe dessert (bingsu) — Ikseon-dong area",
        "Bossam (boiled pork wraps) — Jongno-gu, Seoul",
        "Dakgalbi (spicy stir-fried chicken) — Hongdae area",
        "Galbi (marinated ribs) — Gangnam area",
        "Gimbap — everywhere (quick snack)",
        "Jjajangmyeon (black bean noodles) — local Chinese-Korean spots",
    ]
    return random.sample(food_list, k=min(10, len(food_list)))
