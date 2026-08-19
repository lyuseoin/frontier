"""Gemini 기반 미디어 키워드 분류기.

AI는 '이 키워드가 어떤 주제이고 어느 관점에 가까운가'라는 판단만 담당한다.
점수 계산·군집화·추천은 core.analyze / core.recommend 의 결정론적 코드가 처리한다.
API 호출이 실패해도 앱이 멈추지 않도록 규칙 기반 폴백을 둔다.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

load_dotenv()

# gemini-2.5-flash 는 신규 사용자에게 더 이상 제공되지 않아(404) 3.6-flash 를 쓴다.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

_PROMPT = """너는 미디어 리터러시 교육 도구의 분류기다.
사용자가 최근 소비한 미디어 키워드 목록을 받아, 각 항목을 분류해라.

주제(topic)는 반드시 다음 중 하나로만 답한다:
{topics}

관점(stance)은 -1.0 ~ +1.0 사이의 실수다. 각 주제의 양 끝 의미는 다음과 같다:
{axes}
-1.0에 가까울수록 앞쪽 관점, +1.0에 가까울수록 뒤쪽 관점, 0.0은 중립·사실 전달.

규칙:
- 특정 입장이 옳거나 틀리다고 판단하지 마라. 어느 쪽에 가까운지만 위치시켜라.
- 판단 근거가 부족하면 stance 를 0.0 으로 둔다.
- label 은 입력 키워드를 그대로 쓴다.

입력 키워드:
{keywords}
"""

_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "topic": {"type": "string"},
            "stance": {"type": "number"},
            "note": {"type": "string"},
        },
        "required": ["label", "topic", "stance"],
    },
}


def _fallback(keywords: list[str], media: dict) -> list[dict]:
    """API를 못 쓸 때의 규칙 기반 분류. 주제명이 들어간 키워드만 매칭한다."""
    topics = list(media["axes"].keys())
    items = []
    for kw in keywords:
        topic = next((t for t in topics if t in kw), topics[0])
        items.append(
            {"label": kw, "topic": topic, "stance": 0.0, "note": "규칙 기반 임시 분류"}
        )
    return items


def _normalize(raw: list[dict], keywords: list[str], media: dict) -> list[dict]:
    """모델 응답을 신뢰하지 않고 값 범위와 주제명을 코드로 강제한다."""
    topics = list(media["axes"].keys())
    by_label = {str(r.get("label", "")).strip(): r for r in raw if isinstance(r, dict)}

    items = []
    for kw in keywords:
        r = by_label.get(kw, {})
        topic = r.get("topic")
        if topic not in topics:
            topic = next((t for t in topics if t in kw), topics[0])
        try:
            stance = float(r.get("stance", 0.0))
        except (TypeError, ValueError):
            stance = 0.0
        items.append(
            {
                "label": kw,
                "topic": topic,
                "stance": max(-1.0, min(1.0, stance)),
                "note": str(r.get("note", "")),
            }
        )
    return items


def classify(keywords: list[str], media: dict, model: str = DEFAULT_MODEL) -> tuple[list[dict], str]:
    """키워드를 (주제, 성향)으로 분류한다. (결과, 사용한 방식) 을 돌려준다."""
    keywords = [k.strip() for k in keywords if k.strip()]
    if not keywords:
        return [], "empty"

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback(keywords, media), "fallback: API 키 없음"

    try:
        from google import genai

        axes_text = "\n".join(
            f"- {t}: {labels[0]} (-1.0) ↔ {labels[1]} (+1.0)"
            for t, labels in media["axes"].items()
        )
        prompt = _PROMPT.format(
            topics=", ".join(media["axes"].keys()),
            axes=axes_text,
            keywords="\n".join(f"- {k}" for k in keywords),
        )

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": _SCHEMA,
                "temperature": 0.0,
            },
        )
        return _normalize(json.loads(resp.text), keywords, media), f"gemini ({model})"

    except Exception as exc:  # 네트워크·할당량·응답 형식 등 어떤 실패든 앱은 계속 돈다
        return _fallback(keywords, media), f"fallback: {type(exc).__name__}"


# ── 화면 캡처(비전) 분류 ─────────────────────────────────────────────────────

_IMAGE_PROMPT = """이 이미지는 사용자가 최근 본 미디어 플랫폼(유튜브, 인스타그램, 틱톡,
뉴스앱 등)의 화면 캡처다. 화면에 보이는 콘텐츠(영상 제목, 썸네일 문구, 게시물,
해시태그, 채널명 등)를 최대 8개까지 읽어서 각각을 분류해라.

주제(topic)는 반드시 다음 중 하나로만 답한다:
{topics}

관점(stance)은 -1.0 ~ +1.0 사이의 실수다. 각 주제의 양 끝 의미는 다음과 같다:
{axes}
0.0 은 중립·사실 전달.

규칙:
- 실제로 화면에서 읽히는 것만 뽑아라. 안 보이면 지어내지 마라.
- label 은 화면에서 읽은 제목/문구를 짧게 그대로 쓴다.
- 특정 입장이 옳거나 틀리다고 판단하지 말고 위치만 정해라.
- 판단 근거가 부족하면 stance 를 0.0 으로 둔다."""


def _normalize_free(raw: list[dict], media: dict) -> list[dict]:
    """입력 키워드 목록 없이, 모델이 자유롭게 뽑은 결과를 정리한다(비전용)."""
    topics = list(media["axes"].keys())
    items = []
    for r in raw if isinstance(raw, list) else []:
        if not isinstance(r, dict):
            continue
        label = str(r.get("label", "")).strip()
        if not label:
            continue
        topic = r.get("topic")
        if topic not in topics:
            topic = next((t for t in topics if t in label), None)
            if topic is None:
                continue  # 8개 주제로 분류 못 하면 버린다
        try:
            stance = float(r.get("stance", 0.0))
        except (TypeError, ValueError):
            stance = 0.0
        items.append({
            "label": label[:40],
            "topic": topic,
            "stance": max(-1.0, min(1.0, stance)),
            "note": str(r.get("note", "")),
        })
    return items[:8]


def classify_image(
    image_bytes: bytes, mime_type: str, media: dict, model: str = DEFAULT_MODEL
) -> tuple[list[dict], str]:
    """화면 캡처 이미지에서 미디어 콘텐츠를 읽어 (주제, 성향)으로 분류한다.

    이미지 기반이라 규칙 폴백이 불가능하다. 실패하면 빈 목록을 돌려주고, 호출부가
    사용자에게 다른 입력 방법을 안내한다.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return [], "fallback: API 키 없음"

    try:
        from google import genai
        from google.genai import types

        axes_text = "\n".join(
            f"- {t}: {labels[0]} (-1.0) ↔ {labels[1]} (+1.0)"
            for t, labels in media["axes"].items()
        )
        prompt = _IMAGE_PROMPT.format(
            topics=", ".join(media["axes"].keys()), axes=axes_text
        )
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type or "image/png")

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=[prompt, part],
            config={
                "response_mime_type": "application/json",
                "response_schema": _SCHEMA,
                "temperature": 0.2,
            },
        )
        return _normalize_free(json.loads(resp.text), media), f"vision ({model})"

    except Exception as exc:
        return [], f"fallback: {type(exc).__name__}"


# ── 필터버블 진단 풀이 ───────────────────────────────────────────────────────

_READ_PROMPT = """너는 청소년 미디어 리터러시 교육 도구의 해설자다. 사용자가 최근 본
미디어 키워드로 '추천 알고리즘이 이 사람을 어떤 필터버블에 가두고 있는지'를 읽어준다.
아래 유형과 수치는 코드가 이미 계산해 둔 것이다. 바꾸지 말고 풀어서 설명만 해라.

[사용자가 최근 본 것]
{consumed}

[판정된 필터버블 유형]
{type_name} ({emoji}) — {tagline}
주제 폭: {topic_level} / 관점: {stance_level}
주 관심사: {main_topic} ({main_side})
필터버블 지수: {bias_score}/100 (높을수록 좁은 방)

[알고리즘이 앞으로 계속 밀어줄 콘텐츠 (유사도 최고 = 에코 체임버)]
{echo}

[알고리즘이 거의 안 보여주는 콘텐츠 (유사도 최저 = 사각지대)]
{counter}

다음을 써라. 말투는 청소년에게 친근한 '해요체'.

1. diagnosis: 추천 알고리즘이 지금 이 사람을 어떻게 가두고 있는지 3~4문장.
   주어는 '당신의 알고리즘' 또는 '지금 당신의 피드'다. 사람 자체를 단정하지 말고
   '알고리즘이 그렇게 본다'로 표현해라.

2. feed: "앞으로도 계속 ~을 밀어줄 거예요" 형태의 예측 4개.
   위 [계속 밀어줄 콘텐츠]에 근거해 구체적인 소재를 짚어라. 각 40자 이내.

3. blindspots: 이 필터버블 때문에 잘 안 닿는 이야기 3개. 각 30자 이내.

4. forecast: "이대로라면 ~" 형태로, 필터버블이 더 두꺼워지면 어떻게 될지 1~2문장.

5. action: 필터버블을 넓히는 구체적 행동 한 문장. 40자 이내.

규칙: 특정 입장이 옳거나 틀리다고 말하지 마라. 관점은 취향일 뿐 좋고 나쁨이 아니다.
겁주지 말고, 필터버블은 누구에게나 생기는 자연스러운 것으로 담담하게 설명해라."""

_READ_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "feed": {"type": "array", "items": {"type": "string"}},
        "blindspots": {"type": "array", "items": {"type": "string"}},
        "forecast": {"type": "string"},
        "action": {"type": "string"},
    },
    "required": ["diagnosis", "feed", "blindspots", "forecast", "action"],
}


def _read_fallback(persona: dict, echo: list[dict], counter: list[dict]) -> dict:
    """API를 못 쓸 때. 유형 설명과 실제 카드 제목으로 화면을 채운다."""
    action = f"가끔 '{counter[0]['title']}' 같은 결이 다른 글도 열어보세요." if counter else ""
    return {
        "diagnosis": f"{persona['tagline']} {persona['blurb']}",
        "feed": [f"'{c['title']}' 같은 이야기" for c in echo],
        "blindspots": [f"{c['topic']} — {c['title']}" for c in counter],
        "forecast": "이대로라면 비슷한 콘텐츠만 점점 더 많이 추천되어, 다른 관점은 더 보기 어려워져요.",
        "action": action,
    }


def read_persona(
    items: list[dict],
    persona: dict,
    echo: list[dict],
    counter: list[dict],
    model: str = DEFAULT_MODEL,
) -> tuple[dict, str]:
    """판정된 필터버블 유형을 사람이 읽을 문장으로 풀어낸다."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _read_fallback(persona, echo, counter), "fallback: API 키 없음"

    try:
        from google import genai

        prompt = _READ_PROMPT.format(
            consumed="\n".join(
                f"- {i.get('label', i.get('title', ''))} ({i['topic']}, {i['stance']:+.1f})"
                for i in items
            ),
            type_name=persona["name"],
            emoji=persona["emoji"],
            tagline=persona["tagline"],
            topic_level=persona["topic_level"],
            stance_level=persona["stance_level"],
            main_topic=persona["main_topic"],
            main_side=persona["main_side"],
            bias_score=persona.get("bias_score", 0),
            echo="\n".join(f"- {c['title']} ({c['topic']})" for c in echo),
            counter="\n".join(f"- {c['title']} ({c['topic']})" for c in counter),
        )

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": _READ_SCHEMA,
                "temperature": 0.9,  # 서술이므로 매번 조금씩 다르게
            },
        )
        data = json.loads(resp.text)
        if not data.get("diagnosis") or not data.get("feed"):
            raise ValueError("빈 응답")
        return data, f"gemini ({model})"

    except Exception as exc:
        return _read_fallback(persona, echo, counter), f"fallback: {type(exc).__name__}"
