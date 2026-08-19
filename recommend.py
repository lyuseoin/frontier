"""역발상 콘텐츠 기반 필터링.

기존 추천 알고리즘이 '가장 비슷한 것'을 고르는 자리에서, 이 모듈은
'가장 덜 비슷한 것'을 고른다. 다만 품질 하한을 두어 반대편이라는 이유만으로
극단적인 콘텐츠를 추천하지 않는다.
"""

from __future__ import annotations

import numpy as np

from core.analyze import QUALITY_MIN, to_vector, vectors_of


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도. 영벡터는 0으로 처리한다."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def user_centroid(items: list[dict], topics: list[str]) -> np.ndarray:
    """사용자 소비 이력의 중심 벡터."""
    if not items:
        return np.zeros(len(topics) + 1)
    return vectors_of(items, topics).mean(axis=0)


def explain(card: dict, items: list[dict], axes: dict) -> str:
    """이 카드가 왜 대조군인지 설명한다. 추천 결과에는 항상 이유가 따라붙는다."""
    main_topic = max(
        {i["topic"] for i in items},
        key=lambda t: sum(1 for i in items if i["topic"] == t),
        default="",
    )
    labels = axes.get(card["topic"], ["관점 A", "관점 B"])
    if abs(card["stance"]) < 0.15:
        side = "중립·사실 전달"
    else:
        side = labels[1] if card["stance"] > 0 else labels[0]

    if card["topic"] != main_topic:
        return f"주로 보던 '{main_topic}' 주제 밖의 '{card['topic']}' 이야기입니다. 관점은 '{side}'."
    return f"같은 '{card['topic']}' 주제이지만 반대편인 '{side}' 관점입니다."


def counter_cards(
    items: list[dict],
    media: dict,
    exclude_ids: set[str] | None = None,
    top_n: int = 3,
    quality_min: float = QUALITY_MIN,
) -> list[dict]:
    """유사도가 가장 낮은 순으로 대조군 카드를 추천한다."""
    topics = list(media["axes"].keys())
    axes = media["axes"]
    exclude_ids = exclude_ids or set()

    centroid = user_centroid(items, topics)

    scored = []
    for card in media["cards"]:
        if card["id"] in exclude_ids:
            continue
        if card["quality"] < quality_min:  # 극단·저품질 콘텐츠는 대조군에서 제외
            continue
        vec = to_vector(card["topic"], card["stance"], topics)
        scored.append({**card, "similarity": round(_cosine(centroid, vec), 4)})

    scored.sort(key=lambda c: c["similarity"])
    picked = scored[:top_n]
    for card in picked:
        card["reason"] = explain(card, items, axes)
    return picked


def explain_echo(card: dict, items: list[dict], axes: dict) -> str:
    """이 카드가 왜 '알고리즘이 다음에 줄 것'인지 설명한다."""
    labels = axes.get(card["topic"], ["관점 A", "관점 B"])
    if abs(card["stance"]) < 0.15:
        side = "중립적인"
    else:
        side = labels[1] if card["stance"] > 0 else labels[0]
    return f"당신이 본 '{card['topic']}' 이야기와 결이 같습니다. ({side})"


def echo_cards(
    items: list[dict],
    media: dict,
    exclude_ids: set[str] | None = None,
    top_n: int = 3,
) -> list[dict]:
    """유사도가 가장 높은 순으로, 알고리즘이 다음에 보여줄 콘텐츠를 예측한다.

    counter_cards 와 달리 품질 하한을 두지 않는다. 실제 추천 알고리즘은
    품질을 가려주지 않으며, 그 점을 그대로 보여주는 것이 이 화면의 목적이다.
    """
    topics = list(media["axes"].keys())
    axes = media["axes"]
    exclude_ids = exclude_ids or set()

    centroid = user_centroid(items, topics)

    scored = []
    for card in media["cards"]:
        if card["id"] in exclude_ids:
            continue
        vec = to_vector(card["topic"], card["stance"], topics)
        scored.append({**card, "similarity": round(_cosine(centroid, vec), 4)})

    scored.sort(key=lambda c: -c["similarity"])
    picked = scored[:top_n]
    for card in picked:
        card["reason"] = explain_echo(card, items, axes)
    return picked


def detox_progress(initial_score: int, current_score: int, goal: int = 40) -> float:
    """디톡스 게이지(0.0~1.0). 목표 점수까지 얼마나 내려왔는지."""
    if initial_score <= goal:
        return 1.0
    dropped = initial_score - current_score
    return max(0.0, min(1.0, dropped / (initial_score - goal)))
