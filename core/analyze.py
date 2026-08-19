"""편향도 진단 엔진. 판단이 아닌 계산만 담당하는 순수 함수 모음."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "media.json"

TOPIC_WEIGHT = 1.0  # 벡터에서 주제 축이 갖는 비중
QUALITY_MIN = 0.80  # 대조군 추천에서 제외할 품질 하한
RANDOM_STATE = 42   # 시연 재현성을 위해 고정


def load_media(path: Path | str = DATA_PATH) -> dict:
    """미디어 카드 데이터셋을 읽는다."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_vector(topic: str, stance: float, topics: list[str]) -> np.ndarray:
    """(주제, 성향)을 좌표 공간의 벡터로 변환한다. 마지막 축이 성향."""
    vec = np.zeros(len(topics) + 1, dtype=float)
    if topic in topics:
        vec[topics.index(topic)] = TOPIC_WEIGHT
    vec[-1] = float(stance)
    return vec


def vectors_of(items: list[dict], topics: list[str]) -> np.ndarray:
    """항목 목록을 벡터 행렬로 변환한다."""
    return np.array([to_vector(i["topic"], i["stance"], topics) for i in items])


def topic_concentration(items: list[dict], topics: list[str]) -> float:
    """주제 쏠림도(0=고르게 분산, 1=한 주제에 집중)를 엔트로피로 계산한다."""
    n = len(items)
    if n <= 1:
        return 1.0

    counts: dict[str, int] = {}
    for i in items:
        counts[i["topic"]] = counts.get(i["topic"], 0) + 1

    entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
    # 항목 수가 주제 수보다 적으면 그 개수만큼만 분산될 수 있으므로 그 한계로 정규화
    max_entropy = math.log(min(n, len(topics)))
    if max_entropy == 0:
        return 1.0
    return 1.0 - entropy / max_entropy


def stance_skew(items: list[dict]) -> float:
    """성향 치우침(0=양쪽 균형, 1=한쪽으로 완전히 쏠림)을 계산한다."""
    if not items:
        return 0.0
    return abs(float(np.mean([i["stance"] for i in items])))


def bias_score(items: list[dict], topics: list[str]) -> int:
    """편향도 점수 0~100. 높을수록 정보 편식이 심한 상태."""
    if not items:
        return 0
    raw = 0.5 * topic_concentration(items, topics) + 0.5 * stance_skew(items)
    return int(round(raw * 100))


def bias_detail(items: list[dict], topics: list[str]) -> dict:
    """대시보드 지표 카드에 쓸 진단 요약."""
    used = sorted({i["topic"] for i in items})
    return {
        "bias_score": bias_score(items, topics),
        "topic_concentration": round(topic_concentration(items, topics), 3),
        "stance_skew": round(stance_skew(items), 3),
        "topic_count": len(used),
        "topics": used,
        "item_count": len(items),
    }


def topic_ratio(items: list[dict], topics: list[str]) -> dict[str, float]:
    """주제별 소비 비중. 방사형 그래프용."""
    n = len(items)
    counts = {t: 0 for t in topics}
    for i in items:
        if i["topic"] in counts:
            counts[i["topic"]] += 1
    return {t: (c / n if n else 0.0) for t, c in counts.items()}


def cluster_map(items: list[dict], topics: list[str], k: int = 3) -> dict:
    """K-Means 군집화 후 2차원 좌표로 투영한다. 소비 패턴의 밀집도 시각화용."""
    n = len(items)
    if n == 0:
        return {"coords": [], "labels": [], "k": 0}

    vecs = vectors_of(items, topics)

    if n == 1:
        return {"coords": [[0.0, 0.0]], "labels": [0], "k": 1}

    k = max(1, min(k, n))
    labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(vecs)

    n_comp = min(2, n, vecs.shape[1])
    coords = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit_transform(vecs)
    if n_comp == 1:  # 2차원을 못 만들면 두 번째 축은 0으로 채운다
        coords = np.hstack([coords, np.zeros((n, 1))])

    return {
        "coords": coords.tolist(),
        "labels": labels.tolist(),
        "k": int(k),
    }
