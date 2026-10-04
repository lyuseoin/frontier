import numpy as np
import pytest

from core import analyze
from core.recommend import counter_cards, detox_progress, user_centroid, _cosine


@pytest.fixture(scope="module")
def media():
    return analyze.load_media()


@pytest.fixture(scope="module")
def topics(media):
    return list(media["axes"].keys())


def item(topic, stance):
    return {"topic": topic, "stance": stance}


# --- 편향도 점수 -------------------------------------------------------------

def test_극단_편식은_점수가_매우_높다(topics):
    items = [item("기술", 0.9) for _ in range(5)]
    assert analyze.bias_score(items, topics) >= 90


def test_완전_편식은_100점(topics):
    items = [item("기술", 1.0) for _ in range(5)]
    assert analyze.bias_score(items, topics) == 100


def test_주제와_성향이_고른_소비는_점수가_낮다(topics):
    items = [
        item("기술", 0.9), item("환경", -0.9),
        item("경제", 0.9), item("교육", -0.9),
        item("사회", 0.0), item("문화", 0.0),
    ]
    assert analyze.bias_score(items, topics) <= 10


def test_균형잡힌_소비가_편식보다_항상_낮다(topics):
    biased = [item("기술", 0.9) for _ in range(4)]
    balanced = [item("기술", 0.9), item("환경", -0.9), item("경제", 0.5), item("교육", -0.5)]
    assert analyze.bias_score(balanced, topics) < analyze.bias_score(biased, topics)


def test_빈_입력은_0점(topics):
    assert analyze.bias_score([], topics) == 0


def test_항목이_하나면_쏠림도는_최대(topics):
    assert analyze.topic_concentration([item("기술", 0.0)], topics) == 1.0


def test_점수는_항상_0에서_100_사이(topics):
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(1, 12))
        items = [
            item(topics[int(rng.integers(0, len(topics)))], float(rng.uniform(-1, 1)))
            for _ in range(n)
        ]
        assert 0 <= analyze.bias_score(items, topics) <= 100


# --- 군집화 -----------------------------------------------------------------

def test_군집_좌표는_항상_2차원(topics):
    for n in (1, 2, 3, 8):
        items = [item(topics[i % len(topics)], 0.5) for i in range(n)]
        result = analyze.cluster_map(items, topics)
        assert len(result["coords"]) == n
        assert all(len(c) == 2 for c in result["coords"])
        assert len(result["labels"]) == n


def test_군집_결과는_재현된다(topics):
    items = [item(topics[i % 4], (-1) ** i * 0.6) for i in range(9)]
    a = analyze.cluster_map(items, topics)
    b = analyze.cluster_map(items, topics)
    assert a == b


def test_주제_비중_합은_1(topics):
    items = [item("기술", 0.5), item("환경", -0.5), item("기술", 0.1)]
    ratio = analyze.topic_ratio(items, topics)
    assert ratio["기술"] == pytest.approx(2 / 3)
    assert sum(ratio.values()) == pytest.approx(1.0)


# --- 대조군 추천 -------------------------------------------------------------

def test_추천은_실제로_유사도_최저순이다(media, topics):
    items = [item("기술", 0.9) for _ in range(3)]
    picked = counter_cards(items, media, top_n=5)

    centroid = user_centroid(items, topics)
    allowed = [c for c in media["cards"] if c["quality"] >= analyze.QUALITY_MIN]
    sims = sorted(
        _cosine(centroid, analyze.to_vector(c["topic"], c["stance"], topics))
        for c in allowed
    )
    assert [p["similarity"] for p in picked] == pytest.approx(sims[:5], abs=1e-4)


def test_저품질_극단_카드는_추천되지_않는다(media):
    items = [item("교육", -0.9) for _ in range(3)]
    picked = counter_cards(items, media, top_n=40)
    ids = {p["id"] for p in picked}
    assert {"d1", "s1", "h1"}.isdisjoint(ids)
    assert all(p["quality"] >= analyze.QUALITY_MIN for p in picked)


def test_이미_고른_카드는_다시_추천되지_않는다(media):
    items = [item("기술", 0.9)]
    first = counter_cards(items, media, top_n=1)[0]
    again = counter_cards(items, media, exclude_ids={first["id"]}, top_n=5)
    assert first["id"] not in {c["id"] for c in again}


def test_추천에는_항상_이유가_붙는다(media):
    items = [item("기술", 0.9), item("기술", 0.5)]
    for card in counter_cards(items, media, top_n=3):
        assert card["reason"]


def test_대조군을_더하면_편향도가_내려간다(media, topics):
    items = [item("기술", 0.9) for _ in range(4)]
    before = analyze.bias_score(items, topics)
    picked = counter_cards(items, media, top_n=3)
    after = analyze.bias_score(items + picked, topics)
    assert after < before


# --- 디톡스 게이지 -----------------------------------------------------------

def test_게이지는_목표_도달시_가득_찬다():
    assert detox_progress(90, 40) == 1.0
    assert detox_progress(90, 30) == 1.0


def test_게이지는_시작시점에_비어있다():
    assert detox_progress(90, 90) == 0.0


def test_게이지는_중간값을_비율로_반환한다():
    assert detox_progress(90, 65) == pytest.approx(0.5)


def test_이미_목표를_달성한_상태는_가득_찬_게이지():
    assert detox_progress(30, 30) == 1.0
