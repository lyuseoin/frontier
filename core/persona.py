"""필터버블 유형 판정.

최근 본 미디어 키워드에서 '알고리즘이 당신을 얼마나 좁은 방에 가두고 있는지'를
유형으로 뽑아준다. 유형 이름에는 가벼운 캐릭터 느낌만 남기고, 설명은 필터버블
진단 톤으로 한다. 판정은 두 지표(주제 폭 · 관점 치우침)만으로 결정되므로 같은
입력이면 항상 같은 유형이 나온다.
"""

from __future__ import annotations

from core.analyze import bias_score, stance_skew, topic_concentration

# 3단계 경계. 0.34 미만 / 0.67 미만 / 그 이상
LOW, MID = 0.34, 0.67

TOPIC_LEVELS = ["넓음", "보통", "좁음"]       # 주제 쏠림이 클수록 좁다
STANCE_LEVELS = ["균형", "보통", "치우침"]

# 주 관심 주제 → 화면 강조색 (사주 '기운' 명칭은 걷어내고 색 강조만 남긴다)
TOPIC_COLORS: dict[str, str] = {
    "정치": "#6C8CFF",
    "환경": "#22C55E",
    "경제": "#F5A623",
    "기술": "#2F6BFF",
    "교육": "#38BDF8",
    "사회": "#A78BFA",
    "건강": "#FF6B6B",
    "문화": "#F472B6",
}
DEFAULT_COLOR = "#9AA3B2"

# (주제 단계, 관점 단계) -> 유형. tagline/blurb 은 필터버블 진단 톤.
TYPES: dict[tuple[int, int], dict] = {
    (0, 0): {
        "name": "만화경형",
        "emoji": "🌈",
        "vibe": "어디에도 갇히지 않은",
        "tagline": "여러 주제를 양쪽 관점으로 넘나들어서, 알고리즘이 당신을 한 칸에 가두지 못하고 있어요.",
        "blurb": "필터버블이 거의 없는 상태. 지금의 균형을 유지하는 것이 관건이에요.",
    },
    (0, 1): {
        "name": "나침반형",
        "emoji": "🧭",
        "vibe": "넓게 보되 살짝 기운",
        "tagline": "주제는 넓게 열려 있지만 관점이 조금씩 한쪽으로 기울고 있어요. 알고리즘은 그 미세한 기울기를 이미 읽었어요.",
        "blurb": "아직 얕은 필터버블. 기우는 방향을 스스로 알아차리면 벗어나기 쉬워요.",
    },
    (0, 2): {
        "name": "등대형",
        "emoji": "🗼",
        "vibe": "넓게 보되 결론은 한 방향",
        "tagline": "다양한 주제를 보지만 결론은 매번 같은 편에 서요. 알고리즘에게 당신은 '주제를 가리지 않는 확실한 한쪽'이에요.",
        "blurb": "주제는 넓은데 관점이 굳은 필터버블. 반대편 관점이 잘 안 들어와요.",
    },
    (1, 0): {
        "name": "저울형",
        "emoji": "⚖️",
        "vibe": "양쪽을 재보는",
        "tagline": "몇 개의 관심사 안에서 양쪽 이야기를 고루 들어요. 알고리즘 입장에선 예측하기 까다로운 사용자예요.",
        "blurb": "관심사는 좁지만 관점은 열린 편. 비교적 건강한 소비 습관이에요.",
    },
    (1, 1): {
        "name": "물결형",
        "emoji": "🌊",
        "vibe": "가장 흔한 자리의",
        "tagline": "특별히 넓지도 치우치지도 않은 상태예요. 바로 여기서 추천 알고리즘이 방향을 정하기 시작해요.",
        "blurb": "필터버블이 만들어지기 직전의 갈림길. 지금 습관이 방향을 정해요.",
    },
    (1, 2): {
        "name": "화살형",
        "emoji": "🏹",
        "vibe": "관심은 여럿, 관점은 하나",
        "tagline": "여러 주제를 보지만 관점은 한쪽으로 굳어 있어요. 알고리즘은 그 관점에 맞는 것만 골라 넣기 시작했어요.",
        "blurb": "관점이 굳어가는 필터버블. 같은 결론의 콘텐츠가 계속 쌓여요.",
    },
    (2, 0): {
        "name": "우물형",
        "emoji": "🪔",
        "vibe": "깊게 파되 양쪽 벽을 살피는",
        "tagline": "한 주제에 깊이 들어가 있지만 그 안에서 반대 의견도 함께 봐요. 좁지만 건강한 몰입이에요.",
        "blurb": "주제는 좁아도 관점은 열린 필터버블. 시야만 조금 넓히면 좋아요.",
    },
    (2, 1): {
        "name": "자석형",
        "emoji": "🧲",
        "vibe": "한 세계로 끌려드는",
        "tagline": "한 주제에 시간을 몰아 쓰고 있어요. 알고리즘은 이 주제 바깥의 세상을 거의 보여주지 않아요.",
        "blurb": "두꺼워지는 필터버블. 좋아하는 주제 밖이 점점 안 보여요.",
    },
    (2, 2): {
        "name": "블랙홀형",
        "emoji": "🕳️",
        "vibe": "하나의 방에 갇힌",
        "tagline": "하나의 주제를 하나의 관점으로만 보고 있어요. 알고리즘에게 가장 예측하기 쉬운, 가장 두꺼운 필터버블 안이에요.",
        "blurb": "가장 강한 필터버블. 비슷한 것만 계속 돌아오는 에코 체임버 상태예요.",
    },
}


def _level(value: float) -> int:
    """0.0~1.0 값을 3단계로 나눈다."""
    return 0 if value < LOW else 1 if value < MID else 2


def main_topic(items: list[dict]) -> str:
    """가장 많이 본 주제."""
    if not items:
        return ""
    counts: dict[str, int] = {}
    for i in items:
        counts[i["topic"]] = counts.get(i["topic"], 0) + 1
    return max(counts, key=lambda t: counts[t])


def main_side(items: list[dict], axes: dict) -> str:
    """주 관심 주제에서 어느 쪽 관점에 서 있는지."""
    topic = main_topic(items)
    labels = axes.get(topic)
    if not labels:
        return "중립"
    same = [i["stance"] for i in items if i["topic"] == topic]
    avg = sum(same) / len(same) if same else 0.0
    if abs(avg) < 0.15:
        return "중립·사실 전달"
    return labels[1] if avg > 0 else labels[0]


def persona(items: list[dict], media: dict) -> dict:
    """소비 이력에서 필터버블 유형을 판정한다."""
    topics = list(media["axes"].keys())
    conc = topic_concentration(items, topics) if items else 0.0
    skew = stance_skew(items)

    t_level, s_level = _level(conc), _level(skew)
    base = TYPES[(t_level, s_level)]

    top = main_topic(items)

    return {
        **base,
        "color": TOPIC_COLORS.get(top, DEFAULT_COLOR),
        "topic_level": TOPIC_LEVELS[t_level],
        "stance_level": STANCE_LEVELS[s_level],
        "topic_index": t_level,
        "stance_index": s_level,
        "main_topic": top,
        "main_side": main_side(items, media["axes"]),
        "concentration": round(conc, 3),
        "skew": round(skew, 3),
        "bias_score": bias_score(items, topics),
    }
