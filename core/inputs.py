"""입력 보조 — 키워드 칩 정의와 매핑.

칩(선택형 입력)은 관심 '주제'만 담고 관점(stance)은 담지 않는다. 그래서 Gemini를
부르지 않고 결정론적으로 주제만 매핑한다(오프라인·즉시). 관점 쏠림까지 보려면
직접 입력이나 화면 캡처를 쓴다.
"""

from __future__ import annotations

# (칩 라벨, 매핑되는 주제). 주제는 data/media.json 의 8개 축과 일치해야 한다.
# 청소년이 실제로 많이 보는 관심사를 8개 주제에 고루 걸쳐 배치한다.
CHIPS: list[tuple[str, str]] = [
    ("🗳️ 정치·시사", "정치"),
    ("💰 경제·주식", "경제"),
    ("🤖 AI·IT", "기술"),
    ("📚 입시·공부", "교육"),
    ("🌱 환경·기후", "환경"),
    ("🏙️ 사회·뉴스", "사회"),
    ("💪 건강·운동", "건강"),
    ("🎤 아이돌·K팝", "문화"),
    ("🎮 게임", "문화"),
    ("🎬 드라마·영화", "문화"),
    ("📱 유튜브 쇼츠", "사회"),
    ("⚽ 스포츠", "건강"),
]

CHIP_LABELS: list[str] = [label for label, _ in CHIPS]
CHIP_TOPIC: dict[str, str] = dict(CHIPS)


def chips_to_items(selected: list[str]) -> list[dict]:
    """선택한 칩들을 분석용 항목으로 바꾼다. 관점은 알 수 없으므로 중립(0.0)."""
    return [
        {"label": label, "topic": CHIP_TOPIC[label], "stance": 0.0, "note": "키워드 선택"}
        for label in selected
        if label in CHIP_TOPIC
    ]
