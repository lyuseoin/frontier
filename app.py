"""나의 필터버블 진단 대시보드 — Streamlit 앱.

캡스톤 계획서(docs/plan.md)의 '나의 필터버블 진단 대시보드'를 구현한다.
사용자가 최근 소비한 미디어 키워드를 입력하면, 계획서가 제시한 세 가지 인공지능
요소로 정보 편식 상태를 진단하고 시각화한다.

  1) 자연어 처리·텍스트 마이닝 : 키워드의 주제·관점을 AI가 분류 (core.gemini)
  2) K-Means 군집화            : 소비 패턴이 좌표 공간에 얼마나 밀집됐는지 (core.analyze)
  3) 역발상 콘텐츠 필터링       : 유사도가 가장 낮은 양질의 대조군 매칭 (core.recommend)

어떤 관점이 옳고 그른지는 판단하지 않는다. Gemini 호출은 '진단하기' 때 한 번만 한다.
"""

from __future__ import annotations

import math
from collections import defaultdict

import plotly.graph_objects as go
import streamlit as st

from core.analyze import cluster_map, load_media, topic_ratio
from core.gemini import classify, classify_image, read_persona
from core.inputs import CHIP_LABELS, chips_to_items
from core.persona import TOPIC_COLORS, persona as judge_persona
from core.recommend import counter_cards, echo_cards

METHOD_CHIP = "🏷️ 키워드 선택"
METHOD_TEXT = "🔤 직접 입력"
METHOD_IMAGE = "📸 화면 캡처"

BLUE, GRAY, GRID, INK = "#2F6BFF", "#9AA3B2", "#EEF0F5", "#1F2430"
RED, YELLOW, GREEN = "#FF4D4F", "#F5A623", "#22C55E"
CLUSTER_PALETTE = ["#2F6BFF", "#F5A623", "#22C55E", "#F472B6", "#38BDF8"]

st.set_page_config(
    page_title="나의 필터버블 진단", page_icon="🫧", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #F4F6FB; }
      div[data-testid="stMetric"],
      div[data-testid="stVerticalBlockBorderWrapper"] > div:has(> div[data-testid="stVerticalBlock"]) {
        background: #fff; border-radius: 16px;
      }
      div[data-testid="stMetric"] { padding: 16px 18px; box-shadow: 0 2px 8px rgba(0,0,0,.05); }
      div[data-testid="stMetricValue"] { font-size: 26px; font-weight: 800; }
      div[data-testid="stMetricLabel"] { color: #9AA3B2; font-size: 13px; }
      div[data-testid="stButton"] > button { min-height: 44px; border-radius: 12px; font-weight: 700; }
      .type-tag {
        display: inline-block; border-radius: 999px; padding: 4px 12px;
        font-size: 13px; font-weight: 700; color: #fff; margin-right: 6px;
      }
      .feed-row { font-size: 15px; line-height: 1.9; }
      .callout {
        border-left: 4px solid; border-radius: 8px; padding: 12px 16px;
        background: #fff; font-size: 15px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_media() -> dict:
    return load_media()


MEDIA = get_media()
TOPICS = list(MEDIA["axes"].keys())

for key, default in [
    ("items", None), ("how", ""), ("persona", None), ("read", None), ("rhow", ""), ("_warn", ""),
]:
    st.session_state.setdefault(key, default)


def score_color(score: int) -> str:
    return RED if score >= 67 else YELLOW if score >= 34 else GREEN


def radar_chart(items: list[dict], color: str) -> go.Figure:
    """① 주제 분포 — 방사형 그래프 (계획서: '방사형 그래프 등 시각 대시보드')."""
    ratio = topic_ratio(items, TOPICS)
    values = [ratio[t] for t in TOPICS]
    fig = go.Figure(
        go.Scatterpolar(
            r=values + values[:1],
            theta=TOPICS + TOPICS[:1],
            fill="toself",
            line=dict(color=color, width=2),
            fillcolor="rgba(47,107,255,.16)",
            hovertemplate="%{theta}: %{r:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="#fff",
            radialaxis=dict(visible=True, range=[0, max(values + [0.4])], showticklabels=False, gridcolor=GRID),
            angularaxis=dict(gridcolor=GRID, tickfont=dict(size=12, color=GRAY)),
        ),
        showlegend=False, height=320, margin=dict(l=50, r=50, t=24, b=24),
        paper_bgcolor="#fff",
    )
    return fig


def cluster_chart(items: list[dict]) -> go.Figure:
    """② K-Means 군집도 — 소비 패턴이 좌표 공간에 얼마나 밀집됐는지 (계획서 AI 요소).

    같은 (주제·관점)이면 PCA 좌표가 완전히 겹친다. 그대로 두면 점과 글씨가 포개져
    읽을 수 없으므로, 겹치는 점들을 작은 링으로 흩고 라벨 대신 점 안에 번호를 찍는다.
    (번호 ↔ 키워드 매핑은 차트 아래 범례가 담당한다.)
    """
    result = cluster_map(items, TOPICS, k=3)
    coords, labels = result["coords"], result["labels"]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    # 좌표가 겹치는 점들을 같은 중심 둘레의 작은 원으로 분산 (결정론적)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for idx in range(len(xs)):
        groups[(round(xs[idx], 3), round(ys[idx], 3))].append(idx)
    ring = 0.16
    for (cx, cy), idxs in groups.items():
        if len(idxs) < 2:
            continue
        for k, idx in enumerate(idxs):
            angle = 2 * math.pi * k / len(idxs)
            xs[idx] = cx + ring * math.cos(angle)
            ys[idx] = cy + ring * math.sin(angle)

    colors = [CLUSTER_PALETTE[lbl % len(CLUSTER_PALETTE)] for lbl in labels]
    numbers = [str(i + 1) for i in range(len(items))]
    hover = [
        f"{i + 1}. {str(items[i].get('label', items[i].get('title', '')))} · "
        f"{items[i]['topic']} {items[i]['stance']:+.1f}"
        for i in range(len(items))
    ]
    fig = go.Figure(
        go.Scatter(
            x=xs, y=ys, mode="markers+text",
            text=numbers, textposition="middle center",
            textfont=dict(size=12, color="#fff"),
            marker=dict(size=26, color=colors, line=dict(width=2, color="#fff")),
            hovertext=hover, hoverinfo="text", cliponaxis=False,
        )
    )
    pad = 0.45
    fig.update_layout(
        height=320, margin=dict(l=20, r=20, t=24, b=20),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[min(xs) - pad, max(xs) + pad]),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, showticklabels=False,
                   range=[min(ys) - pad, max(ys) + pad]),
    )
    return fig


def cluster_legend(items: list[dict]) -> str:
    """군집도의 번호 ↔ 키워드 범례 HTML. 번호·색은 차트의 점과 일치한다."""
    cl = cluster_map(items, TOPICS, k=3)
    parts = []
    for i, it in enumerate(items):
        c = CLUSTER_PALETTE[cl["labels"][i] % len(CLUSTER_PALETTE)]
        name = str(it.get("label", it.get("title", "")))[:16]
        parts.append(
            f"<span style='background:{c}22;color:{c};border-radius:7px;"
            f"padding:2px 8px;margin:3px 5px 0 0;font-size:12px;display:inline-block'>"
            f"{i + 1} {name}</span>"
        )
    return "".join(parts)


def stance_chart(items: list[dict]) -> go.Figure:
    """관점 분포 — 각 콘텐츠의 관점을 한 축에 찍어 확증편향(치우침)을 드러낸다."""
    xs = [i["stance"] for i in items]
    ys = [(idx % 5) * 0.12 - 0.24 for idx in range(len(items))]  # 겹침 방지용 약한 세로 분산
    colors = [TOPIC_COLORS.get(i["topic"], GRAY) for i in items]
    mean = sum(xs) / len(xs) if xs else 0.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=15, color=colors, line=dict(width=2, color="#fff")),
            text=[f"{i['topic']} {i['stance']:+.1f}" for i in items], hoverinfo="text",
        )
    )
    fig.add_vline(x=0, line=dict(color=GRID, width=1))
    fig.add_vline(x=mean, line=dict(color=INK, width=2, dash="dot"))
    fig.add_annotation(x=mean, y=0.45, text=f"평균 {mean:+.2f}", showarrow=False,
                       font=dict(size=12, color=INK))
    fig.update_layout(
        height=210, margin=dict(l=20, r=20, t=30, b=30),
        paper_bgcolor="#fff", plot_bgcolor="#fff", showlegend=False,
        xaxis=dict(range=[-1.12, 1.12], tickvals=[-1, 0, 1],
                   ticktext=["◀ 한쪽 관점", "중립", "반대쪽 ▶"],
                   tickfont=dict(size=12, color=GRAY), zeroline=False, gridcolor=GRID),
        yaxis=dict(visible=False, range=[-0.6, 0.6]),
    )
    return fig


def _finish_diagnosis(items: list[dict], how: str) -> None:
    """분류된 항목으로 유형·풀이·추천을 만들어 세션에 담는다. (공통 마무리 단계)"""
    st.session_state["_warn"] = ""
    with st.spinner("알고리즘이 만든 필터버블을 진단하는 중… 🫧"):
        p = judge_persona(items, MEDIA)
        echo = echo_cards(items, MEDIA, top_n=4)
        counter = counter_cards(items, MEDIA, top_n=3)
        read, rhow = read_persona(items, p, echo, counter)
    st.session_state.update(items=items, how=how, persona=p, read=read, rhow=rhow)


def run_text(raw: str) -> None:
    """① 직접 입력 — 키워드를 Gemini 로 분류한다."""
    keywords = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(keywords) < 3:
        st.session_state["_warn"] = "키워드를 최소 3개 이상 적어주세요. 많을수록 진단이 정확해져요."
        return
    with st.spinner("키워드를 분석하는 중… 🔤"):
        items, how = classify(keywords, MEDIA)
    _finish_diagnosis(items, how)


def run_chips(selected: list[str]) -> None:
    """② 키워드 선택 — 칩을 주제로 매핑한다(관점은 중립). Gemini 없이 즉시."""
    selected = selected or []
    if len(selected) < 3:
        st.session_state["_warn"] = "관심 있게 본 주제를 3개 이상 골라주세요."
        return
    _finish_diagnosis(chips_to_items(selected), "chips")


def run_image(upload) -> None:
    """③ 화면 캡처 — 이미지를 Gemini 비전으로 읽어 분류한다."""
    if upload is None:
        st.session_state["_warn"] = "유튜브·인스타 등 화면 캡처 이미지를 먼저 첨부해주세요."
        return
    with st.spinner("화면 속 콘텐츠를 읽는 중… 📸"):
        items, how = classify_image(upload.getvalue(), upload.type, MEDIA)
    if len(items) < 2:
        st.session_state["_warn"] = (
            "화면에서 콘텐츠를 충분히 읽지 못했어요. 제목이 잘 보이는 화면을 캡처하거나 "
            "다른 입력 방법을 써보세요."
        )
        return
    _finish_diagnosis(items, how)


# ── 사이드바 ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🫧 나의 필터버블 진단")
    st.caption("추천 알고리즘이 나를 어떤 방에 가뒀는지, 미디어 소비 습관으로 진단해요")
    st.divider()

    st.markdown("**최근 본 미디어를 어떻게 알려줄까요?**")
    method = st.radio(
        "입력 방법", [METHOD_CHIP, METHOD_TEXT, METHOD_IMAGE],
        label_visibility="collapsed", key="input_method",
    )

    if method == METHOD_CHIP:
        st.caption("관심 있게 본 주제를 눌러주세요 (여러 개, 3개 이상)")
        selected = st.pills(
            "주제", CHIP_LABELS, selection_mode="multi",
            label_visibility="collapsed", key="chips",
        )
        if st.button("진단하기 🫧", type="primary", use_container_width=True):
            run_chips(selected)
        st.caption("※ 선택형은 관심 '주제'만 봐요. 관점 쏠림까지 보려면 직접 입력이나 화면 캡처를 써보세요.")

    elif method == METHOD_TEXT:
        st.caption("최근 본 영상·기사 제목이나 키워드를 한 줄에 하나씩")
        raw = st.text_area(
            "한 줄에 하나씩 적어주세요",
            placeholder="AI 주식 투자로 돈 버는 법\n전기차 보조금 확대해야\n주 4일제 도입 찬성\n스타트업 규제 완화",
            height=170, label_visibility="collapsed", key="raw_input",
        )
        if st.button("진단하기 🫧", type="primary", use_container_width=True):
            run_text(raw)

    else:  # METHOD_IMAGE
        st.caption("유튜브·인스타 등 추천 화면을 캡처해서 올려주세요")
        upload = st.file_uploader(
            "화면 캡처", type=["png", "jpg", "jpeg"],
            label_visibility="collapsed", key="shot",
        )
        if upload is not None:
            st.image(upload, caption="이 화면을 분석해요", use_container_width=True)
        if st.button("진단하기 🫧", type="primary", use_container_width=True):
            run_image(upload)
        st.caption("※ 이미지는 분석에만 쓰고 저장하지 않아요.")

    if st.session_state.get("_warn"):
        st.warning(st.session_state["_warn"])
    if st.session_state["persona"]:
        if st.button("다시 진단", use_container_width=True):
            st.session_state.update(items=None, persona=None, read=None, rhow="", _warn="")
            st.rerun()

    st.divider()
    with st.expander("이 진단은 어떻게 작동하나요?"):
        st.markdown(
            "1. **키워드 분류(자연어 처리·비전)** — 입력·화면 속 콘텐츠의 주제·관점을 AI가 분류해요.\n"
            "2. **K-Means 군집화** — 소비 패턴이 좌표 공간에 얼마나 밀집됐는지 계산해요.\n"
            "3. **역발상 필터링** — 당신과 가장 결이 먼 양질의 콘텐츠를 골라 매칭해요."
        )
    st.caption("적은 내용은 서버에 저장되지 않고, 브라우저를 닫으면 사라져요.")


# ── 본문 ────────────────────────────────────────────────────────────────────
p = st.session_state["persona"]

if not p:
    st.title("🫧 나의 필터버블 진단")
    st.write("요즘 본 영상·기사의 제목이나 키워드를 왼쪽에 **3개 이상** 적고 **진단하기**를 눌러보세요.")
    st.info(
        "현대 청소년은 추천 알고리즘이 주는 정보에 강하게 의존하며 **필터버블·확증편향**에 노출돼요. "
        "이 도구는 어떤 생각이 옳고 그른지 판단하지 않아요. "
        "당신의 정보 소비가 **얼마나 한쪽에 가두어져 있는지**만 비춰줘요.",
        icon="🫧",
    )
    st.stop()

items = st.session_state["items"]
read = st.session_state["read"]
color = p["color"]
score = p["bias_score"]
sev_color = score_color(score)
topic_count = len({i["topic"] for i in items})
verdict = (
    "정보가 한쪽에 크게 몰려 있어요" if score >= 67
    else "어느 정도 쏠림이 보여요" if score >= 34
    else "비교적 고르게 보고 있어요"
)

if st.session_state["how"].startswith("fallback"):
    st.warning(
        f"AI 분류를 못 써서 임시 규칙으로 진단했어요 ({st.session_state['how']}). 결과가 부정확할 수 있어요.",
        icon="⚠️",
    )

st.title("🫧 나의 필터버블 진단")

# 상단 지표 — 필터버블 지수가 headline
c1, c2, c3 = st.columns([1, 1, 1])
c1.metric("필터버블 지수", f"{score} / 100", verdict, delta_color="off")
c2.metric("필터버블 유형", f"{p['emoji']} {p['name']}")
c3.metric("살펴본 주제", f"{topic_count} / {len(TOPICS)}")

# 심각도 미터
st.markdown(
    f"<div style='height:12px;background:{GRID};border-radius:8px;overflow:hidden;margin:6px 0 4px'>"
    f"<div style='height:100%;width:{score}%;background:{sev_color}'></div></div>"
    f"<div style='display:flex;justify-content:space-between;color:{GRAY};font-size:12px;margin-bottom:14px'>"
    f"<span>🌈 열린 방</span><span>🕳️ 갇힌 방</span></div>",
    unsafe_allow_html=True,
)

# 진단 + 주제 분포
left, right = st.columns([1.15, 1])
with left:
    with st.container(border=True):
        st.markdown(
            f"<span class='type-tag' style='background:{color}'>{p['emoji']} {p['name']}</span>"
            f"<span style='color:{GRAY};font-size:13px'>주제 폭 {p['topic_level']} · 관점 {p['stance_level']}</span>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("**🔍 진단**")
        st.write(read["diagnosis"])
with right:
    with st.container(border=True):
        st.markdown("**🗺️ 주제 분포 (방사형)**")
        st.caption("한쪽으로 뾰족할수록 그 주제에 갇혀 있다는 뜻")
        st.plotly_chart(radar_chart(items, color), use_container_width=True)

# K-Means 군집도 + 관점 분포
g1, g2 = st.columns([1, 1])
with g1:
    with st.container(border=True):
        st.markdown("**🧩 소비 패턴 군집도 (K-Means)**")
        st.caption("점이 뭉쳐 있을수록 비슷한 콘텐츠만 봤다는 뜻이에요")
        st.plotly_chart(cluster_chart(items), use_container_width=True)
        st.markdown(cluster_legend(items), unsafe_allow_html=True)
with g2:
    with st.container(border=True):
        st.markdown("**⚖️ 관점 분포**")
        st.caption("점선(평균)이 가운데에서 멀수록 한쪽 관점으로 쏠려 있어요 (확증편향)")
        st.plotly_chart(stance_chart(items), use_container_width=True)
        legend = " ".join(
            f"<span style='color:{TOPIC_COLORS.get(t, GRAY)};font-size:12px'>●{t}</span>"
            for t in sorted({i['topic'] for i in items})
        )
        st.markdown(legend, unsafe_allow_html=True)

st.divider()

# ③ 에코 체임버 — 알고리즘이 계속 밀어주는 것
st.subheader("🔁 알고리즘이 앞으로도 계속 밀어줄 콘텐츠")
st.caption("당신이 본 것과 가장 비슷한 것들 — 이렇게 필터버블은 점점 두꺼워져요")
if read.get("feed"):
    st.markdown(
        "<div class='feed-row'>" + "".join(f"• {f}<br>" for f in read["feed"]) + "</div>",
        unsafe_allow_html=True,
    )
echo = echo_cards(items, MEDIA, top_n=3)
cols = st.columns(3)
for col, card in zip(cols, echo):
    with col, st.container(border=True):
        st.markdown(f"**{card['title']}**")
        similarity = (card["similarity"] + 1) / 2
        st.caption(f"{card['topic']} · 지금 취향과 {similarity:.0%} 닮음")
        st.write(card["summary"])

st.divider()

# ③ 역발상 필터링 — 필터버블 밖 (사각지대)
st.subheader("🌱 필터버블 밖, 알고리즘이 잘 안 보여주는 이야기")
if read.get("action"):
    st.markdown(f"<span style='color:{BLUE};font-weight:600'>💡 {read['action']}</span>", unsafe_allow_html=True)
st.caption("역발상 필터링으로 고른, 당신과 결이 가장 먼 양질의 콘텐츠예요. 한 번씩 열어보면 방이 넓어져요.")
counter = counter_cards(items, MEDIA, top_n=3)
cols = st.columns(3)
for col, card in zip(cols, counter):
    with col, st.container(border=True):
        st.markdown(f"**{card['title']}**")
        distance = (1 - card["similarity"]) / 2
        st.caption(f"{card['topic']} · 결이 먼 정도 {distance:.0%}")
        st.write(card["summary"])
        st.markdown(f"<span style='color:{GRAY};font-size:13px'>🧭 {card['reason']}</span>", unsafe_allow_html=True)

# 이대로라면
if read.get("forecast"):
    st.write("")
    st.markdown(
        f"<div class='callout' style='border-color:{sev_color}'>"
        f"<b>⏳ 이대로라면</b><br>{read['forecast']}</div>",
        unsafe_allow_html=True,
    )

st.caption(
    "※ 이 진단은 참고용이에요. 보이지 않던 알고리즘의 편향을 스스로 인지하고, "
    "비판적 미디어 리터러시를 기르는 것이 목적이에요."
)
