import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import FinanceDataReader as fdr
import rag_module

st.set_page_config(page_title="주식 종목 탐색", page_icon="📈", layout="wide")
st.title("주식 종목 탐색 서비스")
st.markdown("관심 있는 테마나 종목을 질문해 보세요. AI가 검색 결과를 바탕으로 요약 답변을 드립니다.")


# ──────────────────────────────────────────
# 클라이언트 초기화
# ──────────────────────────────────────────
@st.cache_resource
def init_clients():
    try:
        es_client = rag_module.get_es_client()
        oa_client = rag_module.get_openai_client()
        return es_client, oa_client
    except Exception as e:
        st.error(f"❌ 클라이언트 초기화 실패: {str(e)}")
        st.stop()


es, openai_client = init_clients()


# ──────────────────────────────────────────
# 세션 상태 초기화
# ──────────────────────────────────────────
INIT_MSG = {"role": "assistant", "content": "안녕하세요! 주식 관련 정보를 물어보세요. 특정 종목, 테마, 업종 등 무엇이든 질문해 주세요 😊"}

if "messages" not in st.session_state:
    st.session_state.messages = [INIT_MSG]

# LLM에 전달할 대화 이력 (role/content만 포함, UI 전용 키 제외)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ──────────────────────────────────────────
# 주가 차트 헬퍼
# ──────────────────────────────────────────
def fetch_price_data(details: list, start_date: str | None, end_date: str | None) -> dict:
    """
    종목 리스트의 주가 데이터를 딕셔너리로 반환.
    반환: {회사명: DataFrame(Date, Close)}
    """
    today = datetime.date.today().isoformat()
    end = end_date or today
    result = {}

    for stock in details:
        code = stock.get("종목코드")
        name = stock.get("회사명", code)
        listing_date = stock.get("상장일")

        if start_date:
            start = start_date
        elif listing_date:
            start = str(listing_date).replace(".", "-").replace("/", "-")[:10]
        else:
            start = (datetime.date.today() - datetime.timedelta(days=3650)).isoformat()

        if not code:
            continue
        try:
            df_price = fdr.DataReader(code, start, end).reset_index()
            if df_price.empty:
                continue
            date_col = "Date" if "Date" in df_price.columns else df_price.columns[0]
            df_price = df_price.rename(columns={date_col: "Date"})
            result[name] = df_price[["Date", "Close"]]
        except Exception as e:
            st.warning(f"⚠️ {name}({code}) 주가 조회 실패: {e}")

    return result


def render_price_chart(
    price_data: dict,
    start_date: str | None,
    end_date: str | None,
    compare_mode: bool = False,
):
    """
    주가 시계열 차트 렌더링.

    compare_mode=True : 기준일 종가=100으로 정규화해 상대 수익률 비교
    compare_mode=False: 절대 종가(원) 표시
    """
    if not price_data:
        st.info("📭 주가 데이터를 가져올 수 없습니다.")
        return

    today = datetime.date.today().isoformat()
    end = end_date or today
    period_label = f"{start_date or '상장일'} ~ {end}"

    fig = go.Figure()

    for name, df in price_data.items():
        if df.empty:
            continue

        if compare_mode:
            base = df["Close"].iloc[0]
            if base == 0:
                continue
            y_values = (df["Close"] / base * 100).round(2)
            hover_suffix = "%"
            hover_fmt = ".2f"
            yaxis_title = "상대 수익률 (기준일=100)"
        else:
            y_values = df["Close"]
            hover_suffix = "원"
            hover_fmt = ",.0f"
            yaxis_title = "종가 (원)"

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=y_values,
                mode="lines",
                name=name,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"날짜: %{{x|%Y-%m-%d}}<br>"
                    f"{'수익률' if compare_mode else '종가'}: %{{y:{hover_fmt}}}{hover_suffix}"
                    "<extra></extra>"
                ),
            )
        )

    chart_title = (
        f"📊 {'상대 수익률 비교' if compare_mode else '주가 시계열'} 차트 ({period_label})"
    )
    fig.update_layout(
        title=chart_title,
        xaxis_title="날짜",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
        template="plotly_white",
    )

    # 비교 모드일 때 기준선(100) 표시
    if compare_mode:
        fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)

    st.plotly_chart(fig, use_container_width=True)


def render_chart_section(chart_params: dict):
    """차트 섹션 전체 렌더링 (모드 토글 포함). chart_params는 세션에 저장된 dict."""
    details = chart_params["details"]
    start_date = chart_params["start_date"]
    end_date = chart_params["end_date"]

    price_data = fetch_price_data(details, start_date, end_date)
    if not price_data:
        return

    # 종목이 2개 이상일 때만 비교 모드 토글 표시
    multi = len(price_data) > 1
    compare_mode = False
    if multi:
        compare_mode = st.toggle(
            "📊 상대 수익률 비교 모드",
            value=chart_params.get("compare_mode", False),
            key=f"toggle_{chart_params.get('_key', id(chart_params))}",
            help="ON: 기준일 종가=100 정규화 / OFF: 절대 종가(원)",
        )
        chart_params["compare_mode"] = compare_mode  # 상태 반영

    render_price_chart(price_data, start_date, end_date, compare_mode)


# ──────────────────────────────────────────
# 신뢰도 배지 렌더링 (중복 제거 — 1개만 유지)
# ──────────────────────────────────────────
def render_confidence_badges(scored_docs: list):
    """검색된 종목별 신뢰도 카드를 가로로 표시"""
    if not scored_docs:
        return

    # 등급별 색상 매핑
    COLOR_MAP = {"높음": "#2ecc71", "보통": "#f39c12", "낮음": "#e74c3c"}
    BG_MAP    = {"높음": "#eafaf1", "보통": "#fef9e7", "낮음": "#fdedec"}

    cols = st.columns(len(scored_docs))
    for col, doc in zip(cols, scored_docs):
        label, emoji = rag_module.get_confidence_label(doc["score"])
        color = COLOR_MAP[label]
        bg    = BG_MAP[label]
        with col:
            st.markdown(
                f"""
                <div style="
                    background:{bg};
                    border:1.5px solid {color};
                    border-radius:10px;
                    padding:10px 12px;
                    text-align:center;
                    line-height:1.6;
                ">
                    <div style="font-size:0.78rem;color:#555;margin-bottom:2px;">
                        {doc['rank']}위
                    </div>
                    <div style="font-size:0.95rem;font-weight:700;color:#222;">
                        {doc['회사명']}
                    </div>
                    <div style="
                        display:inline-block;
                        margin-top:6px;
                        padding:2px 10px;
                        border-radius:20px;
                        background:{color};
                        color:white;
                        font-size:0.8rem;
                        font-weight:600;
                    ">
                        {emoji} 신뢰도 {label}
                    </div>
                    <div style="font-size:0.72rem;color:#888;margin-top:5px;">
                        RRF 점수: {doc['score']:.4f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────
with st.sidebar:
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = [INIT_MSG]
        st.session_state.chat_history = []
        st.rerun()

    st.header("⚙️ 서비스 설명")

    st.markdown("**🔍 검색 방식**")
    st.info("Hybrid Search (시맨틱 + 렉시컬 RRF)\n\n두 검색 방식의 결과를 자동으로 결합하여 최적의 결과를 제공합니다.", icon="ℹ️")

    st.markdown("---")
    st.markdown("**🎯 검색 신뢰도 기준**")
    st.markdown(
        "신뢰도는 **RRF 점수** 기반입니다.  \n"
        "시맨틱 + 렉시컬 두 검색의 순위 점수를 합산하며,  \n"
        "`점수 = 1/(60 + 순위)` 공식으로 계산됩니다."
    )
    st.markdown(
        "| 등급 | 기준 | 의미 |\n"
        "|------|------|------|\n"
        "| 🟢 높음 | ≥ 0.030 | 두 검색 모두 상위권 |\n"
        "| 🟡 보통 | ≥ 0.016 | 한 검색에서 상위권 |\n"
        "| 🔴 낮음 | < 0.016 | 한 검색 하위권만 |"
    )

    st.markdown("---")
    st.markdown(
        "**💡 사용 팁**\n\n"
        "- 테마: `여름 관련주 알려줘`\n"
        "- 종목 주가: `삼성전자 2023년 주가`\n"
        "- 비교: `SK하이닉스와 삼성전자 비교`\n"
        "- 후속 질문: `그 중 반도체 관련만 보여줘`"
    )


# ──────────────────────────────────────────
# 기존 대화 내용 출력
# ──────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 신뢰도 배지
        if "scored_docs" in msg and msg["scored_docs"]:
            with st.expander("🎯 검색 신뢰도 보기"):
                render_confidence_badges(msg["scored_docs"])

        # 상세 데이터 테이블
        if "df" in msg and not msg["df"].empty:
            with st.expander("📋 상세 데이터 보기"):
                st.dataframe(
                    msg["df"].style.format({"신뢰도점수": "{:.4f}"}),
                    use_container_width=True,
                )

        # 주가 차트 재렌더링
        if "chart_params" in msg:
            render_chart_section(msg["chart_params"])


# ──────────────────────────────────────────
# 채팅 입력 및 처리
# ──────────────────────────────────────────
if prompt := st.chat_input("메시지를 입력하세요 (예: 2차전지 관련주 비교해줘)"):

    # ── 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI가 데이터를 찾고 답변을 생성 중입니다... 🧠"):
            try:
                # ① RAG 답변 (멀티턴 이력 전달)
                answer, scored_docs, price_info = rag_module.answer_question(
                    es,
                    openai_client,
                    prompt,
                    chat_history=st.session_state.chat_history[:-1],
                )

                # ② 텍스트 답변 출력
                st.markdown(answer)

                # ③ 신뢰도 배지
                if scored_docs:
                    with st.expander("🎯 검색 신뢰도 보기"):
                        render_confidence_badges(scored_docs)
                    company_names = [d["회사명"] for d in scored_docs]
                    st.caption(f"🏢 참조 기업: {', '.join(company_names)}")

                # ④ 상세 데이터 테이블 (신뢰도 기준 정렬)
                df = pd.DataFrame()
                details = []
                if scored_docs:
                    company_names = [d["회사명"] for d in scored_docs]
                    details = rag_module.search_stock_details(es, company_names)
                    if details:
                        df = pd.DataFrame(details)
                        HIDDEN_COLS = ["combined_text", "embedding"]
                        df = df.drop(columns=[c for c in HIDDEN_COLS if c in df.columns])

                        # 🌟 신뢰도 점수·등급 컬럼 추가 후 내림차순 정렬
                        score_map = {d["회사명"]: d["score"] for d in scored_docs}
                        df["신뢰도점수"] = df["회사명"].map(score_map)
                        df["신뢰도"] = df["신뢰도점수"].apply(
                            lambda s: rag_module.get_confidence_label(s)[1]
                                      + " " + rag_module.get_confidence_label(s)[0]
                        )
                        df = df.sort_values("신뢰도점수", ascending=False).reset_index(drop=True)

                        # 신뢰도 컬럼을 맨 앞으로
                        priority = ["신뢰도", "신뢰도점수", "종목코드", "회사명", "업종", "주요제품"]
                        ordered = priority + [c for c in df.columns if c not in priority]
                        df = df[[c for c in ordered if c in df.columns]]

                        with st.expander("📋 상세 데이터 보기"):
                            st.dataframe(
                                df.style.format({"신뢰도점수": "{:.4f}"}),
                                use_container_width=True,
                            )

                # ⑤ 주가 차트 (주가 질문일 때)
                chart_params = None
                if price_info.get("is_price_query") and details:
                    st.markdown("---")
                    chart_params = {
                        "details": details,
                        "start_date": price_info.get("start_date"),
                        "end_date": price_info.get("end_date"),
                        "compare_mode": False,
                        "_key": len(st.session_state.messages),
                    }
                    render_chart_section(chart_params)

                # ⑥ 세션 저장
                session_msg = {
                    "role": "assistant",
                    "content": answer,
                    "scored_docs": scored_docs,
                    "df": df,
                }
                if chart_params:
                    session_msg["chart_params"] = chart_params

                st.session_state.messages.append(session_msg)

                # LLM 이력에는 순수 텍스트만 저장
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"❌ 오류가 발생했습니다: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})