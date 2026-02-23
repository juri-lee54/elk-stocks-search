"""ELK-RAG 공통 모듈 (주식 검색용) — 고도화 버전"""

import os
import json
import datetime
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI

load_dotenv()

INDEX_NAME = "stock_info"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
CHAT_MODEL = "gpt-4o-mini"

# 신뢰도 등급 기준 (RRF 점수)
# RRF 최대 이론값: 1/61 + 1/61 ≈ 0.0328 (양쪽 1위 동시 달성)
# 실제 점수 범위를 반영해 등급 기준 설정
CONFIDENCE_THRESHOLDS = {"높음": 0.030, "보통": 0.016}


# ──────────────────────────────────────────
# 클라이언트
# ──────────────────────────────────────────
def get_es_client():
    """Elasticsearch 클라이언트 반환"""
    return Elasticsearch("http://localhost:9200", http_compress=True)


def get_openai_client():
    """OpenAI 클라이언트 반환. API 키 미설정 시 ValueError 발생."""
    api_key = os.getenv("AI_API_KEY")
    if not api_key:
        raise ValueError("AI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    return OpenAI(api_key=api_key)


# ──────────────────────────────────────────
# 임베딩
# ──────────────────────────────────────────
def get_embedding(openai_client, text: str) -> list:
    """텍스트를 임베딩 벡터로 변환"""
    response = openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


# ──────────────────────────────────────────
# 하이브리드 검색 (Semantic + Lexical → RRF)
# ──────────────────────────────────────────
def search_hybrid(es, openai_client, query: str, k: int = 5) -> list:
    """
    Reciprocal Rank Fusion(RRF)으로 시맨틱 + 렉시컬 검색 결과를 결합.

    반환: [{"회사명": str, "score": float, "rank": int}, ...]
      - score: RRF 점수 (높을수록 관련성 높음, 신뢰도 판단에 사용)
      - rank : 최종 순위 (1-based)
    """
    query_embedding = get_embedding(openai_client, query)

    # ── 시맨틱 검색 (KNN)
    sem_resp = es.search(
        index=INDEX_NAME,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": k,
            "num_candidates": k * 10,
        },
        source={"excludes": ["embedding", "combined_text"]},
        size=k,
    )

    # ── 렉시컬 검색 (multi_match, 회사명 가중치 높임)
    lex_resp = es.search(
        index=INDEX_NAME,
        query={
            "bool": {
                "should": [
                    {
                        # 회사명이 정확히 일치하면 boost=10으로 압도적 우선순위 부여
                        "term": {
                            "회사명.keyword": {
                                "value": query,
                                "boost": 10,
                            }
                        }
                    },
                    {
                        # 부분 매칭은 기존과 동일하게 유지 (업종·주요제품 포함)
                        "multi_match": {
                            "query": query,
                            "fields": ["회사명^3", "업종^2", "주요제품"],
                            "type": "best_fields",
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
        source={"excludes": ["embedding", "combined_text"]},
        size=k,
    )


    # ── RRF 점수 계산 (RRF_K=60 은 Elasticsearch 공식 권장값)
    RRF_K = 60
    rrf_scores: dict = {}
    name_map: dict = {}

    for rank, hit in enumerate(sem_resp["hits"]["hits"], start=1):
        doc_id = hit["_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank)
        name_map[doc_id] = hit["_source"]["회사명"]

    for rank, hit in enumerate(lex_resp["hits"]["hits"], start=1):
        doc_id = hit["_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank)
        name_map[doc_id] = hit["_source"]["회사명"]

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    return [
        {"회사명": name_map[doc_id], "score": round(score, 6), "rank": i + 1}
        for i, (doc_id, score) in enumerate(sorted_docs)
    ]


def get_confidence_label(score: float) -> tuple:
    """RRF 점수 → (신뢰도 레이블, 이모지 색상)"""
    if score >= CONFIDENCE_THRESHOLDS["높음"]:
        return "높음", "🟢"
    elif score >= CONFIDENCE_THRESHOLDS["보통"]:
        return "보통", "🟡"
    else:
        return "낮음", "🔴"


# ──────────────────────────────────────────
# 상세 정보 조회
# ──────────────────────────────────────────
def search_stock_details(es, company_names: list) -> list:
    """회사명 목록으로 상세 정보 조회 (임베딩·combined_text 제외)"""
    if not company_names:
        return []
    result = es.search(
        index=INDEX_NAME,
        query={"terms": {"회사명.keyword": company_names}},
        source={"excludes": ["embedding", "combined_text"]},
        size=len(company_names),
    )
    return [hit["_source"] for hit in result["hits"]["hits"]]


# ──────────────────────────────────────────
# 주가 질문 감지 & 날짜 파싱 (멀티턴 맥락 포함)
# ──────────────────────────────────────────
def detect_price_query(openai_client, query: str, chat_history: list) -> dict:
    """
    이전 대화 맥락을 고려해 주가 조회 여부와 날짜 범위를 추출한다.

    Parameters
    ----------
    chat_history : [{"role": "user"|"assistant", "content": str}, ...]

    Returns
    -------
    {"is_price_query": bool, "start_date": str|None, "end_date": str|None}
    """
    today = datetime.date.today().isoformat()
    system_prompt = (
        "당신은 대화 맥락을 분석하는 도우미입니다. "
        "이전 대화와 현재 질문을 종합하여 아래 JSON만 반환하세요. 다른 텍스트는 절대 포함하지 마세요.\n\n"
        "{\n"
        '  "is_price_query": true/false,\n'
        '  "start_date": "YYYY-MM-DD 또는 null",\n'
        '  "end_date": "YYYY-MM-DD 또는 null"\n'
        "}\n\n"
        "is_price_query: 주가, 차트, 시세, 가격 추이 요청이면 true.\n"
        "start_date: 명시된 시작일. 없으면 null (→ 상장일 기준).\n"
        f"end_date: 명시된 종료일. 없으면 null (→ 오늘 {today}).\n"
        "연도만 있으면 해당 연도 1월 1일 / 12월 31일로 변환하세요.\n"
        "이전 대화에서 언급된 기간이 현재 질문에 암묵적으로 적용될 수 있습니다."
    )

    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": query})

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"is_price_query": False, "start_date": None, "end_date": None}


# ──────────────────────────────────────────
# RAG 답변 (멀티턴 + 하이브리드 검색 + 신뢰도)
# ──────────────────────────────────────────
def answer_question(
    es,
    openai_client,
    query: str,
    chat_history: list = None,
) -> tuple:
    """
    하이브리드 RAG 기반 질문 답변.

    Parameters
    ----------
    es            : Elasticsearch 클라이언트
    openai_client : OpenAI 클라이언트
    query         : 현재 사용자 입력
    chat_history  : [{"role": ..., "content": ...}] 이전 대화 (기본값: [])

    Returns
    -------
    (answer: str, scored_docs: list[dict], price_info: dict)
      scored_docs 예시: [{"회사명": "삼성전자", "score": 0.0312, "rank": 1}, ...]
      price_info  예시: {"is_price_query": True, "start_date": "2023-01-01", "end_date": None}
    """
    if chat_history is None:
        chat_history = []

    # ① 주가 질문 여부 (대화 맥락 포함)
    price_info = detect_price_query(openai_client, query, chat_history)

    # ② 하이브리드 검색 → 점수 포함 결과
    scored_docs = search_hybrid(es, openai_client, query, k=5)

    if not scored_docs:
        return "죄송합니다. 질문과 관련된 주식 종목을 찾을 수 없습니다.", [], price_info

    company_names = [d["회사명"] for d in scored_docs]
    detail_info = search_stock_details(es, company_names)

    # ③ 시스템 프롬프트
    if price_info.get("is_price_query"):
        system_content = (
            "당신은 친절하고 전문적인 주식 탐색 비서입니다. "
            "사용자가 주가(시세) 정보를 요청했습니다. "
            "검색된 종목의 회사명, 업종, 주요제품을 간략히 소개하고, "
            "주가 차트는 별도로 표시될 예정임을 안내해 주세요. "
            "응답은 300자 이내로 간결하게 작성하세요."
        )
    else:
        system_content = (
            "당신은 친절하고 전문적인 주식 탐색 비서입니다. "
            "제공된 '회사 상세 정보'를 바탕으로 사용자의 질문에 답변하세요. "
            "검색된 회사의 주요 제품이나 업종 특징을 엮어서 자연스러운 한국어로 설명해 주세요. "
            "이전 대화 흐름을 고려해 자연스럽게 이어서 답변하세요. "
            "응답은 500자 이내로 핵심만 요약해서 답변하세요."
        )

    # ④ 멀티턴: 이전 대화 + 현재 컨텍스트 결합
    context_str = ", ".join(company_names)
    user_prompt = (
        f"검색된 관련 회사들: {context_str}\n\n"
        f"회사 상세 정보: {detail_info}\n\n"
        f"사용자 질문: {query}\n\n"
        f"답변:"
    )

    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    messages = [{"role": "system", "content": system_content}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": user_prompt})

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )
    answer = response.choices[0].message.content
    return answer, scored_docs, price_info