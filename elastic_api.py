from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# 🌟 1. 검색어 변환용 임베딩 모델 로드 (서버 켤 때 한 번만 로드됨)
print("검색용 AI 모델을 불러오는 중입니다...")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

ES_HOST = 'http://localhost:9200'
client = Elasticsearch(ES_HOST)

def search_stocks_vector(query_text, max_results=5):
    """
    사용자의 질문을 벡터로 변환하여 엘라스틱서치에서 KNN(유사도) 검색 수행
    """
    # 2. 사용자가 입력한 자연어 검색어를 768차원 숫자 벡터로 변환
    query_vector = model.encode(query_text).tolist()
    
    # 3. 엘라스틱서치 KNN(K-Nearest Neighbors) 검색 쿼리 구성
    knn_query = {
        "field": "embedding",       # 벡터가 저장된 필드명
        "query_vector": query_vector, # 사용자의 검색어 벡터
        "k": max_results,           # 최종적으로 가져올 결과 수
        "num_candidates": 100       # 유사도 계산을 수행할 후보군 수
    }
    
    # 4. 검색 실행 (무거운 embedding 필드는 빼고 필요한 텍스트 정보만 가져옴)
    response = client.search(
        index="stock_info",
        knn=knn_query,
        _source=["회사명", "업종", "주요제품", "종목코드"] 
    )
    
    return response