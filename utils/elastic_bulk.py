from elasticsearch import Elasticsearch, helpers
import pandas as pd
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# .env 로드 및 OpenAI 클라이언트 설정
load_dotenv()
api_key = os.getenv("AI_API_KEY")
client = OpenAI(api_key=api_key)

# OpenAI 임베딩 설정 (rag_module과 동일하게)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536 

def get_stock_info():
    base_url = "http://kind.krx.co.kr/corpgeneral/corpList.do"    
    method = "download"
    url = f"{base_url}?method={method}"
    df = pd.read_html(url, header=0, encoding='euc-kr')[0]
    df['종목코드'] = df['종목코드'].apply(lambda x: f"{x:06}")     
    return df

print("KRX에서 데이터를 가져오는 중...")
df = get_stock_info()

# 1. 텍스트 병합 (홈페이지 제외)
def create_combined_text(row):
    texts = [f"{col}: {row[col]}" for col in df.columns if col != '홈페이지' and pd.notna(row[col])]
    return " | ".join(texts)

df['combined_text'] = df.apply(create_combined_text, axis=1)

# 2. OpenAI를 이용한 벡터 변환
def get_embeddings_bulk(text_list):
    print(f"총 {len(text_list)}개 데이터 임베딩 생성 중...")
    # 비용 절약 및 속도를 위해 한 번에 요청 (Batch)
    response = client.embeddings.create(input=text_list, model=EMBEDDING_MODEL)
    return [data.embedding for data in response.data]

# 데이터가 많을 경우를 대비해 100개씩 끊어서 임베딩 (API 제한 방지)
all_embeddings = []
batch_size = 100
for i in range(0, len(df), batch_size):
    batch_text = df['combined_text'].iloc[i:i+batch_size].tolist()
    all_embeddings.extend(get_embeddings_bulk(batch_text))

df['embedding'] = all_embeddings

# 3. 엘라스틱서치 설정
es = Elasticsearch("http://localhost:9200", http_compress=True)
index_name = 'stock_info'

index_settings = {
    "settings": {
        "index.max_ngram_diff": 3,  # 🌟 이 설정을 추가하여 차이값 제한을 해제합니다.
        "analysis": {
            "tokenizer": {
                "ngram_tokenizer": {
                    "type": "ngram", 
                    "min_gram": 2, 
                    "max_gram": 5, 
                    "token_chars": ["letter", "digit"]
                }
            },
            "analyzer": {
                "ngram_analyzer": {
                    "type": "custom", 
                    "tokenizer": "ngram_tokenizer"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "회사명": {
                "type": "text", 
                "analyzer": "ngram_analyzer",
                "fields": {"keyword": {"type": "keyword"}} 
            },
            "업종": {"type": "text", "analyzer": "ngram_analyzer"},
            "주요제품": {"type": "text", "analyzer": "ngram_analyzer"},
            "종목코드": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": 1536, 
                "index": True,
                "similarity": "cosine" 
            }
        }
    }
}

# 기존 인덱스 삭제 후 재생성
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
es.indices.create(index=index_name, body=index_settings)

# 4. Bulk 적재
json_records = json.loads(df.to_json(orient='records'))
action_list = [
    {
        '_op_type': 'index',
        '_index': index_name,
        '_source': row
    } for row in json_records
]

helpers.bulk(es, action_list)
print("✅ OpenAI 임베딩 기반으로 데이터 적재가 완료되었습니다!")