import os
import requests
import numpy as np

from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings

from app.type import QA
from app.utils import fetch_messages, format_message
from app.constants import CONVERSATION_ID
from app.db_client import get_db_client

load_dotenv()

def contains_korean(text: str) -> bool:
    """
    주어진 텍스트에 한국어가 포함되어 있는지 확인하는 함수.

    Args:
        text (str)

    Returns:
        bool: 한국어가 포함되어 있으면 True, 그렇지 않으면 False.
    """
    return any('가' <= char <= '힣' for char in text)

def calculate_similarity(embedded_query, embedded_documents):
    """
    쿼리 임베딩과 문서 임베딩 간의 유사성을 계산하는 함수.

    Args:
        embedded_query (np.array): 쿼리 임베딩.
        embedded_documents (np.array): 문서 임베딩.

    Returns:
        np.array: 쿼리와 문서 간의 유사도 점수.
    """
    embedded_query = np.array(embedded_query)
    embedded_documents = np.array(embedded_documents)
    similarity = np.dot(embedded_query, embedded_documents.T)
    return np.squeeze(similarity)

def translate_text_with_api(text: str, model: str) -> str:
    """
    Solar-API를 사용하여 주어진 텍스트를 영어로 번역하는 함수.

    Args:
        text (str): 번역할 텍스트.
        model (str): 사용할 번역 모델.

    Returns:
        str: 번역된 텍스트.
    """
    API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
    HEADERS = {"Authorization": f"Bearer {os.getenv('UPSTAGE_API_KEY')}"}

    data = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "stream": False
    }

    response = requests.post(API_URL, headers=HEADERS, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"] # 성공 시 번역 결과 반환
    else:
        print(f"Translation API error: {response.status_code}, {response.text}")
        return text  # 오류 발생 시 원본 텍스트 반환


def translate_q_and_a(conversation_list: list[QA], translation_model: str = "solar-1-mini-translate-koen") -> list[QA]:
    """
    주어진 Q&A 목록을 번역하는 함수.
    
    Solar-LLM :
    1. 임베딩 모델을 사용하여 원본 텍스트와 번역된 텍스트 간의 유사도 계산합니다.
    2. 번역 모델을 사용하여 한글 텍스트를 영어로 번역합니다.
    - 사용 모델명 : solar-embedding-1-large-passage, solar-1-mini-translate-koen

    Args:
        conversation_list (list[QA]): 번역할 Q&A 목록.
        translation_model (str): 사용할 번역 모델. 기본값 = "solar-1-mini-translate-koen"

    Returns:
        list[QA]: 번역된 Q&A 목록.
    """
    translated_conversations = []

    # 임베딩 모델 초기화
    passage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")

    def translate_if_needed(text: str) -> str:
        """
        텍스트에 한국어가 포함되어 있다면 번역하고, 유사도를 계산한 후 결과를 반환하는 함수.
        한국어가 없다면 원본을 반환.
        """
        if not contains_korean(text):
            return text

        attempts = 0
        similarity = 0.0
        translated_text = text

        while (similarity < 0.5 or contains_korean(translated_text)) and attempts < 5:
            translated_text = translate_text_with_api(text, translation_model)
            embedded_original = passage_embeddings.embed_documents([text])
            embedded_translated = passage_embeddings.embed_documents([translated_text])
            similarity = calculate_similarity(embedded_original, embedded_translated)
            attempts += 1

        if attempts == 5:
            print(f"최대 재시도 횟수 도달. 최종 번역: {translated_text}")

        return translated_text

    # 대화 리스트에서 질문과 답변 각각 번역 처리
    for conversation in conversation_list:
        translated_q = translate_if_needed(conversation['q'])  
        translated_a = translate_if_needed(conversation['a'])  

        # 번역된 질문과 답변을 Q&A 형식으로 리스트에 추가
        translated_conversations.append(QA(q=translated_q, a=translated_a))

    return translated_conversations


if __name__ == "__main__":
    database = get_db_client() 
    
    try:
        conversation = fetch_messages(database, CONVERSATION_ID["EXAMPLE_1_KOR"])
        conversation_data = format_message(conversation)
        
    except Exception as e:
        print(f"Error fetching conversation data: {e}")
        conversation_data = []   # 오류 발생 시 빈 리스트 초기화

    # Q&A 번역 수행
    translated_result = translate_q_and_a(conversation_data)

    for idx, conversation in enumerate(translated_result):
        print(f"번역된 질문 {idx+1}: {conversation['q']}")
        print(f"번역된 답변 {idx+1}: {conversation['a']}")
