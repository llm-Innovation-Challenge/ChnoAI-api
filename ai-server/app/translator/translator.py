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
        text (str): 검사할 문자열.

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
    similarity = np.dot(embedded_query, embedded_documents.T)  # 내적을 통한 유사도 계산
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
    API_URL = "https://api.upstage.ai/v1/solar/chat/completions"  # API URL
    HEADERS = {"Authorization": f"Bearer {os.getenv('UPSTAGE_API_KEY')}"}  # API 키 설정

    data = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "stream": False
    }

    response = requests.post(API_URL, headers=HEADERS, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]  # 성공 시 번역 결과 반환
    else:
        print(f"Translation API error: {response.status_code}, {response.text}")
        return text  # 오류 발생 시 원본 텍스트 반환

def translate_q_and_a(conversation_list: list[QA]):
    """
    주어진 Q&A 목록을 번역하는 함수.
    
    Solar-LLM :
    1. 임베딩 모델을 사용하여 원본 텍스트와 번역된 텍스트 간의 유사도 계산합니다.
    2. 번역 모델을 사용하여 한글 텍스트를 영어로 번역합니다.
    - 사용 모델명 : solar-embedding-1-large-passage, solar-1-mini-translate-koen

    Args:
        conversation_list (list[QA]): 번역할 Q&A 목록.
        translation_model (str): 사용할 번역 모델.

    Returns:
        list[QA]: 번역된 Q&A 목록.
    """
    # 번역 모델명
    translation_model = "solar-1-mini-translate-koen"

    # 임베딩 모델 초기화
    passage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")  

    translated_conversations = []  # 번역된 대화를 저장할 리스트
    
    for conversation in conversation_list:
        question = conversation['q']  # 질문 추출
        answer = conversation['a']  # 답변 추출

        # 질문이 한국어를 포함하는 경우 번역 시도
        if contains_korean(question):
            attempts = 0  # 시도 횟수 초기화
            similarity_q = 0.0  # 유사도 초기화
            translated_q = question  # 초기 번역 결과 설정

            # 유사도가 기준 이하이거나 여전히 한국어가 포함된 경우 최대 5회 시도
            while (similarity_q < 0.5 or contains_korean(translated_q)) and attempts < 5:
                translated_q = translate_text_with_api(question, translation_model)  # 질문 번역

                # 원본 질문과 번역된 질문 간의 유사도 계산
                embedded_query = passage_embeddings.embed_documents([question])
                embedded_translated_q = passage_embeddings.embed_documents([translated_q])
                similarity_q = calculate_similarity(embedded_query, embedded_translated_q)

                attempts += 1  # 시도 횟수 증가
            if attempts == 5:
                print(f"최대 재시도 횟수 도달. 질문 최종 번역: {translated_q}")
        else:
            translated_q = question  # 질문이 영문일 시, 번역하지 않음.
        
        # 답변이 한국어를 포함하는 경우 번역 시도
        if contains_korean(answer):
            attempts = 0  # 시도 횟수 초기화
            similarity_a = 0.0  # 유사도 초기화
            translated_a = answer  # 초기 번역 결과 설정

            # 유사도가 기준 이하이거나 여전히 한국어가 포함된 경우 최대 5회 시도
            while (similarity_a < 0.5 or contains_korean(translated_a)) and attempts < 5:
                translated_a = translate_text_with_api(answer, translation_model)  # 답변 번역

                # 원본 답변과 번역된 답변 간의 유사도 계산
                embedded_answer = passage_embeddings.embed_documents([answer])
                embedded_translated_a = passage_embeddings.embed_documents([translated_a])
                similarity_a = calculate_similarity(embedded_answer, embedded_translated_a)

                attempts += 1  # 시도 횟수 증가
            if attempts == 5:
                print(f"최대 재시도 횟수 도달. 답변 최종 번역: {translated_a}")
        else:
            translated_a = answer  # 답변이 영문일 시, 번역하지 않음
        
        # 번역된 Q&A 추가
        translated_conversations.append(QA(q=translated_q, a=translated_a))

    return translated_conversations

if __name__ == "__main__":
    database = get_db_client()  # 데이터베이스 클라이언트 초기화

    try:
        conversation = fetch_messages(database, CONVERSATION_ID["EXAMPLE_1"])
        # 대화 데이터를 가져와서 포맷팅
        conversation_data = format_message(conversation)

    except Exception as e:
        print(f"Error fetching conversation data: {e}")
        conversation_data = []  # 오류 발생 시 빈 리스트 초기화

    # Q&A 번역 수행
    translated_result = translate_q_and_a(conversation_data)

    # 번역 결과 출력
    for idx, conversation in enumerate(translated_result):
        print(f"번역된 질문 {idx+1}: {conversation['q']}")
        print(f"번역된 답변 {idx+1}: {conversation['a']}")
