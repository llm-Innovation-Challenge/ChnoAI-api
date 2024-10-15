import logging
import warnings
from functools import wraps
from app.type import Message, QA
from supabase import Client
from datetime import datetime

# langfuse
from langfuse import Langfuse
langfuse = Langfuse()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def deprecated(func):
    """메서드 호출을 기록하고 사용 중단 경고를 하는 데코레이터입니다."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 메서드 호출 시 로그 기록
        logger.info(f"호출된 메서드: {func.__name__}")
        
        # 사용 중단 경고 발행
        warnings.warn(f"{func.__name__}는 사용 중단되었으며 향후 버전에서 제거될 예정입니다.",
                      category=DeprecationWarning, stacklevel=2)
        
        # 원래 함수 호출
        return func(*args, **kwargs)
    
    return wrapper

def format_message(conversation: list[Message]) -> list[QA]:
    """
    주어진 대화를 포맷하여 각 'q' (질문)와 'a' (답변) 쌍을 포함하는 
    딕셔너리 목록으로 변환합니다.

    Args:
        conversation (List[Dict[str, str]]): 메시지 딕셔너리 목록으로 구성된 대화입니다.
    
    Returns:
        List[Dict[str, str]]: 포맷된 대화 쌍의 목록으로, 
                              [{'q': str, 'a': str}, ...] 형태입니다.
    """
    
    formatted_conversation = []
    
    # 대화를 두 메시지씩 반복(질문/답변 쌍)
    for i in range(0, len(conversation), 2):
        try:
            q_message = conversation[i]["message_content"]
            a_message = conversation[i + 1]["message_content"]
            formatted_conversation.append({"q": q_message, "a": a_message})
        except IndexError:
            # 완전한 q/a 쌍이 없으면 건너뜁니다.
            break
    
    return formatted_conversation


def fetch_messages(database:Client, conversation_id:int) -> list[Message]:
    '''
    주어진 대화 ID에 대한 메시지를 데이터베이스에서 가져옵니다.
    
    이 함수는 특정 대화와 관련된 모든 메시지를 가져오며, 
    메시지는 순서 번호에 따라 오름차순으로 정렬됩니다. 
    지정된 대화 ID에 대한 메시지가 없으면 예외가 발생합니다.

    Args:
        database (Client): "messages" 테이블을 쿼리하는 데 사용되는 데이터베이스 클라이언트 인스턴스입니다.
        conversation_id (int): 메시지를 가져올 대화의 ID입니다.
    
    Returns:
        list[Message]: 각 메시지를 "sequence_number", "message_type", "message_content"를 포함하는 
        딕셔너리로 표현한 메시지 목록입니다.
    
    Raises:
        Exception: 지정된 대화 ID에 대한 메시지가 없을 경우 발생합니다.
    '''
    try:
        response = database.table("messages") \
                    .select("sequence_number, message_type, message_content") \
                    .eq("conversation_id", conversation_id) \
                    .order("sequence_number", desc=False) \
                    .execute()
        
        messages = response.data
        if len(messages) == 0 :
            raise Exception(f"대화 {conversation_id}에 관련된 메시지가 없습니다.")

    except Exception as e:
        print(f"내부 서버 오류: {str(e)}") 
    
    return messages

def format_extracted_code(items):
    # 데이터베이스에서 가져온 코드 스니펫의 형태를 변환합니다.
    result_dict = {item['code_index']: item['code_snippet'] for item in items}
    return result_dict

def format_input(input_dict):
    # 입력된 딕셔너리의 값들을 줄바꿈으로 연결하여 하나의 문자열로 만듭니다.
    return '\n'.join(input_dict.values())
    
def get_current_datetime():
    # 현재 날짜와 시간을 가져옵니다
    now = datetime.now()
    # 연-월-일 시:분 형식으로 변환합니다
    formatted_datetime = now.strftime("%Y년 %m월 %d일 %H시 %M분")
    return formatted_datetime

def load_conversation(chat_name="chat_ex2"):
    # Langfuse에 저장된 샘플 대화 가져오기 
    # (chat_ex1 -> 짧은 대화, chat_ex2 -> 긴 대화)
    try:
        conversation_data = langfuse.get_prompt(chat_name).prompt
    except Exception as e:
        print(f"대화 데이터를 가져오는 중 오류 발생: {e}")
        conversation_data = []

    return conversation_data