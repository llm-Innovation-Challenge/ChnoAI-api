from typing import TypedDict, Optional

class Message(TypedDict):
    '''
        개별 메시지를 나타내는 딕셔너리.
        1. id -> 고유 메시지 ID: int
        2. conversation_id -> 대화 ID: int
        3. sequence_number -> 메시지 순서 번호: int
        4. message_type -> 메시지 유형 (예: 질문, 답변): str
        5. message_content -> 메시지 내용: str
    '''
    id: int
    conversation_id: int
    sequence_number: int
    message_type: str
    message_content: str

class CodeStorage(TypedDict):
    '''
        대화에 포함된 코드 정보를 나타내는 딕셔너리.
        1. code_description -> 코드 설명: str
        2. code_index -> 코드의 고유 인덱스: str
        3. code_snippet -> 코드 스니펫: str
    '''
    code_description: str
    code_index: str
    code_snippet: str

class QA(TypedDict):
    '''
        개별 Q&A 세트를 나타내는 딕셔너리.
        1. q -> 질문 내용: str
        2. a -> 답변 내용: str
    '''
    q: str  
    a: str

class QAProcessorGraphState(TypedDict):
    '''
        qna_processor 모듈을 실행하는 데 필요한 상태 정보를 나타냄.
        1. processing_data -> 현재 처리 중인 Q&A 세트: Optional[QA]
        2. not_processed_conversations -> 아직 처리되지 않은 Q&A 세트 목록: list[QA]
        3. processed_conversations -> 처리된 Q&A 세트 목록: list[QA]
        4. code_documents -> 처리된 코드 문서 정보 목록: list[CodeStorage]
    '''
    processing_data: Optional[QA]         
    not_processed_conversations: list[QA]
    processed_conversations: list[QA]
    code_documents: list[CodeStorage]

class WriterGraphState(TypedDict): 
    '''
        Writing 모듈을 실행하는 데 필요한 입력 정보.
        1. preprocessed_conversations -> qna_processor 모듈에 의해 전처리된 Q&A 세트: list[QA]
        2. code_document -> qna_processor 모듈에 의해 생성된 코드 스니펫 딕셔너리: dict{'Code_Snippet_1': 'code'}
        3. message_to_index_dict -> subtitle_generator 모듈에서 생성된 각 Q&A 세트에 해당하는 인덱스: dict['0': [1-1, 1-2, 1-3]] ('0'은 첫 번째 Q&A 세트를 나타냄)
        4. final_documents -> 작성 중인 문서들: dict['1-1': '## 1-1) 제목']
    '''
    preprocessed_conversations: list[QA]
    code_document: dict
    message_to_index_dict: dict
    final_documents: dict
