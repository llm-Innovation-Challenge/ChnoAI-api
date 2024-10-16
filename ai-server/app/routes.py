import requests
import logging
import json
import re

from flask import request, jsonify, render_template, Blueprint
from supabase import  Client
from langfuse.callback import CallbackHandler

from app.db_client import get_db_client
from app.utils import fetch_messages, format_message, format_input, format_extracted_code
from app.type import WriterGraphState
from app.subtitle_generator.subtitle_generator import SubtitleGenerator
from app.title_generator.title_generator import TitleGenerator
from app.qna_processor.qna_processor import run_processor_qna
from app.writer.writer import compiled_graph
from app.process_url import run_headless_browser
from app.translator.translator import translate_q_and_a



bp = Blueprint('main', __name__)
database: Client = get_db_client()
langfuse_handler = CallbackHandler()


@bp.route('/')
def index():
    return render_template('index.html')

@bp.route("/process-url", methods=["GET"])
def process_url():
    '''
    ChatGPT 대화 URL을 받아 크롤링하는 API입니다.

    요청 예시:
        GET /process-url?url=<ChatGPT 대화 URL>

    Returns:
        JSON 응답:
            - chatUrl: 크롤링된 대화의 URL
            - chatRoomTitle: 대화 방 제목
            - data: 크롤링된 대화 내용
    '''
    # 쿼리 파라미터에서 URL을 가져옵니다.
    url = request.args.get('url')
    print("Processing URL:", url)
    
    # URL이 제공되지 않은 경우 400 오류 응답을 반환합니다.
    if not url:
        return "URL is required", 400
    
    try:
        # 주어진 URL을 크롤링하고 결과를 가져옵니다.
        chat_url, chat_room_title, data = run_headless_browser(url)
        # 크롤링된 데이터와 함께 JSON 응답을 반환합니다.
        return jsonify({"chatUrl": chat_url, "chatRoomTitle": chat_room_title, "data": data}), 200
    except Exception as e:
        # 크롤링 중 오류가 발생한 경우 500 오류 응답을 반환합니다.
        return str(e), 500

@bp.route('/generate-blog', methods=['POST'])
def generate_blog():
    '''
    Solar-LLM 및 Solar-API를 활용하여 GPT와의 대화를 정리한 블로그를 생성하는 API입니다.

    모듈:
    1. translator 모듈을 사용하여 한글로 된 대화를 영어로 번역합니다.
       - 사용 모델명: solar-embedding-1-large-passage, solar-1-mini-translate-koen
    2. subtitle_generator 모듈을 사용하여 노션 페이지의 목차를 생성합니다.
       - 사용 모델명: solar-pro, solar-embedding-1-large
    3. qna_processor 모듈을 사용하여 질문을 요약하고, GPT의 답변에서 코드를 추출하며, 
       해당 코드에 대한 설명을 생성합니다.
       - 사용 모델명: solar-pro
    4. title_generator 모듈을 사용하여 노션 페이지의 제목을 생성합니다.
       - 사용 모델명: solar-pro
    5. writer 모듈을 사용하여 최종 블로그 콘텐츠를 작성합니다.
       - 사용 모델명 : solar-pro

    요청 예시:
    POST /generate-blog
        Body: {
            "conversation_id": "<대화 ID>"
        }
    Returns:
        생성된 블로그 내용 또는 오류 메시지를 포함한 JSON 응답.
    '''
    # 0. 대화 메세지 로드
    data = request.json
    conversation_id = data.get('conversation_id')
    messages = fetch_messages(database, conversation_id)
    conversation_data = format_message(messages)

    # 1. 한글 대화 텍스트 영어로 번역 (translator 모듈)
    translated_conversation_data = translate_q_and_a(conversation_data)
    
    # 2. 목차 생성 (subtitle_generator 모듈)
    subtitle_generator = SubtitleGenerator(config_path = "app/configs/subtitle_generator.yaml")
    subtitle_docs = subtitle_generator(translated_conversation_data)

    # 3. 질문 압축 및 코드 추출 (qna_processor 모듈)
    processed_qna_list, code_documents = run_processor_qna(translated_conversation_data)

    # 3-1.블로그 작성 모듈 이전에 목차 생성 모듈에서 나온 결과 전처리
    ## 목차 딕셔너리의 value 리스트 내에 있는 값들을 모두 문자열로 처리
    for key, value in subtitle_docs[1].items():
        subtitle_docs[1][key] = [str(v) for v in value]

    # 3-2. 블로그 작성 모듈 이전에 질문 압축 및 코드 추출 모듈에서 나온 결과 전처리
    processed_code_documents = format_extracted_code(code_documents)

    # 3. 블로그 제목 생성 (title_generator 모듈)
    title_generator = TitleGenerator(config_path="app/configs/title_generator.yaml")
    title = title_generator(subtitle_docs[1])

    # 4. 블로그 작성 (writer 모듈)
    ## 들어갈 graph_state를 정의
    graph_state = WriterGraphState(
        preprocessed_conversations=processed_qna_list,
        code_document=processed_code_documents,
        message_to_index_dict=subtitle_docs[1],
        final_documents=subtitle_docs[0]
    )

    ## graph_state를 이용하여 블로그 작성
    final_state = compiled_graph.invoke(
        graph_state, 
        config={
            "configurable": {"thread_id": 42}, 
            "callbacks": [langfuse_handler]}
    )

    # Writer module 결과물 로깅
    # json.dumps를 사용하여 객체를 문자열로 변환, string만 출력이 가능합니다.
    logging.info("final_state: %s", json.dumps(final_state["final_documents"], default=str))

    # 작성된 내용에서 ## 이후의 숫자를 1씩 더합니다.
    final_state["final_documents"] = {k: re.sub(r'(\d+)', lambda x: str(int(x.group()) + 1), v) for k, v in final_state["final_documents"].items()}

    # 입력된 딕셔너리의 값들을 줄바꿈으로 연결하여 하나의 문자열로 만듭니다.
    final_technote = format_input(final_state["final_documents"])

    # 5. 노션 페이지 생성 및 게시
    notion_title = title
    notion_content = final_technote
    question_type = []
    framework_tags = []
    language_tags = []
    os_tags = []
    tech_stack_tags = []

    notion_response = requests.post('http://localhost:4000/publish-to-notion', json={
        "title": notion_title, 
        "content": notion_content, 
        "question_type": question_type,
        "os_tags": os_tags,
        "framework_tags": framework_tags,
        "language_tags": language_tags,
        "tech_stack_tags": tech_stack_tags
    })
    
    if notion_response.status_code == 200:
        notion_page_id = notion_response.json().get('page_id')
        notion_page_url = notion_response.json().get('url')
        notion_page_public_url = notion_response.json().get('public_url')
        return jsonify({"message": "Blog generated and published to Notion successfully", "notion_page_id": notion_page_id, "notion_page_url": notion_page_url, "notion_page_public_url": notion_page_public_url}), 200
    else:
        return jsonify({"error": "Failed to publish to Notion", "details": notion_response.json()}), 500


    
