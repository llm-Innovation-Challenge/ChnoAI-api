from playwright.sync_api import sync_playwright
import sys

def run_headless_browser(url):
    """
    주어진 URL에서 헤드리스 브라우저를 실행하여 사용자 메시지와 
    어시스턴트 메시지를 추출합니다.

    Args:
        url (str): ChatGPT 공유 URL. 유효한 URL은 "https://chatgpt.com/share/"로 시작해야 합니다.

    Returns:
        tuple: (chat_url, chat_room_title, data)
            - chat_url (str): 접근한 채팅 URL
            - chat_room_title (str): 채팅방 제목
            - data (list of dict): 사용자 질문과 어시스턴트 답변의 리스트, 
              각 딕셔너리는 "question"과 "answer" 키를 가집니다.
              예: [{"question": "질문1", "answer": "답변1"}, ...]
    """
    if not url.startswith("https://chatgpt.com/share/"):
        raise ValueError("Invalid URL")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # 헤드리스 모드에서 브라우저 실행
        page = browser.new_page()

        # 불필요한 리소스 차단 (이미지, 스타일시트, 폰트 등)
        page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "stylesheet", "font"] else route.continue_())

        # 주어진 URL로 이동하고 네트워크가 안정될 때까지 대기
        page.goto(url, wait_until="networkidle")

        # 현재 페이지 URL 및 채팅방 제목 추출
        chat_url = page.url
        chat_room_title = page.query_selector("h1").inner_text()

        # 사용자 메시지 추출
        user_messages = page.query_selector_all('[data-message-author-role="user"]')
        print(f"Found {len(user_messages)} user messages")
        sys.stdout.flush()
        user_texts = [msg.inner_text() for msg in user_messages]
        print(f"User messages: {user_texts}")
        sys.stdout.flush()

        # 어시스턴트 메시지 추출
        assistant_messages = page.query_selector_all('[data-message-author-role="assistant"]')
        print(f"Found {len(assistant_messages)} assistant messages")
        sys.stdout.flush()
        assistant_texts = [msg.inner_text() for msg in assistant_messages]
        print(f"Assistant messages: {assistant_texts}")
        sys.stdout.flush()

        # 사용자 질문과 어시스턴트 답변의 데이터 구조 생성
        data = [{"question": question, "answer": assistant_texts[index] if index < len(assistant_texts) else ""} for index, question in enumerate(user_texts)]

        browser.close()  # 브라우저 닫기
        return chat_url, chat_room_title, data  # 결과 반환
