import os
import re

from dotenv import load_dotenv
from typing import Annotated, Tuple
from tqdm import tqdm

# langchin
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage

# langfuse
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from app.type import CodeStorage, QA, QAProcessorGraphState as GraphState
from app.constants import CONVERSATION_ID
from app.utils import fetch_messages, format_message
from app.qna_processor.evaluate_score import evaluate_processed_answer, evaluate_coherence  

langfuse_handler = CallbackHandler()
langfuse = Langfuse()

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class QnAProcessor:
    """
    QnAProcessor 클래스는 Q&A 쌍을 처리하는 데 사용됩니다.
    이 클래스는 질문과 답변에서 코드 스니펫을 추출하고, 해당 코드를 Solar-LLM을 사용하여 설명으로 대체합니다.

    Attributes:
    - pair_list (list[QA]): 처리할 Q&A 쌍의 리스트.
    - model: LLM 모델 인스턴스.
    - code_documents (list[CodeStorage]): 각 Q&A 쌍에 대한 코드 문서 정보를 저장하는 리스트.

    Methods:
    - process_qna_pair(): Q&A 쌍을 처리하고 코드 스니펫을 설명으로 대체하며, 최종 Q&A 쌍과 코드 문서 리스트를 반환합니다.
    - extract_code_and_replace_with_description(qna_pair): 질문과 답변에서 코드를 추출하고 설명으로 대체합니다.
    - backtick_process_with_llm(answer): 답변 내 코드에 백틱(```)을 추가하여 명확하게 표시합니다.
    - describe_code_with_llm(code_snippet): Solar-LLM을 사용하여 주어진 코드 스니펫에 대한 설명을 생성합니다.
    - summarize_question_with_llm(question): Solar-LLM을 사용하여 주어진 질문을 요약합니다.
    """

    def __init__(self, qna_list: list[QA], model) -> None:
        self.qna_list = qna_list
        self.model = model
        self.code_documents: list[CodeStorage] = []
    
    def process_qna_pair(self, graph_state: GraphState, MAX_ITERATION: int = 3) -> Tuple[list[QA], list[CodeStorage]]:
        """
        Q&A 쌍을 처리하고 코드 스니펫을 설명으로 대체하며, 최종 Q&A 쌍과 코드 문서 리스트를 반환합니다.

        Args:
            graph_state (GraphState): 현재 그래프 상태를 저장하는 데이터 구조.
            MAX_ITERATION (int): 질문 요약 및 답변 처리 반복 횟수.

        Returns:
            Tuple[list[QA], list[CodeStorage]]: 처리된 Q&A 리스트와 코드 저장소 리스트.
        """
        for qna_pair in tqdm(self.qna_list, desc="Processing Q&A Pairs", unit="pair"):
            graph_state["processing_data"] = qna_pair

            question = qna_pair["q"]
            answer = qna_pair["a"]

            coherent_score = 0
            best_summarized_question = "" 
            for i in range(MAX_ITERATION):
                # 질문을 요약합니다.
                summarized_question = self.summarize_question_with_llm(question)
                coherence_result = evaluate_coherence(question, summarized_question)
                
                current_score = coherence_result.get("coherence_score")
                coherent_reason = coherence_result.get("reason")

                print(f"Iteration {i + 1}/{MAX_ITERATION}")
                print(f"Coherent score: {current_score}")
                print(f"Coherent reason: {coherent_reason}")

                # 현재 점수가 이전 점수보다 높으면 최상의 요약 질문 업데이트
                if current_score > coherent_score:
                    coherent_score = current_score
                    best_summarized_question = summarized_question 
                
                # 기준 점수를 초과하면 반복 종료
                if coherent_score >= 0.8:
                    print("Coherent score 기준점을 넘음. 반복 종료.")
                    break
                else:
                    print("Coherent score 기준점을 넘지 못하여 다시 summarize 실행 중...")

            # coherent_score가 기준 점수를 넘으면 요약 질문을 반영
            graph_state["processing_data"]["q"] = best_summarized_question

            recall_score = 0
            best_processed_answer = ""
            for i in range(MAX_ITERATION):
                # 답변을 백틱 처리합니다.
                processed_answer = self.backtick_process_with_llm(answer)
                evaluation_results = evaluate_processed_answer(answer, processed_answer)

                current_recall_score = evaluation_results.get("recall")
                print(f"Iteration {i + 1}/{MAX_ITERATION}")
                print(f"Recall score: {current_recall_score}")

                # 현재 점수가 이전 점수보다 높으면 최상의 처리된 답변 업데이트
                if current_recall_score > recall_score:
                    recall_score = current_recall_score
                    best_processed_answer = processed_answer  
                
                # 기준 점수를 초과하면 반복 종료
                if recall_score >= 0.90:
                    print("Recall score 기준점을 넘음. 반복 종료.")
                    break
                else:
                    print("Recall score 기준점을 넘지 못하여 다시 backtick 처리 실행 중...")

            # 최상의 처리된 답변을 반영
            graph_state["processing_data"]["a"] = best_processed_answer
            
            # 코드 스니펫을 추출하고 설명으로 대체
            question_without_code, answer_without_code = self.extract_code_and_replace_with_description(summarized_question, processed_answer)            
            graph_state["code_documents"] = self.code_documents
            qna_pair["q"] = question_without_code
            qna_pair["a"] = answer_without_code

            # 처리된 Q&A 쌍을 그래프 상태에 추가
            graph_state["processing_data"] = qna_pair
            graph_state["processed_conversations"].append(qna_pair)

        return self.qna_list, self.code_documents


    def extract_code_and_replace_with_description(self, question: str, answer: str, description_prefix="Code_Snippet") -> Tuple[str, str]:
        """질문과 답변에서 코드 스니펫을 추출하고 설명으로 대체합니다.

        Args:
            question (str): 코드 스니펫이 포함된 질문.
            answer (str): 코드 스니펫이 포함된 답변.
            description_prefix (str): 생성된 코드 설명의 접두사.

        Returns:
            Tuple[str, str]: 코드가 설명으로 대체된 질문과 답변.
        """

        code_pattern = r"```(.*?)```"
        code_index_counter = len(self.code_documents)

        def _replace_code_with_placeholder(match):
            nonlocal code_index_counter
            code_snippet = match.group(1).strip()
            # 코드 설명 생성
            code_description = self.describe_code_with_llm(code_snippet=code_snippet)
            code_index_counter += 1
            code_index = f"{description_prefix}_{code_index_counter}"

            placeholder = f"{code_index}: {code_description}"

            # 코드 저장
            code_storage = CodeStorage(
                code_snippet=code_snippet,
                code_index=code_index,
                code_description=code_description,
            )
            self.code_documents.append(code_storage)

            return f"<-- {placeholder} -->"

        # 질문과 답변에서 코드 스니펫을 찾아 설명으로 대체
        question_without_code = re.sub(code_pattern, _replace_code_with_placeholder, question, flags=re.DOTALL)
        answer_without_code = re.sub(code_pattern, _replace_code_with_placeholder, answer, flags=re.DOTALL)
        
        return question_without_code, answer_without_code
    
    def backtick_process_with_llm(self, answer:str):
        """
            Solar-LLM을 사용하여 코드에 백틱을 추가합니다.
            Solar-LLM : 코드 스니펫에 백틱을 추가하여 포맷팅을 향상시킵니다.
            - 사용 모델명 : solar-pro
            
            Args:
                answer (str): 백틱을 추가할 코드 스니펫 또는 텍스트.

            Returns:
                str: 백틱이 추가된 코드 스니펫.
        """
        backtick_processor: Annotated[str, HumanMessage] = langfuse.get_prompt("backtick_processor")
        prompt = backtick_processor.compile(answer=answer)
        response = self.model.invoke(prompt)
        return response.content.strip()


    def describe_code_with_llm(self, code_snippet: str) -> str:
        """Solar-LLM을 사용하여 코드 설명을 생성합니다.
        
        Solar-LLM : 주어진 코드 스니펫에 대한 설명을 생성.
        - 사용 모델명 : solar-pro
        
        Args:
            code_snippet (str): 설명할 코드 스니펫.

        Returns:
            str: 코드 스니펫에 대한 설명.
        """
        short_code_description: Annotated[str, HumanMessage] = langfuse.get_prompt("short_code_description")
        prompt = short_code_description.compile(code_snippet=code_snippet)
        response = self.model.invoke(prompt)
        return response.content.strip()
    
    
    def summarize_question_with_llm(self, question: str):
        """Solar-LLM을 사용하여 질문 요약.
        
        Solar-LLM : 주어진 질문을 요약하여 간결한 형태로 제공.
        - 사용 모델명 : solar-pro

        Args:
            question (str): 요약할 질문.

        Returns:
            str: 요약된 질문.
        """
        question_summarizer: Annotated[str, HumanMessage] = langfuse.get_prompt("question_summarizer")
        prompt = question_summarizer.compile(question=question)
        response = self.model.invoke(prompt)
        return response.content.strip()


def run_processor_qna(conversation_data: list[QA], model_name="solar-pro") :
   
    model = ChatUpstage(model=model_name)

    qna_processor = QnAProcessor(conversation_data, model)
    init_graph_state = GraphState(
        not_processed_conversations=conversation_data,
        processing_data=None,
        processed_conversations=[],
        code_documents=[]
    )

    processed_qna_list, code_documents = qna_processor.process_qna_pair(graph_state=init_graph_state)
    return processed_qna_list, code_documents


if __name__ == "__main__":

    model_name = "solar-pro"
    conversation_id = CONVERSATION_ID["EXAMPLE_1"]
    conversation = fetch_messages(conversation_id)
    conversation_data = format_message(conversation)
    processed_qna_list, code_documents = run_processor_qna(conversation_data, model_name)
    
    print(f"Processed QnA:\n {processed_qna_list} \n")
    print(f"Code document:\n {code_documents} \n")
