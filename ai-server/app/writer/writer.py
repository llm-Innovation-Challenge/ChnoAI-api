import re
from tqdm import tqdm
from dotenv import load_dotenv

# langchin
from langchain_upstage import ChatUpstage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

# langfuse
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from typing import Tuple
from app.type import QA, WriterGraphState as GraphState

load_dotenv()

langfuse_handler = CallbackHandler()
langfuse = Langfuse()

##########################평가 관련 함수 정의###############################

def find_code_snippets(text:str) -> list[str]:
    '''
    주어진 텍스트에서 코드 스니펫을 찾아 리스트로 반환하는 함수입니다.

    Args:
        text (str): 코드 스니펫을 검색할 문자열입니다.

    Returns:
        list[str]: 텍스트에서 발견된 코드 스니펫의 리스트입니다.
    '''
    pattern = r"<-- Code_Snippet_\d+: .*? -->"
    return re.findall(pattern, text)

def calculate_precision_recall(true_positive:int, generated_total:int, gt_total:int)-> Tuple[float, float]:
    '''
    Precision 및 Recall을 계산하는 함수입니다.

    Args:
        true_positive (int): 참 긍정 수.
        generated_total (int): 생성된 항목의 총 수.
        gt_total (int): 실제 항목의 총 수.

    Returns:
        Tuple[float, float]: 계산된 precision과 recall의 튜플입니다.
    '''
    # Precision = TP / (TP + FP)
    precision = true_positive / generated_total if generated_total else 0
    
    # Recall = TP / (TP + FN)
    recall = true_positive / gt_total if gt_total else 0
    
    return precision, recall

def overall_precision_recall(generated_doc:dict[str,str], gt_doc:dict[str,str]) -> Tuple[float, float]:
    '''
    전체 문서에 대해 목차별로 True Positive를 계산 후 전체 Precision과 Recall을 계산하는 함수입니다.

    Args:
        generated_doc (dict[str, str]): 생성된 문서의 각 섹션을 포함하는 딕셔너리.
        gt_doc (dict[str, str]): 실제 문서의 각 섹션을 포함하는 딕셔너리.

    Returns:
        Tuple[float, float]: 전체 문서의 precision과 recall의 튜플입니다.
    '''
    total_true_positive = 0
    total_generated_snippets = 0
    total_gt_snippets = 0
    
    for section in generated_doc:
        # generated_doc과 gt_doc에서 목차에 해당하는 내용을 추출
        generated_content = generated_doc[section]
        gt_content = gt_doc[section]
        
        # 각각의 문서에서 code_snippet 추출
        generated_snippets = find_code_snippets(generated_content)
        gt_snippets = find_code_snippets(gt_content)
        
        # 각 목차에서 true_positive 계산
        true_positive = len(set(generated_snippets) & set(gt_snippets))
        
        # 총 true_positive, generated_snippets, gt_snippets 합산
        total_true_positive += true_positive
        total_generated_snippets += len(generated_snippets)
        total_gt_snippets += len(gt_snippets)
    
    # 전체 precision/recall 계산
    precision, recall = calculate_precision_recall(total_true_positive, total_generated_snippets, total_gt_snippets)
    
    return precision, recall


##########################블로그 초안 작성하는 노드와 관련 함수 정의###############################

model = ChatUpstage(model="solar-pro")

def write(model, q_and_a:QA, document:str):
    '''
    주어진 문서 제목에 따라 ChatGPT와의 대화에서 질문-답변 쌍을 요약하고 정리합니다.
    Solar-LLM : 블로그 초안 작성
    - 사용 모델명 : 문서 작성을 위한 모델로, ChatGPT와의 대화 내용을 바탕으로 요약
    Args:
        model: 모델 객체
        q_and_a (QA): 질문-답변 쌍을 나타내는 QA 객체. 
                      질문은 [Q]로, ChatGPT의 응답은 [A]로 구분됩니다.
        document (str): 제목이 포함된 문서 
    '''
    writing_prompt = langfuse.get_prompt("writing_prompt")
    prompt = writing_prompt.compile(q=q_and_a['q'], a=q_and_a['a'], document=document)
    updated_doc = model.invoke(prompt)
    return updated_doc, prompt
    

def remove_after_second_hashes(text:str) -> str:
    '''
    주어진 텍스트에서 두 번째 해시(#) 이후의 내용을 제거하는 함수입니다.

    Args:
        text (str): 해시를 포함한 원본 문자열입니다.

    Returns:
        str: 두 번째 해시 이후의 내용이 제거된 문자열입니다.
    '''
    # '##'를 기준으로 문자열을 나누기
    parts = text.split('##')
    
    # 두 번째 '##' 이후의 내용 삭제
    if len(parts) > 2:
        return '## ' + parts[1].strip()  # '##' 포함 첫 번째 부분만 반환
    return text  # '##'가 두 번 이상 없을 때는 원래 문자열 반환

def create_draft_blog(state: GraphState) -> GraphState:
    '''
    GraphState를 입력으로 받아 문서 초안을 생성하는 함수입니다.

    Args:
        state (GraphState): 문서 초안 생성을 위한 상태 정보를 포함하는 GraphState 객체입니다.

    Returns:
        GraphState: 생성된 문서 초안을 포함하는 GraphState 객체입니다.
    '''
    
    preprocessed_conversations = state['preprocessed_conversations']
    for i in tqdm(range(len(preprocessed_conversations))):
        qa = preprocessed_conversations[i]
        indices_for_qa = state['message_to_index_dict'][str(i)]

        # 인덱스 목록을 기반으로 문서 업데이트
        for index in indices_for_qa:
            document = state['final_documents'][index]
            for i in range(10):
                generated_doc, _ = write(model, qa, document)

                # 생성된 문서에서 두 번째 해시(#) 이후 내용을 제거
                updated_doc = remove_after_second_hashes(generated_doc.content)
                
                # 생성된 문서가 특정 패턴을 포함하지 않을 때 루프 종료
                unwanted_patterns = ['[Q]', '[A]', '[Doc]', '[/Q]', '[/A]', '[/Doc]', '```']
                if all(pattern not in updated_doc for pattern in unwanted_patterns):
                    break

            state['final_documents'][index] = updated_doc

    return state


##################블로그 초안을 받아 하나의 코드가 하나의 목차에만 들어가게 수정하는 노드와 관련 함수 정의##############

def extract_heading(text:str) -> str:
    '''
    주어진 텍스트에서 헤딩을 추출하는 함수입니다.

    Args:
        text (str): 헤딩을 추출할 문자열입니다.

    Returns:
        str: 추출된 헤딩 문자열입니다.
    '''
    start_index = text.find("##")
    if start_index == -1: 
        return ""
    
    end_index = text.find("\n", start_index)
    if end_index == -1: 
        return text[start_index:].strip()
    
    return text[start_index+2:end_index].strip()

def find_indices_and_snippet_with_code_id(code_id: str, doc_dict:dict[str, str]) -> Tuple[list[str], list[str], str]:
    '''
    주어진 코드 ID를 기반으로 문서에서 코드 스니펫과 해당 인덱스를 찾는 함수입니다.

    Args:
        code_id (str): 검색할 코드 스니펫의 ID입니다. (예시 : Code_Snippet_1)
        doc_dict: 코드 스니펫이 포함된 문서의 딕셔너리.

    Returns:
        Tuple[list[str], list[str], str]: 코드 스니펫과 해당 인덱스의 리스트, 그리고 코드 ID를 반환합니다.
    '''
    pattern = fr"<-- {code_id}: .*? -->"
    indices_list = []
    heading_list = []
    whole_snippet = None

    for key in doc_dict:
        text = doc_dict[key]  # 현재 문서 텍스트 가져오기
        match = re.search(pattern, text)  # 패턴에 맞는 코드 스니펫 검색
        if match:
            heading = extract_heading(text) # 현재 문서의 제목 추출
            indices_list.append(key)
            heading_list.append(heading)
            whole_snippet = match.group().strip()
    return indices_list, heading_list, whole_snippet

def make_heading_list_for_prompt(heading_list: list[str]) -> str:
    '''
    주어진 헤딩 리스트를 프롬프트 형식으로 변환하는 함수입니다.

    Args:
        heading_list (list[str]): 변환할 헤딩 문자열의 리스트입니다.

    Returns:
        str: 프롬프트 형식으로 변환된 헤딩 문자열입니다.
    '''
    text = ''
    for heading in heading_list:
        text = text + heading + '\n'
    return text[:-1]

def refine_draft_blog(state: GraphState) -> GraphState:
    '''
    블로그 초안을 다듬고 코드 스니펫을 문서에서 제거하는 함수입니다.

    Solar-LLM : 주어진 코드 설명과 가장 잘 어울리는 제목을 찾고, 문서에서 코드 스니펫과 관련된 내용을 제거
    사용 모델명 : solar-pro

    Args:
        state (GraphState): 현재 상태를 나타내는 딕셔너리로, 
                            'final_documents'와 'code_document'를 포함해야 합니다. 
                            'final_documents'는 후처리할 블로그 내용을 포함하고, 
                            'code_document'는 코드 스니펫에 대한 정보를 포함합니다.
    
    Returns:
        GraphState: 후처리된 블로그 내용이 업데이트된 상태 딕셔너리입니다.
    '''
    # 코드 스니펫의 키 목록을 가져옴
    code_list = list(state['code_document'].keys())

    # 두 가지 문서 정제 프롬프트를 가져옴
    document_refinement_1 = langfuse.get_prompt("document_refinement_1")
    document_refinement_2 = langfuse.get_prompt("document_refinement_2")
    
    # 각 코드 스니펫에 대해 반복
    for code_id in tqdm(code_list):
        # 코드 ID에 해당하는 코드 스니펫의 인덱스 및 전체 스니펫을 찾음
        indices_list, heading_list, whole_snippet = find_indices_and_snippet_with_code_id(code_id, state['final_documents'])
        
        # 인덱스가 2개 미만인 경우 다음으로 넘어감
        if len(indices_list) < 2:
            continue        
        
        # 제목 목록을 프롬프트용으로 포맷팅
        indices = make_heading_list_for_prompt(heading_list)

        # 첫 번째 프롬프트를 작성하여 코드 스니펫과 관련된 제목을 찾음
        prompt = document_refinement_1.compile(code_snippet=whole_snippet, indices=indices)
        
        # 최대 10번까지 모델을 호출하여 제목을 선택
        for i in range(10):
            selected = model.invoke(prompt)
            selected = selected.content
            
            # 선택한 제목이 인덱스 목록에 있는 경우 종료
            if selected in indices_list:
                break

        # 선택한 제목에 따라 각 인덱스를 업데이트
        for index in indices_list:
            if selected != index:  # 선택한 제목이 현재 인덱스와 다를 경우
                doc = state['final_documents'][index]  # 현재 문서 가져오기
                prompt = document_refinement_2.compile(code_snippet=whole_snippet, doc=doc)  # 두 번째 프롬프트 준비
                updated = model.invoke(prompt)  # 모델 호출하여 문서 업데이트
                state['final_documents'][index] = updated.content  # 업데이트된 내용을 상태에 반영

    return state  # 최종 업데이트된 상태 반환


######################작성한 블로그의 코드 스니펫을 원래 코드로 교체 및 헤딩 표시(#) 지우기######################

def replace_code_snippets(document:str, snippets_dict:dict[str, str]) -> str:
    '''
    주어진 문서에서 코드 스니펫의 플레이스홀더를 실제 코드로 교체하는 함수입니다.

    Args:
        document (str): 코드 스니펫 플레이스홀더가 포함된 원본 문서 텍스트입니다.
        snippets_dict (dict[str, str]): 코드 스니펫의 키와 실제 코드 내용을 매핑하는 딕셔너리입니다.
                                        키는 플레이스홀더 형식과 일치해야 하며, 값은 교체할 코드 스니펫입니다.

    Returns:
        str: 플레이스홀더가 실제 코드로 교체된 문서 텍스트입니다.
    '''
    # Code_Snippet_1 및 Code_Snippet_2 키를 사용해 대체할 패턴을 찾음
    for snippet_key in snippets_dict:
        # 대체할 패턴을 정의
        pattern = f"<-- {snippet_key}:.*?-->"
        # document 내에서 해당 placeholder를 딕셔너리의 value로 대체
        document = re.sub(pattern, "\n```\n" + snippets_dict[snippet_key] + "\n```\n", document)
    
    return document

def postprocess_blog(state: GraphState)-> GraphState:
    '''
    블로그 글을 후처리하는 함수입니다. 문서 내의 코드 인덱스에 원래 코드를 삽입하고,
    최종 블로그 문서를 생성합니다.

    Args:
        state (GraphState): 후처리를 위한 상태 정보를 포함하는 GraphState 객체입니다.
    Returns:
        GraphState: 후처리가 완료된 최종 문서를 포함하는 GraphState 객체입니다.
    '''
    for index in state['final_documents']:
        text = state['final_documents'][index]
        # 코드 스니펫을 원래 코드로 교체하고, 결과를 t에 저장합니다
        t = replace_code_snippets(text, state['code_document'])
        state['final_documents'][index] = t
    return state


##########################그래프 내의 요소(node, edge)들을 정의###############################
# 메모리를 정의
memory = MemorySaver() 

# 새로운 graph 정의
writer_graph = StateGraph(GraphState)

# 사용할 node를 정의(다른 단계를 수행할 node를 제작하고 싶다면 여기에 node를 추가)
writer_graph.add_node("블로그 초안 작성", create_draft_blog)
writer_graph.add_node("블로그 글 다듬기", refine_draft_blog)
writer_graph.add_node("블로그 글 후처리", postprocess_blog)


# 그래프 시작점정의
writer_graph.set_entry_point("블로그 초안 작성")

# Define Edges
writer_graph.add_edge('블로그 초안 작성', "블로그 글 다듬기")
writer_graph.add_edge("블로그 글 다듬기", "블로그 글 후처리")

# 그래프를 컴파일
compiled_graph = writer_graph.compile(checkpointer=memory)


################  example9 데이터로 모듈 성능을 테스트 ################
if __name__ == '__main__':
    import json

    with open('data_for_writing.json', 'r', encoding='utf-8') as json_file:
        loaded_data = json.load(json_file)

    # 들어갈 graph_state를 정의
    graph_state = GraphState(
        preprocessed_conversations=loaded_data['EXAMPLE9']['preprocessed_conversations'],
        code_document=loaded_data['EXAMPLE9']['code_document'],
        message_to_index_dict=loaded_data['EXAMPLE9']['message_to_index_dict'],
        final_documents=loaded_data['EXAMPLE9']['final_documents']
    )

    # 그래프를 실행
    final_state = compiled_graph.invoke(
        graph_state, 
        config={
            "configurable": {"thread_id": 42}, 
            "callbacks": [langfuse_handler]}
    )

    # evalation
    precision, recall = overall_precision_recall(graph_state['final_documents'], loaded_data['EXAMPLE9']['GT_document'])
    print(f"Overall Precision: {precision:.2f}")
    print(f"Overall Recall: {recall:.2f}")

    # 최종 그래프 출력
    for text in list(final_state['final_documents'].values()):
        print(text, end='\n\n')