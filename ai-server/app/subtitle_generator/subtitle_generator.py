import os
import sys
import yaml
import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from tqdm import tqdm
from dotenv import load_dotenv

# langchain
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_core.messages import HumanMessage
from typing import Annotated, Tuple, Union

# langfuse
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from app.utils import fetch_messages, format_message
from app.db_client import get_db_client
from app.type import QA
from app.constants import CONVERSATION_ID

load_dotenv()

# Langfuse 핸들러 및 클라이언트 초기화
langfuse_handler = CallbackHandler()
langfuse = Langfuse()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 부모 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class SubtitleGenerator():
    def __init__(self, config_path="subtitle_generator.yaml"):
        # YAML 파일에서 설정 로드
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # 설정 값 로드
        logging.info(f'Subtitle generator configuration: {config}')
        self.model = ChatUpstage(model=config.get('model'))
        self.embedding_model = UpstageEmbeddings(model=config.get('embedding_model'))
        self.length_limit = config.get('length_limit')  
        self.merge_strategy = config.get('merge_strategy')
        self.merge_cluster_num = config.get('merge_cluster_num')  
        self.debug = config.get('debug') 

    
    def generate_subtitles(self, question:str, answer:str) -> str:
        '''
        단일 QA 쌍에 대한 자막을 생성합니다.
        Solar-LLM : 질문과 답변 쌍에 대한 자막 생성
        - 사용 모델명: solar-pro

        Args:
            question (str): 생성할 자막의 질문 내용.
            answer (str): 생성할 자막의 답변 내용.
        
        Returns:
            str: 생성된 자막 문자열.
        '''
        subtitle_generation: Annotated[str, HumanMessage] = langfuse.get_prompt("subtitle_generator")
        prompt = subtitle_generation.compile(question=question, answer=answer)
        response = self.model.invoke(prompt)
        return response.content.strip()
    
    def _reorder_subtitles(self, subtitle_list:list[str], qa_index:list[list[int]]) -> Tuple[list[str], list[list[int]]]:
        """
        자막의 순서를 재정렬하여 먼저 언급된 자막이 이전 인덱스를 갖도록 합니다.
        """
        # num_lists의 첫 번째 숫자 출현을 기반으로 리넘버 매핑 생성
        number_map = {}
        next_number = 0
        renumbered_qa_lists = []

        for sublist in qa_index:
            renumbered_sublist = []
            for num in sublist:
                if num not in number_map:
                    number_map[num] = next_number
                    next_number += 1
                renumbered_sublist.append(number_map[num])
            renumbered_qa_lists.append(renumbered_sublist)

        # 리넘버 매핑을 기반으로 subtitle_list 재정렬
        reordered_subtitles = [None] * len(subtitle_list)

        # 매핑에 따라 재정렬된 자막을 채움
        for i, _ in enumerate(reordered_subtitles):
            reordered_subtitles[number_map[i]] = subtitle_list[i]

        return reordered_subtitles, renumbered_qa_lists
    
    def _get_sentence_embedding(self, sentence: Union[str, list[str]])-> list:
        """
        주어진 입력에 대한 문장 임베딩을 계산합니다.
        Solar-LLM : 문장 임베딩 계산
        - 사용 모델명 : solar-embedding-1-large
        
        Args:
            sentence (str 또는 str의 리스트): 임베딩을 계산할 문장 또는 문장 리스트입니다. 
                                                리스트가 제공될 경우, 각 문장은 별도로 처리됩니다.
        Returns:
            list: 각각 4096 차원을 갖는 임베딩의 리스트입니다.
        Raises:
            ValueError: 입력이 문자열도 아니고 문자열 리스트도 아닐 경우 발생합니다.
        """
        result = []
        if isinstance(sentence, list) and all(isinstance(item, str) for item in sentence):
            result = self.embedding_model.embed_documents(sentence)
        elif isinstance(sentence, str):
            result.append(self.embedding_model.embed_query(sentence))
        else:
            raise ValueError("입력 유형은 str 또는 str의 리스트여야 합니다.")
        return result
    
    def _format_data(self, subtitle_list:list[str], qa_index_list:list[list[int]]) -> Tuple[dict[str,str], dict[str,str]]:
        """
        작성자에게 맞는 형태로 데이터를 포맷합니다.
        """
        subtitle_dict = {}
        qa_index_dict = {}
        
        for idx, (subtitle) in enumerate(subtitle_list):
            subtitle = subtitle.replace('"', '')  # 자막의 따옴표 제거
            subtitle_dict[str(idx)] = f'## {idx}) {subtitle}'  # 인덱스와 함께 포맷된 자막 추가
        
        for idx, (qa_index) in enumerate(qa_index_list):
            qa_index_dict[str(idx)] = qa_index  # QA 인덱스 추가

        return subtitle_dict, qa_index_dict

    def merge_subtitle(self, subtitle_list: list[list[str]])-> Tuple[list[str], list[list[int]]]:
        """
        각 QA 쌍에서 생성된 자막을 병합하여 고유한 자막 집합으로 만듭니다.
        Solar-LLM: 클러스터 내의 여러 자막을 단일 자막으로 병합
        - 사용 모델명: solar-pro 

        Args:
            subtitle_list (list(list(str))): 각 QA 쌍에 대해 생성된 자막을 포함하는 리스트입니다.

        Returns:
            subtitles (list(str)): 병합된 고유 자막 리스트입니다.
            subtitle_index (list(list(int))): 각 QA 쌍에 해당하는 자막의 인덱스를 나타내는 리스트입니다.
        """
        logging.info(f'merging start. merging strategy: {self.merge_strategy}')

        result = []

        if self.merge_strategy == 'llm':
            raise NotImplementedError()  # LLM 병합 전략은 구현되지 않음
        
        elif self.merge_strategy == 'embedding':
            subtitle_embedding_list = []  # 각 자막 임베딩의 리스트
            subtitle_embedding_numpy = []  # numpy 배열로 변환할 자막 임베딩 리스트
            subtitle_index = []  # 각 자막이 속한 인덱스 리스트

            # 자막의 임베딩 계산
            for subtitles in subtitle_list:
                subtitle_embedding = self._get_sentence_embedding(subtitles)
                subtitle_embedding_list.append(subtitle_embedding)
            
            # 모든 자막 임베딩 리스트를 하나의 리스트로 결합하여 numpy 벡터로 변환
            for subtitle_embedding in subtitle_embedding_list:
                subtitle_embedding_numpy.extend(subtitle_embedding)
            subtitle_embedding_numpy = np.array(subtitle_embedding_numpy)  # (N, 4096)

            # 클러스터 수가 데이터 포인트 수를 초과하지 않도록 조정
            n_samples = subtitle_embedding_numpy.shape[0]  # 데이터 포인트 개수
            n_clusters = min(n_samples, self.merge_cluster_num)  # 클러스터 수 결정
            
            # KMeans를 사용하여 임베딩 클러스터링
            logging.info('Start calculating kMeans...')
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(subtitle_embedding_numpy)

            # 각 자막 임베딩과 클러스터 중심 간의 유클리드 거리 계산
            logging.info('Find closest kMean point for each subtitle embeddings...')
            for subtitle_embedding in subtitle_embedding_list:
                subtitle_index_sublist = []
                for subtitle_single_embedding in subtitle_embedding:
                    distances = euclidean_distances([subtitle_single_embedding], kmeans.cluster_centers_)
                    subtitle_index_sublist.append(np.argmin(distances))
                subtitle_index.append(subtitle_index_sublist)
            if self.debug:
                print(subtitle_index)
            
            # 동일 클러스터의 자막을 합쳐 LLM으로 단일 자막으로 병합
            subtitle_clustered = [[] for _ in range(n_clusters)]
            for i, (subtitles) in enumerate(subtitle_list):
                for j, (single_subtitle) in enumerate(subtitles):
                    subtitle_clustered[subtitle_index[i][j]].append(single_subtitle)
            if self.debug:
                print(subtitle_clustered)

            logging.info(f'Merging subtitles into {n_clusters} subtitles...')
            for i in range(n_clusters):
                subtitle_merge_prompt: Annotated[str, HumanMessage] = langfuse.get_prompt("subtitle_generator_merge")
                prompt = subtitle_merge_prompt.compile(subtitles=str(subtitle_clustered[i]))
                response = self.model.invoke(prompt)
                result.append(response.content.strip())

            if self.debug:
                print(result, [sorted(set(sublist)) for sublist in subtitle_index])
                
            return result, [sorted(set(sublist)) for sublist in subtitle_index]

        else:
            raise ValueError("지정된 merge_strategy가 유효하지 않습니다.")

    def generate(self, conversation_data:list[QA]) -> list[list[str]]:
        """
        대화에서 자막을 생성합니다.

        Args:
            conversation_data (list[QA]): 
                대화 내용을 포함하는 Q&A 세트의 리스트입니다. 
                각 Q&A 세트는 질문과 답변을 포함한 사전 형식입니다.
                예시: [{'q': '질문', 'a': '답변'}]

        Returns:
            list[list[str]]: 
                각 Q&A 세트에 대해 생성된 자막 리스트를 반환합니다. 
                반환되는 리스트는 각 Q&A 쌍에 해당하는 자막들이 포함된 리스트의 리스트입니다.
                예시: [['자막1', '자막2'], ['자막3', '자막4']]
        """
        subtitle_list = []

        # 각 QA 쌍에 대한 자막 생성
        for idx, conversation in tqdm(enumerate(conversation_data), total=len(conversation_data), desc="Generating Subtitles"):
            conversation = conversation_data[idx]
            
            # 대화가 너무 길 경우 잘라냄
            if (len(conversation['q']) > self.length_limit):
                conversation['q'] = conversation['q'][:self.length_limit]
            if (len(conversation['a']) > self.length_limit):
                conversation['a'] = conversation['a'][:self.length_limit]

            # 각 QA 쌍에 대한 자막 생성
            subtitle_result = self.generate_subtitles(conversation['q'], conversation['a'])

            # 답변을 파싱하고 리스트로 변환
            subtitle_result = subtitle_result.splitlines()
            subtitle_result = [line for line in subtitle_result if line.strip()]

            if self.debug:
                tqdm.write(f"Processed subtitles for QA pair {idx + 1}: {subtitle_result}")

            subtitle_list.append(subtitle_result)
        logging.info('generating subtitles is done.')

        return subtitle_list
    
    def __call__(self, conversation):
        """
        클래스 인스턴스를 호출하여 자막을 생성하고 포맷을 정리합니다.
        """
        subtitle_list = self.generate(conversation)
        subtitle_list, qa_indices = self.merge_subtitle(subtitle_list)
        subtitle_list, qa_indices = self._reorder_subtitles(subtitle_list=subtitle_list, qa_index=qa_indices)
        return self._format_data(subtitle_list, qa_indices)

# 디버깅 목적의 데모
if __name__ == '__main__':
    database = get_db_client()
    conversation = fetch_messages(database, CONVERSATION_ID["EXAMPLE_1"])
    conversation_data = format_message(conversation)
    print('current path:', os.getcwd())
    test = SubtitleGenerator(config_path="../configs/subtitle_generator.yaml")
    print('result:', test(conversation_data))
