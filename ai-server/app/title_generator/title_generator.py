import os
import yaml
import logging

from tqdm import tqdm
from dotenv import load_dotenv

# langchain
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage
from typing import Annotated

# langfuse
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

load_dotenv()

# Langfuse 핸들러 및 클라이언트 초기화
langfuse_handler = CallbackHandler()
langfuse = Langfuse()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TitleGenerator():
    """
    TitleGenerator

    SubtitleGenerator에서 생성한 subtitle dict를 입력으로 받아 blog post의 title을 generate하여 str 형태로 반환합니다.

    Example Usage:
        generator = TitleGenerator("title_generator.yaml")
        subtitle_dict, qa_indices = subtitle_generator(input)
        title = generator(subtitle_dict)
    """
    def __init__(self, config_path="title_generator.yaml"):
        # YAML 파일에서 설정 로드

        full_path = os.path.abspath(config_path)
        print ('full_path:', full_path)
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        else:
            raise FileNotFoundError(f"The specified file '{config_path}' does not exist.")
        
        # 설정 값 로드
        logging.info(f'Title generator configuration: {config}')
        self.model = ChatUpstage(model=config.get('model'))
        self.debug = config.get('debug')

    def generate(self, subtitle_string:str) -> str:
        '''
        생성된 subtitle set을 확인하고 이를 이용하여 전체 post의 title을 generate한 뒤 return합니다.
        Solar-LLM: 제공되는 subtitles에 대해서 모든 subtitles 내용을 포괄하는 title을 generation하기 위해 사용합니다.
        - 사용 모델명: solar-pro

        Args:
            subtitle_list (str):
                지정된 Q&A 세트에 대해 생성된 subtitle들이 '\n'을 두고 한 줄에 하나씩 제공되는 형태의 string입니다.
        
        Returns:
            str: 전체 post에 대한 title을 string 형태로 return합니다.
        '''
        subtitle_generation: Annotated[str, HumanMessage] = langfuse.get_prompt("title_generator")
        prompt = subtitle_generation.compile(subtitle_string=subtitle_string)
        response = self.model.invoke(prompt)
        return response.content.strip()
    
    def _dict_to_str(self, subtitle_dict):
        '''
        SubtitleGenerator module에서 생성한 dict를 LLM에게 feed할 수 있는 string 형태로 변환합니다.
        '''
        result = ''
        for key, value in subtitle_dict.items():
            result += f'{value[3:]}\n'
        return result

    
    def __call__(self, subtitle_dict:dict) -> str:
        """
        generate를 call하여 전체 post에 대한 title을 generate한 뒤 return합니다.

        Args:
            subtitle_dict (dict[str]):
                SubtitleGenerator에서 return하는 subtitle dict입니다.
        
        Returns:
            title (str): 전체 post에 대한 string type의 title입니다. 
        """
        subtitle_string = self._dict_to_str(subtitle_dict)
        title = self.generate(subtitle_string)
        return title

# 디버깅 목적의 데모
if __name__ == '__main__':
    print('current path:', os.getcwd())
    example_dict = {"0": "## 0) Mastering Connection Retry Strategies for Popular Databases: MySQL, PostgreSQL, and MongoDB", "1": "## 1) Integrating MongoDB Database Connectivity into Your Project", "2": "## 2) Tackling PostgreSQL Connection Timeouts: A Practical Guide with Code Examples", "3": "## 3) Code-Driven Guide to Connecting to MySQL and PostgreSQL: Examples and Best Practices"}
    print(os.getcwd())
    test = TitleGenerator(config_path="../configs/title_generator.yaml")
    result = test(example_dict)
    print (result)
