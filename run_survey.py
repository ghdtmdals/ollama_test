import os
from ollama_client import ollama_client
from pydantic import BaseModel, Field

import re
import json
import pandas as pd
from natsort import natsorted

from metrics import *

class SurveyResponse(BaseModel):
    responses: list[int] = Field(..., min_length=17, max_length=17, ge = 1, le = 6)

def get_survey(survey_path):
    # 설문 파일 읽기
    with open(survey_path, 'r', encoding='utf-8') as f:
        survey_content = json.load(f)
    
    return survey_content

def run_survey(survey_path, type):
    model_names = client.get_models()
    model_names = [name for name in model_names if name.startswith(type)]

    survey = get_survey(survey_path)
    prompt = f"당신은 다음 설문에 응답하는 역할을 합니다. 설문 내용은 다음과 같습니다: \n\n{survey}" \
              + "\n\n각 질문에 대해 성실하고 일관되게 답변하고 설문 답변 외의 추가 설명은 하지 마세요."
    
    response_format = SurveyResponse.model_json_schema()

    responses = {}
    for model_name in model_names:
        print(f"Running survey with model: {model_name}")
        response = client.chat(model_name, prompt, response_format)
        response = json.loads(response)['responses']
        responses[model_name] = response

    return responses

def get_scores(responses): ...

def set_persona(type):
    assert type in ['basic', 'extend'], "type은 'basic' 또는 'extend'만 가능합니다."

    ### 페르소나 Modelfile 수집
    model_files = list_all_files('./ollama-persona', type)

    for model_file in model_files:
        model_name = os.path.basename(model_file)
        model_name = model_name.replace("_Modelfile", "") + ":v1"
        client.create_model(model_name, model_file)

### 테스트 후에는 삭제
def delete_persona(type):
    assert type in ['basic', 'extend'], "type은 'basic' 또는 'extend'만 가능합니다."

    ### 생성된 모델 확인
    models = client.get_models()
    for model in models:
        if model.startswith(type):
            print("Delete", model)
            client.delete_model(model)

def get_persona_profile(responses):
    model_names = list(responses.keys())
    
    results = {}
    for model_name in model_names:
        modelfile_info = client.client.show(model_name)

        match = re.search(r"### 프로필 시작 ###(.*?)### 프로필 종료 ###", str(modelfile_info), re.DOTALL)
        if match:
            profile_text = match.group(1).strip()
            results[model_name] = profile_text.replace('\\n', ' ')
    
    return results


def list_all_files(root_dir, type):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.startswith(type):
                file_list.append(os.path.join(root, file))
    
    ### 자연 정렬 함수 (p10, p1, p2... 순 정렬을 p1, p2, p10... 순 정렬로 변환)
    return natsorted(file_list)

if __name__ == "__main__":
    ### Ollama Client 생성
    global client
    client = ollama_client()
    
    type = 'basic'

    # print(get_survey('./sample_survey.txt'))
    # set_persona(type = type)
    # delete_persona(type = type)
    # responses = run_survey('./sample_survey.json', type = type)
    # print(responses)
    responses = {'basic_p1:v1': [1, 2, 3, 1, 2, 2, 3, 1, 4, 4, 4, 4, 3, 4, 3, 2, 4], 
                'basic_p2:v1': [2, 1, 3, 2, 3, 3, 3, 1, 4, 4, 4, 4, 3, 4, 4, 2, 4]}

    profiles = get_persona_profile(responses)

    # persona_consistency(client, get_survey("./sample_survey.json"), responses, profiles)
    print(std_diversity(responses))
    print(entropy_diversity(responses))
    print(ceil_floor_effect(responses, get_survey("./sample_survey.json")))

    print(std_diversity(responses, by_item = False))
    print(entropy_diversity(responses, by_item = False))
    print(ceil_floor_effect(responses, get_survey("./sample_survey.json"), by_item = False))

    print(response_consistency_analysis(responses))