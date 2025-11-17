import os
from ollama_client import ollama_client
from pydantic import BaseModel, Field

import re
import json
import pandas as pd
from natsort import natsorted

from datetime import datetime
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
    model_names = natsorted(model_names)

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

def get_scores(responses, comment = "None", consistency_details = False):
    profiles = get_persona_profile(responses)

    ### 프로필 일관성
    similarities = persona_consistency(client, get_survey("./sample_survey.json"), responses, profiles)
    
    std_by_item = std_diversity(responses)
    entropy_by_item = entropy_diversity(responses)
    cf_effect_by_item = ceil_floor_effect(responses, get_survey("./sample_survey.json"))

    std_by_person = std_diversity(responses, by_item = False)
    entropy_by_person = entropy_diversity(responses, by_item = False)
    cf_effect_by_person = ceil_floor_effect(responses, get_survey("./sample_survey.json"), by_item = False)

    consistency = response_consistency_analysis(responses, include_details = consistency_details)

    results = {
        "Comment": comment,
        "Profile Similarity": similarities,
        "Std by Item": std_by_item,
        "Entropy by Item": entropy_by_item,
        "CF Effect by Item": cf_effect_by_item,
        "Std by Person": std_by_person,
        "Entropy by Person": entropy_by_person,
        "CF Effect by Person": cf_effect_by_person,
        "Response Consistency": consistency,
    }

    if not os.path.isdir("./metrics_records"):
        os.mkdir("./metrics_records")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"./metrics_records/{timestamp}.json", 'w', encoding = 'utf-8-sig') as f:
        json.dump(results, f, ensure_ascii = False, indent = 3)

    return json.dumps(results, ensure_ascii = False, indent = 3)

def set_persona(type, parameters = None):
    assert type in ['basic', 'extend'], "type은 'basic' 또는 'extend'만 가능합니다."

    ### 페르소나 Modelfile 수집
    model_files = list_all_files('./ollama-persona', type)

    for model_file in model_files:
        model_name = os.path.basename(model_file)
        model_name = model_name.replace("_Modelfile", "") + ":v1"
        client.create_model(model_name, model_file, parameters = parameters)

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

def save_survey_results(responses, save_raw = False, survey_path = "./sample_survey.json", base_dir = "./survey_results"):
    # 폴더 생성
    os.makedirs(base_dir, exist_ok = True)

    # 설문 문항 로드
    with open(survey_path, "r", encoding = "utf-8-sig") as f:
        survey = json.load(f)

    # timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"survey_result_{timestamp}.json"
    filepath = os.path.join(base_dir, filename)

    # responses_detailed 생성
    responses_detailed = {}

    for model_name, answer_list in responses.items():
        model_answers = []

        for i, answer_value in enumerate(answer_list):
            if i >= len(survey):  # 안전 처리
                break
            item = survey[i]
            q_id = item.get("id")
            q_text = item.get("question")
            options = item.get("options", {})

            answer_label = options.get(str(answer_value), None)

            model_answers.append({
                "id": q_id,
                "question": q_text,
                "answer_value": answer_value,
                "answer_label": answer_label
            })

        responses_detailed[model_name] = model_answers

    # === 최종 저장 구조 ===
    if save_raw:
        result_to_save = {
            "timestamp": timestamp,
            "responses_raw": responses,
            "responses_detailed": responses_detailed
        }

    else:
        result_to_save = {
            "timestamp": timestamp,
            "responses_detailed": responses_detailed
        }

    # JSON 저장
    with open(filepath, "w", encoding="utf-8-sig") as f:
        json.dump(result_to_save, f, ensure_ascii=False, indent=2)

### 미리 세팅해놓은 파라미터들을 적용하며 테스트
def run_with_parameters(type, parameters_path, show_results = False):
    with open(parameters_path, "r", encoding = "utf-8-sig") as f:
        parameters = json.load(f)

    for i, parameter in enumerate(parameters):
        print(f"({i + 1}/{len(parameters)})Running test with Parameters: {parameter}")
        set_persona(type = type, parameters = parameter)
        responses = run_survey('./sample_survey.json', type = type)
        save_survey_results(responses, save_raw = False)

        comment = f"Model_Names: {list(responses.keys())} | Applied Parameters from {parameters_path} | Settings: {parameter}"
        scores = get_scores(responses, comment)

        if show_results:
            print(scores)

    delete_persona(type = type)

### Modelfile 파라미터 수정 없이 테스트
def run_without_parameters(type, show_results = False):
    set_persona(type = type)
    responses = run_survey('./sample_survey.json', type = type)
    save_survey_results(responses, save_raw = False)

    comment = f"Model_Names: {list(responses.keys())}"
    scores = get_scores(responses, comment)

    if show_results:
        print(scores)

    delete_persona(type = type)

if __name__ == "__main__":
    ### Ollama Client 생성
    global client
    client = ollama_client()

    type = "basic"
    # run_without_parameters(type = type)

    parameters = "experiment1_temperature.json"
    parameters_base_path = "./experiments"
    run_with_parameters(type = type, parameters_path = os.path.join(parameters_base_path, parameters))

