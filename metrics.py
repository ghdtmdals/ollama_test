from ollama_client import ollama_client
from pydantic import BaseModel, Field
import json
import numpy as np
import pandas as pd

from itertools import combinations, product
from scipy.stats import entropy
from scipy.stats import spearmanr

class PorfileComparison(BaseModel):
    profile_similarity: float = Field(..., ge=0.0, le=1.0)

### 페르소나 프로필과 인구통계 문항 비교, LLM 기반 유사도 측정
def persona_consistency(client: ollama_client, survey, responses, persona_profile):
    profile_cmp_group = [1, 2, 3, 4, 5, 6, 7, 8] ### 프로필 비교 문항

    ### 프로필에 해당하는 설문 문항만 추출
    profile_items = []
    for item in survey:
        if item['id'] in profile_cmp_group:
            profile_items.append(item)

    model_names = list(responses.keys())
    temp_responses = {}
    
    ### 모델별 프로필 관련 문항 답변 수집
    for model_name in model_names:
        temp_responses[model_name] = []

        for i, item in enumerate(survey):
            if item['id'] in profile_cmp_group:
                temp_responses[model_name].append(item['options'][str(responses[model_name][i])])

    response_format = PorfileComparison.model_json_schema()

    ### 모델별 유사도 측정
    similarities = {}
    for model_name in model_names:
        prompt = f"다음은 페르소나의 프로필 정보입니다. {persona_profile[model_name]}\n" \
            + f"다음은 설문조사에서 프로필에 해당하는 문항들입니다: {profile_items}" \
            + f"다음은 설문조사에서 해당 페르소나가 선택한 프로필 관련 문항의 답변들입니다: {temp_responses[model_name]}\n" \
            + "이 페르소나의 프로필 정보와 설문조사 답변들 간의 유사도를 0에서 1 사이의 값으로 평가해주세요." \
            + "0은 전혀 유사하지 않음을 의미하고, 1은 완전히 일치함을 의미합니다. 유사도 점수만 JSON 형식으로 응답해주세요." \
            + "프로필 정보는 답변과 완벽히 일치하지 않을 수 있습니다. 예를 들어 35세일 경우 30대로 응답할 수 있으며, " \
            + "서울에서 거주할 경우 수도권으로 응답할 수 있습니다. 이러한 점을 고려하여 유사도를 평가해주세요."
        response = client.chat(model_name = "gemma3:27b", prompt = prompt, format = response_format)
        similarities[model_name] = json.loads(response)['profile_similarity'] ### 왜 1에 가깝게 나오지 않는걸까? 프롬프트 엔지니어링으로 개선 필요
    
    return similarities

#############################################################################################
######################################### 문항별/개인별 측정 #########################################
#############################################################################################

### 표준편차 기반 다양성 측정
def std_diversity(responses, by_item = True, average = False):
    responses = np.array(list(responses.values()))
    
    ### 문항별 대신 개인별 계산
    if not by_item:
        responses = responses.T

    if average:
        ### 전체 평균 표준편차 반환
        return np.mean(np.std(responses, axis = 0))
    else:
        ### 표준편차 반환
        return np.std(responses, axis = 0)

### 문항별 엔트로피 기반 다양성 측정
def entropy_diversity(responses, by_item = True, average = False):
    responses = np.array(list(responses.values()))
    
    ### 개인별 계산
    if not by_item:
        responses = responses.T
    
    num_models, num_items = responses.shape

    entropies = []
    for i in range(num_items):
        item_responses = responses[:, i]
        value, counts = np.unique(item_responses, return_counts=True)
        probabilities = counts / num_models
        entropies.append(entropy(probabilities, base=2)) ### Shannon entropy 계산, 1에 가까울수록 다양성이 높음

    if average:
        ### 전체 평균 엔트로피 반환
        return np.mean(entropies)
    else:
        ### 엔트로피 반환
        return np.array(entropies)

### 문항별 Ceiling/Floor Effect 분석
def ceil_floor_effect(responses, survey, by_item = True, average = False, ceiling_threshold=0.8, floor_threshold=0.8):
    responses = np.array(list(responses.values()))
    
    ### 개인별 계산
    if not by_item:
        responses = responses.T

    # 척도 범위 자동 설정
    min_scores = []
    max_scores = []
    for item in survey:
        min_scores.append(int(min(item['options'].keys())))
        max_scores.append(int(max(item['options'].keys())))
    
    min_scores = np.array(min_scores)
    max_scores = np.array(max_scores)

    ### floor 값의 비율과 ceiling 값의 비율 계산
    floor_counts = np.zeros(responses.shape[1])
    ceil_counts = np.zeros(responses.shape[1])

    for i, response in enumerate(responses):
        if by_item:
            floor_counts += (response == min_scores)
            ceil_counts += (response == max_scores)
        else:
            floor_counts += (response == min_scores[i])
            ceil_counts += (response == max_scores[i])
    
    ### by_item일 경우 답변한 모델의 수로 나눔 -> 문항별 바율
    ### 반대의 경우 문항의 수로 나눔 -> 개인별 비율
    floor_ratios = floor_counts / responses.shape[0]
    ceil_ratios = ceil_counts / responses.shape[0]

    if average:
        return np.mean(floor_ratios), np.mean(ceil_ratios)
    
    else:
        return {"Floor Ratios": floor_ratios, "Ceil Ratios": ceil_ratios}

#############################################################################################
#############################################################################################
#############################################################################################




#############################################################################################
######################################### 문항간 측정 #########################################
#############################################################################################
def response_consistency_analysis(responses, include_details = False, opposite_neg_threshold=0.0):
    # 일관성 비교 그룹 정의 (질문에서 주신 그대로 사용)
    # Level 1: 그룹 리스트
    # Level 2: 그룹 내 문항 리스트, 리스트 수 == 2: 반대 문항
    # Level 3: 문항 번호 리스트, 유사 문항 리스트
    consistency_chk_groups = [
        [[9, 11, 17]],          # 만족도
        [[10, 12, 15]],         # 적합도 및 성장성
        [[9, 14]],              # 근무 환경, 조직 문화 만족도
        [[11, 13, 17]],         # 지속 의향
        [[16], [9, 11, 14, 17]] # 스트레스 vs 만족도 (반대 문항 그룹)
    ]

    # dict → (n_items, n_respondents) 배열로 변환 (질문에서 사용하신 방식 유지)
    # responses.values(): 각 문항의 응답 리스트
    # .T: (응답자 수, 문항 수) → (문항 수, 응답자 수)
    responses_arr = np.array(list(responses.values()))
    responses_arr = responses_arr.T

    group_results = []
    all_within_corrs = []       # 전체 그룹 내 상관
    all_opposite_corrs = []     # 전체 반대 문항 상관

    for g_idx, group in enumerate(consistency_chk_groups, start=1):
        # 1) 반대 문항이 없는 경우 (len(group) == 1)
        if len(group) == 1:
            items = group[0]  # 예: [9, 11, 17]
            pair_corrs = []

            # 같은 그룹 내 모든 쌍 상관관계 계산
            for a, b in combinations(items, 2):
                x = responses_arr[a - 1]  # 문항 번호가 1부터 시작한다고 가정
                y = responses_arr[b - 1]
                corr, _ = spearmanr(x, y)
                pair_corrs.append({"pair": (a, b), "corr": corr})
                all_within_corrs.append(corr)

            group_results.append({
                "group_index": g_idx,
                "type": "single",
                "items": items,
                "within_pair_corrs": pair_corrs,
                "avg_within_corr": float(np.nanmean([p["corr"] for p in pair_corrs])) if pair_corrs else np.nan
            })

        # 2) 반대 문항 두 개의 그룹이 있는 경우 (len(group) == 2)
        else:
            group_a = group[0]  # 예: [16]
            group_b = group[1]  # 예: [9, 11, 14, 17]

            within_a = []
            within_b = []
            across_ab = []

            # A 그룹 내부 상관
            if len(group_a) > 1:
                for a, b in combinations(group_a, 2):
                    x = responses_arr[a - 1]
                    y = responses_arr[b - 1]
                    corr, _ = spearmanr(x, y)
                    within_a.append({"pair": (a, b), "corr": corr})
                    all_within_corrs.append(corr)

            # B 그룹 내부 상관
            if len(group_b) > 1:
                for a, b in combinations(group_b, 2):
                    x = responses_arr[a - 1]
                    y = responses_arr[b - 1]
                    corr, _ = spearmanr(x, y)
                    within_b.append({"pair": (a, b), "corr": corr})
                    all_within_corrs.append(corr)

            # A ↔ B (반대 문항들 간 상관)
            inconsistent_pairs = []
            for a, b in product(group_a, group_b):
                x = responses_arr[a - 1]
                y = responses_arr[b - 1]
                corr, _ = spearmanr(x, y)
                record = {
                    "pair": (a, b),
                    "corr": corr,
                    "is_inconsistent": corr > opposite_neg_threshold
                }
                across_ab.append(record)
                all_opposite_corrs.append(corr)
                if record["is_inconsistent"]:
                    inconsistent_pairs.append(record)

            group_results.append({
                "group_index": g_idx,
                "type": "opposite",
                "group_a": group_a,
                "group_b": group_b,
                "within_a_corrs": within_a,
                "within_b_corrs": within_b,
                "across_corrs": across_ab,
                "avg_within_a_corr": float(np.nanmean([p["corr"] for p in within_a])) if within_a else np.nan,
                "avg_within_b_corr": float(np.nanmean([p["corr"] for p in within_b])) if within_b else np.nan,
                "avg_across_corr": float(np.nanmean([p["corr"] for p in across_ab])) if across_ab else np.nan,
                "inconsistent_pairs": inconsistent_pairs,  # 반대인데 음(-)이 아닌 상관들
            })

    if include_details:
        result_summary = {
            "group_results": group_results,
            "overall_avg_within_corr": float(np.nanmean(all_within_corrs)) if all_within_corrs else np.nan,
            "overall_avg_opposite_corr": float(np.nanmean(all_opposite_corrs)) if all_opposite_corrs else np.nan,
        }
    else:
        result_summary = {
            "overall_avg_within_corr": float(np.nanmean(all_within_corrs)) if all_within_corrs else np.nan,
            "overall_avg_opposite_corr": float(np.nanmean(all_opposite_corrs)) if all_opposite_corrs else np.nan,
        }

    return json.dumps(result_summary, ensure_ascii = False, indent = 3)

#############################################################################################
#############################################################################################
#############################################################################################