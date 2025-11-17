import os
import glob
import json
import pandas as pd
from datetime import datetime
from natsort import natsorted

def load_experiment_files(base_path, pattern="*.json"):
    experiments = {}
    for path in glob.glob(os.path.join(base_path, pattern)):
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        # 파일명에서 확장자 제거하여 실험 이름으로 사용
        exp_name = os.path.splitext(os.path.basename(path))[0]
        experiments[exp_name] = data

    return experiments

# ----------------------------------------
# 2. 각 지표별 DataFrame 생성 함수들
# ----------------------------------------
def build_profile_similarity_df(experiments):
    """
    Profile Similarity: {model_name: value} 구조를
    index=model_name, column=experiment_name 인 DataFrame으로 변환
    """
    records = []
    for exp_name, data in experiments.items():
        prof = data.get("Profile Similarity", {})
        for model_name, value in prof.items():
            records.append({"experiment": exp_name,
                            "model": model_name,
                            "value": value})
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.pivot(index="model", columns="experiment", values="value")

def build_vector_metric_df(experiments, key):
    """
    'Std by Item', 'Entropy by Item', 'Std by Person', 'Entropy by Person'
    같이 리스트 형태인 지표를
    index=번호(0~), column=experiment 인 DataFrame으로 변환
    """
    records = []
    for exp_name, data in experiments.items():
        vec = data.get(key, [])
        for idx, value in enumerate(vec):
            records.append({"experiment": exp_name,
                            "index": idx,
                            "value": value})
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.pivot(index="index", columns="experiment", values="value")

### 모델 정보 보존 용도
def build_std_entropy_by_person_df(experiments, key):
    """
    key = "Std by Person" or "Entropy by Person"
    Output:
        index = model_name
        columns = experiments sorted naturally
    """
    first_exp = next(iter(experiments))
    model_list = natsorted(list(experiments[first_exp]["Profile Similarity"].keys()))

    records = []
    for exp_name, data in experiments.items():
        vec = data.get(key, [])
        for i, model_name in enumerate(model_list):
            records.append({
                "experiment": exp_name,
                "model": model_name,
                "value": vec[i] if i < len(vec) else None
            })

    df = pd.DataFrame(records)
    df = df.pivot(index="model", columns="experiment", values="value")

    # natural sorted both axes
    df = df.loc[natsorted(df.index), natsorted(df.columns)]
    return df

def build_cf_effect_by_item_df(experiments):
    """
    CF Effect by Item 안의 Floor Ratios, Ceil Ratios를 각각 DataFrame으로 생성
    """
    floor_records = []
    ceil_records = []

    for exp_name, data in experiments.items():
        cf_item = data.get("CF Effect by Item", {})
        floors = cf_item.get("Floor Ratios", [])
        ceils = cf_item.get("Ceil Ratios", [])
        for idx, value in enumerate(floors):
            floor_records.append({"experiment": exp_name,
                                  "index": idx,
                                  "value": value})
        for idx, value in enumerate(ceils):
            ceil_records.append({"experiment": exp_name,
                                 "index": idx,
                                 "value": value})

    floor_df = pd.DataFrame(floor_records)
    ceil_df = pd.DataFrame(ceil_records)

    if not floor_df.empty:
        floor_df = floor_df.pivot(index="index", columns="experiment", values="value")
    if not ceil_df.empty:
        ceil_df = ceil_df.pivot(index="index", columns="experiment", values="value")

    return floor_df, ceil_df

def build_cf_effect_by_person_df(experiments):
    """
    CF Effect by Person 안의 Floor Ratios, Ceil Ratios를 각각 DataFrame으로 생성
    index=model_name, column=experiment
    => index=basic_p1:v1 ... 형태로 변경하고 natsort 적용
    """
    # 모델 이름 추출 (Profile Similarity 기준)
    first_exp = next(iter(experiments))
    model_list = natsorted(list(experiments[first_exp]["Profile Similarity"].keys()))

    floor_records = []
    ceil_records = []

    for exp_name, data in experiments.items():
        cf_person = data.get("CF Effect by Person", {})
        floors = cf_person.get("Floor Ratios", [])
        ceils = cf_person.get("Ceil Ratios", [])

        for i, model_name in enumerate(model_list):
            floor_records.append({"experiment": exp_name,
                                  "model": model_name,
                                  "value": floors[i] if i < len(floors) else None})
            ceil_records.append({"experiment": exp_name,
                                 "model": model_name,
                                 "value": ceils[i] if i < len(ceils) else None})

    floor_df = pd.DataFrame(floor_records)
    ceil_df = pd.DataFrame(ceil_records)

    floor_df = floor_df.pivot(index="model", columns="experiment", values="value")
    ceil_df = ceil_df.pivot(index="model", columns="experiment", values="value")

    # natsort on index & columns
    floor_df = floor_df.loc[natsorted(floor_df.index), natsorted(floor_df.columns)]
    ceil_df = ceil_df.loc[natsorted(ceil_df.index), natsorted(ceil_df.columns)]

    return floor_df, ceil_df

def build_response_consistency_df(experiments):
    """
    Response Consistency 안의 overall_avg_within_corr, overall_avg_opposite_corr를
    index=experiment, column=지표명 으로 정리
    """
    records = []
    for exp_name, data in experiments.items():
        rc = data.get("Response Consistency", {})
        records.append({
            "experiment": exp_name,
            "overall_avg_within_corr": rc.get("overall_avg_within_corr", None),
            "overall_avg_opposite_corr": rc.get("overall_avg_opposite_corr", None),
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.set_index("experiment")

# ----------------------------------------
# 3. 공통 포맷터: "값 (+직전 대비)" 문자열로 변환
# ----------------------------------------
def format_with_prev_diff(df, decimals=4):
    """
    df: index=지표(또는 모델/문항/사람), columns=experiment 의 숫자 DataFrame
    각 셀을 "현재값 (+직전실험대비차이)" 형태의 문자열로 변환한 DataFrame 반환
    (첫 실험은 변화량 없이 값만 출력)
    """
    if df.empty:
        return df

    # 실험 이름 정렬(파일명 기준)
    df = df[sorted(df.columns)]
    formatted = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)

    cols = list(df.columns)

    for j, col in enumerate(cols):
        col_vals = df[col]
        if j == 0:
            # 첫 번째 실험: 그냥 값만
            formatted[col] = col_vals.round(decimals).map(
                lambda x: f"{x:.{decimals}f}" if pd.notnull(x) else ""
            )
        else:
            prev_col = cols[j - 1]
            diff_vals = col_vals - df[prev_col]

            def fmt(v, d):
                if pd.isnull(v):
                    return ""
                if pd.isnull(d):
                    return f"{v:.{decimals}f}"
                return f"{v:.{decimals}f} ({d:+.{decimals}f})"

            formatted[col] = [
                fmt(v, d) for v, d in zip(col_vals, diff_vals)
            ]

    return formatted

# ----------------------------------------
# 4. 사용 예시
# ----------------------------------------
if __name__ == "__main__":
    # 1) JSON들이 들어있는 폴더 지정
    base_path = "./metrics_records"   # <- 본인 폴더 경로로 변경

    # 2) 실험 파일 로딩
    experiments = load_experiment_files(base_path)

    # 3) 지표별 DataFrame 생성 (숫자형)
    df_profile = build_profile_similarity_df(experiments)
    df_profile = df_profile.loc[natsorted(df_profile.index), natsorted(df_profile.columns)] ### 자연 정렬

    df_std_item = build_vector_metric_df(experiments, "Std by Item")
    df_entropy_item = build_vector_metric_df(experiments, "Entropy by Item")
    df_std_person = build_std_entropy_by_person_df(experiments, "Std by Person")
    df_entropy_person = build_std_entropy_by_person_df(experiments, "Entropy by Person")


    cf_item_floor_df, cf_item_ceil_df = build_cf_effect_by_item_df(experiments)
    cf_person_floor_df, cf_person_ceil_df = build_cf_effect_by_person_df(experiments)

    df_resp_consistency = build_response_consistency_df(experiments)  # index=experiment

    # 4) 각 지표에 대해 "값 (+직전 대비)" 포맷의 DataFrame 생성
    formatted_profile = format_with_prev_diff(df_profile)
    formatted_std_item = format_with_prev_diff(df_std_item)
    formatted_entropy_item = format_with_prev_diff(df_entropy_item)
    formatted_std_person = format_with_prev_diff(df_std_person)
    formatted_entropy_person = format_with_prev_diff(df_entropy_person)

    formatted_cf_item_floor = format_with_prev_diff(cf_item_floor_df)
    formatted_cf_item_ceil = format_with_prev_diff(cf_item_ceil_df)
    formatted_cf_person_floor = format_with_prev_diff(cf_person_floor_df)
    formatted_cf_person_ceil = format_with_prev_diff(cf_person_ceil_df)

    # Response Consistency 는 index=experiment 이므로 transpose 해서
    # index=지표명, columns=experiment 형태로 바꾼 뒤 동일 포맷 적용
    formatted_resp_consistency = format_with_prev_diff(df_resp_consistency.T)

    # 5) 출력 예시
    print("\n[Profile Similarity] (값, 그리고 괄호 안은 직전 실험 대비 변화량)")
    print(formatted_profile)

    print("\n[Std by Item]")
    print(formatted_std_item)

    print("\n[Entropy by Item]")
    print(formatted_entropy_item)

    print("\n[Std by Person]")
    print(formatted_std_person)

    print("\n[Entropy by Person]")
    print(formatted_entropy_person)

    print("\n[CF Effect by Item - Floor Ratios]")
    print(formatted_cf_item_floor)

    print("\n[CF Effect by Item - Ceil Ratios]")
    print(formatted_cf_item_ceil)

    print("\n[CF Effect by Person - Floor Ratios]")
    print(formatted_cf_person_floor)

    print("\n[CF Effect by Person - Ceil Ratios]")
    print(formatted_cf_person_ceil)

    print("\n[Response Consistency] (overall_avg_within_corr / overall_avg_opposite_corr)")
    print(formatted_resp_consistency)

    formatted = {
        "Profile Similarity": formatted_profile,
        "Std by Item": formatted_std_item,
        "Entropy by Item": formatted_entropy_item,
        "Std by Person": formatted_std_person,
        "Entropy by Person": formatted_entropy_person,
        "CF Item - Floor Ratios": formatted_cf_item_floor,
        "CF Item - Ceil Ratios": formatted_cf_item_ceil,
        "CF Person - Floor Ratios": formatted_cf_person_floor,
        "CF Person - Ceil Ratios": formatted_cf_person_ceil,
        "Response Consistency": formatted_resp_consistency,
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"./experiment_results/experiment_diff_results_{timestamp}.xlsx"
    print(f"\n>>> Writing Excel file: {output_path}")
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for sheet_name, df in formatted.items():
            df.to_excel(writer, sheet_name=sheet_name)