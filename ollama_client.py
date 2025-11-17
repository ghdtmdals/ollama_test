from ollama import Client


import re




class ollama_client:
    def __init__(self):
        self.client = Client(host = 'http://ollama:11434')
    
    def chat(self, model_name, prompt, format):
        response = self.client.chat(
            model = model_name,
            messages = [{'role': 'user', 'content': prompt}],
            format = format,
        )

        return response.message.content

    ### 모델 리스트 확인
    def get_models(self):
        response = self.client.list()
        
        models = []
        for model in response.models:
            models.append(model.model)
        
        return sorted(models)
    
    def create_model(self, model_name, model_file):
        model_details = self.parse_modelfile(model_file)

        response = self.client.create(
            model = model_name,
            from_ = model_details['FROM'],
            system = model_details['SYSTEM'],
            parameters = model_details['PARAMETER'],
        )

        print("Create", model_name, response)

    ### 동일한 이름으로 생성하면 자동으로 Overwrite
    def update_model(self, model_name, model_file):
        self.create_model(model_name, model_file)

    def delete_model(self, model_name):
        self.client.delete(model_name)

    ### Modelfile로부터 모델 정보 추출
    def parse_modelfile(self, path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        result = {}

        # 1) FROM <value>
        from_match = re.search(r'^FROM\s+(.+)$', content, re.MULTILINE)
        if from_match:
            result["FROM"] = from_match.group(1).strip()

        # 2) SYSTEM """ ... """
        system_match = re.search(
            r'SYSTEM\s+"""(.*?)"""',
            content,
            re.DOTALL
        )
        if system_match:
            result["SYSTEM"] = system_match.group(1).strip()

        # 3) PARAMETER key value
        param_matches = re.findall(
            r'^PARAMETER\s+(\S+)\s+(.+)$',
            content,
            re.MULTILINE
        )
        if param_matches:
            # 여러 PARAMETER가 있을 수 있으므로 dict로 처리
            for key, value in param_matches:
                # 숫자형으로 변환 시도
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass  # 변환 실패 시 원래 문자열 유지

                result['PARAMETER'] = {key: value}

        return result


### 테스트 코드
# if __name__ == "__main__":
#     client = ollama_client()
#     client.create_model("create_test:v1", "./ollama-persona/test_Modelfile")
#     print(client.get_models())
    
#     response = client.chat("create_test:v1", "Hello, who are you?")
#     print(response)

#     client.delete_model("create_test:v1")
#     print(client.get_models())
