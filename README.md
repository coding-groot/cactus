<h1 align="center"> Cactus: Towards Psychological Counseling Conversations using Cognitive Behavioral Theory </h1>

<img src="https://github.com/coding-groot/cactus/assets/81813324/bd7cdce1-e85a-4797-8d0e-636a7ea92c41" width="500" height="400"/>


# CACTUS Inference README

## 셋팅 방법

### 1. 가상환경 설정

가상환경을 사용하여 필요한 패키지를 격리하여 설치할 수 있습니다. `conda` 또는 `virtualenv`를 사용하여 가상환경을 설정하세요.

#### Conda 사용 시

```sh
conda create -n therapy-session python=3.8
conda activate therapy-session
```

#### Virtualenv 사용 시

```sh
# virtualenv가 설치되지 않은 경우
pip install virtualenv

# 가상환경 생성
virtualenv .venv
source .venv/bin/activate  # Linux 및 macOS
.venv\Scripts\activate  # Windows
```

### 2. 필요한 패키지 설치

가상환경을 활성화한 후, `requirements.txt` 파일을 사용하여 필요한 패키지를 설치합니다.

```sh
pip install -r requirements.txt
```

### 3. 설정 파일 구성

`conf.d` 폴더에 있는 `config.yaml.example` 파일을 복사하여 `config.yaml` 파일을 만듭니다. 그런 다음, `config.yaml` 파일에 다음 내용을 채웁니다.

```yaml
openai:
  key: <<Your openai API key>>

llama2:
  host: http://<<Server IP or URL>>/v1

llama3:
  host: http://<<Server IP or URL>>/v1
```

## Counselor 에이전트 추가 방법

Counselor 에이전트를 추가하려면 다음 단계를 따르세요.

### 1. 프롬프트 파일 만들기

- 프롬프트 파일은 `prompts` 폴더에 위치해야 합니다.
- 파일명 패턴은 `agent_{counselor_type}_{llm_type}.txt` 형식을 따라야 합니다.
  예: `agent_cactus_chatgpt.txt`
- 프롬프트 파일에는 상담사의 응답 생성을 위한 템플릿이 포함되어야 합니다.
  ```text
  Client information: {client_information}
  Reason for counseling: {reason_counseling}
  CBT plan: {cbt_plan}
  History: {history}
  ```

### 2. Counselor 에이전트 클래스 추가

`CounselorAgent` 클래스를 상속하여 새로운 Counselor 에이전트 클래스를 만듭니다. `self.language`를 반드시 설정해야 하며, 영어는 `english`, 중국어는 `chinese`로 설정합니다.

```python
class NewCounselorAgent(CounselorAgent):
    def __init__(self, llm_type):
        super().__init__(llm_type)
        self.language = "english"  # 예: 영어일 경우
        # self.language = "chinese"  # 예: 중국어일 경우
        prompt_text = self.load_prompt(f"agent_new_{llm_type}.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["history"],
            template=prompt_text)

    def generate(self, history):
        # 필요한 경우 generate 함수를 재정의합니다.
        history = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['message']}"
                for message in history
            ]
        )
        prompt = self.prompt_template.format(history=history)
        return self.llm.generate(prompt)
```

### 3. LLMFactory에 새로운 Counselor 추가

`LLMFactory` 클래스에 새로운 Counselor 에이전트를 추가합니다.

```python
class LLMFactory:
    @staticmethod
    def get_llm(llm_type):
        if llm_type == "chatgpt":
            return ChatGPT()
        elif llm_type == "llama2":
            return LLama2()
        elif llm_type == "llama3":
            return LLama3()
        elif llm_type == "new":
            return NewCounselorAgent(llm_type)
        raise ValueError(f"Unsupported LLM type: {llm_type}")
```

## LLM 추가 방법

새로운 LLM을 추가하려면 다음 단계를 따르세요.

### 1. 새로운 LLM 클래스 만들기

`LLM` 추상 클래스를 상속하여 새로운 LLM 클래스를 만듭니다.

```python
class NewLLM(LLM):
    def __init__(self):
        config = get_config()
        api_key = config['new']['key']
        self.llm = OpenAI(
            temperature=0.7,
            model_name="new-model",
            openai_api_key=api_key
        )

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content
```

### 2. LLMFactory에 새로운 LLM 추가

`LLMFactory` 클래스에 새로운 LLM을 추가합니다.

```python
class LLMFactory:
    @staticmethod
    def get_llm(llm_type):
        if llm_type == "chatgpt":
            return ChatGPT()
        elif llm_type == "llama2":
            return LLama2()
        elif llm_type == "llama3":
            return LLama3()
        elif llm_type == "new":
            return NewLLM()
        raise ValueError(f"Unsupported LLM type: {llm_type}")
```

## 실행 방법

### 1. 필수 파일 및 폴더 준비

- `prompts` 폴더에 필요한 프롬프트 파일을 준비합니다.
- 입력 파일은 클라이언트 intake form을 포함하는 JSON 파일이어야 합니다.

### 2. 프로그램 실행

다음 명령어를 사용하여 프로그램을 실행할 수 있습니다.

```sh
python script.py --input_file {입력 파일 경로} --output_dir {출력 디렉터리} --counselor_type {counselor 유형} --llm_type {LLM 유형} --max_turns {최대 턴 수}
```

예시:

```sh
python script.py --input_file ./data/intake_forms.json --output_dir ./output --counselor_type cactus --llm_type chatgpt --max_turns 20
```

### 3. 실행 스크립트 사용

간편한 실행을 위해 `scripts/inference.sh` 스크립트를 사용할 수 있습니다. 이 스크립트를 사용하려면 다음과 같이 실행합니다.

```sh
sh scripts/inference.sh
```

### 4. vLLM 서버 실행

`chatgpt`를 제외한 모든 모델(`llama2`, `llama3` 등)은 vLLM을 사용하여 모델을 실행해야 합니다. 이를 위해 `scripts/run_vllm.sh` 스크립트를 참고하세요.

```sh
sh scripts/run_vllm.sh
```

이 스크립트는 vLLM 서버를 설정하고 실행하는 데 필요한 모든 명령어를 포함하고 있습니다. vLLM 서버가 실행 중인 상태에서 프로그램을 실행하여 상담 세션을 시뮬레이션할 수 있습니다.

위 명령어를 사용하여 프로젝트를 간편하게 실행하고 결과를 확인할 수 있습니다.
