<h1 align="center"> Cactus: Towards Psychological Counseling Conversations using Cognitive Behavioral Theory </h1>
<p align="center">
  <img src="https://github.com/coding-groot/cactus/assets/81813324/bd7cdce1-e85a-4797-8d0e-636a7ea92c41" width="400" height="350"/>
</p>
This is the official GitHub repository for [Cactus: Towards Psychological Counseling Conversations using Cognitive Behavioral Theory](https://arxiv.org/abs/2407.03103).

# Citation
```
@misc{lee2024cactuspsychologicalcounselingconversations,
      title={Cactus: Towards Psychological Counseling Conversations using Cognitive Behavioral Theory}, 
      author={Suyeon Lee and Sunghwan Kim and Minju Kim and Dongjin Kang and Dongil Yang and Harim Kim and Minseok Kang and Dayi Jung and Min Hee Kim and Seungbeen Lee and Kyoung-Mee Chung and Youngjae Yu and Dongha Lee and Jinyoung Yeo},
      year={2024},
      eprint={2407.03103},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.03103}, 
}
```

# Link
Our dataset & model are available [here](https://huggingface.co/collections/DLI-Lab/cactus-towards-psychological-counseling-conversations-6672312f6f64b0d7be75dd0b).

# CACTUS Inference README

## Setup

### 1. Virtual Environment Setup

We recommend you create a virtual environment using `conda` or `virtualenv`.

#### Using Conda

```sh
conda create -n therapy-session python=3.8
conda activate therapy-session
```

#### Using Virtualenv

```sh
# if virtualenv is not installed
pip install virtualenv

# Create a virtual environment
virtualenv .venv
source .venv/bin/activate  # Linux & macOS
.venv\Scripts\activate  # Windows
```

### 2. Installing Required Packages

After activating the virtual environment, install the necessary packages using the `requirements.txt` file.

```sh
pip install -r requirements.txt
```

### 3. Configuring the Settings File

Copy the `config.yaml.example` file in the conf.d folder to create a `config.yaml` file. Then, fill in the following content in the `config.yaml` file.

```yaml
openai:
  key: <<Your openai API key>>

llama2:
  host: http://<<Server IP or URL>>/v1

llama3:
  host: http://<<Server IP or URL>>/v1
```

## Adding a Counselor Agent

To add a counselor agent, follow these steps.

### 1. Creating a Prompt File

- The prompt file should be located in the `prompts` folder.
- The file name pattern should follow the format `agent_{counselor_type}_{llm_type}.txt`.
  Example: `agent_cactus_chatgpt.txt`
- The prompt file should include a template for generating the counselor's response.

  ```text
  Client information: {client_information}
  Reason for counseling: {reason_counseling}
  CBT plan: {cbt_plan}
  History: {history}
  ```

### 2. Adding a New Counselor Agent Class

Create a new counselor agent class by inheriting from the `CounselorAgent` class. Ensure to set `self.language` to either `english` for English or `chinese` for Chinese.

```python
class NewCounselorAgent(CounselorAgent):
    def __init__(self, llm_type):
        super().__init__(llm_type)
        self.language = "english"  # For English
        # self.language = "chinese"  # For Chinese
        prompt_text = self.load_prompt(f"agent_new_{llm_type}.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["history"],
            template=prompt_text)

    def generate(self, history):
        # Override the generate function if necessary
        history = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['message']}"
                for message in history
            ]
        )
        prompt = self.prompt_template.format(history=history)
        return self.llm.generate(prompt)
```

### 3. Adding the New Counselor to LLMFactory

Add the new counselor agent to the `LLMFactory` class.

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

## Adding a New LLM

To add a new LLM, follow these steps.

### 1. Creating a New LLM Class

Create a new LLM class by inheriting from the `LLM` abstract class.

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

### 2. Adding the New LLM to LLMFactory

Add the new LLM to the `LLMFactory` class.

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

## Running Counseling-Eval

### 1. Prepare necessary files and folders

- Ensure the necessary prompt files are available in the `prompts` folder.
- The input file should be a JSON file containing the client intake form.

### 2. Run the program

Run the program using the following command.

```sh
python script.py --input_file {path to input file} --output_dir {output directory} --counselor_type {counselor type} --llm_type {LLM type} --max_turns {maximum number of turns}
```

Example:

```sh
python script.py --input_file ./data/intake_forms.json --output_dir ./output --counselor_type cactus --llm_type chatgpt --max_turns 20
```

### 3. Using the Execution Script

You can use the 'scripts/inference.sh' script for easy execution. Run it as follows:

```sh
sh scripts/inference.sh
```

### 4. Running the vLLM Server

All models except `chatgpt` (such as `llama2`, `llama3`, etc.) need to run on the vLLM server. Refer to the `scripts/run_vllm.sh` script for this.

```sh
sh scripts/run_vllm.sh
```

This script includes all the commands necessary to set up and run the vLLM server. With the vLLM server running, you can simulate the counseling session using the program.
