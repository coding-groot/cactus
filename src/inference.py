import argparse
import json
import multiprocessing
import re
import traceback
from abc import ABC, abstractmethod
from pathlib import Path

import requests
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI

from utils.config import get_config


class LLM:
    @abstractmethod
    def generate(self, prompt: str):
        pass


class ChatGPT(LLM):
    def __init__(self):
        config = get_config()
        api_key = config['openai']['key']
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            max_tokens=512,
            openai_api_key=api_key,
        )

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content


class LLama2(LLM):
    def __init__(self):
        self.config = get_config()
        self.model_url = f"{self.config['llama2']['host']}/models"
        self.host = self.config['llama2']['host']
        self.llm = OpenAI(
            temperature=0.7,
            openai_api_key='EMPTY',
            openai_api_base=self.host,
            model=self.get_model_name(),
            max_tokens=512,
        )

    def get_model_name(self):
        response = requests.get(self.model_url)
        response = response.json()
        return response["data"][0]["id"]

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt)


class LLama3(LLama2):
    def __init__(self):
        super().__init__()
        self.model_url = f"{self.config['llama3']['host']}/models"
        self.host = self.config['llama3']['host']
        self.llm = OpenAI(
            temperature=0.7,
            openai_api_key='EMPTY',
            openai_api_base=self.host,
            max_tokens=512,
            model=self.get_model_name()
        )


class Agent(ABC):
    def __init__(self, llm_type):
        self.llm = LLMFactory.get_llm(llm_type)
        self.prompt_template = None

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    def load_prompt(self, file_name):
        base_dir = Path(__file__).resolve().parents[1] / "prompts"
        file_path = base_dir / file_name
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()


class ClientAgent(Agent):
    def __init__(self, example):
        super().__init__('chatgpt')
        self.example = example
        prompt_text = self.load_prompt(f"agent_client.txt")
        self.attitude = (
            f"{self.example['AI_client']['attitude']}: "
            f"{self.example['AI_client']['attitude_instruction']}")
        self.prompt_template = PromptTemplate(
            input_variables=["intake_form", "attitude", "history"],
            template=prompt_text)

    def generate(self, history):
        history_text = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['message']}"
                for message in history
            ]
        )
        prompt = self.prompt_template.format(
            intake_form=self.example,
            attitude=self.attitude,
            history=history_text
        )

        return self.llm.generate(prompt)


class CBTAgent(Agent):
    def __init__(self, llm_type, example):
        super().__init__(llm_type)
        self.example = example
        self.pattern = r"CBT technique:\s*(.*?)\s*Counseling plan:\s*(.*)"
        prompt_text = self.load_prompt(f"agent_cbt_{llm_type}.txt")
        self.prompt_template = PromptTemplate(
            input_variables=[
                "client_information",
                "reason_counseling",
                'history',
            ],
            template=prompt_text)

    def generate(self, history):
        prompt = self.prompt_template.format(
            client_information=self.example['AI_counselor']['CBT'][
                'client_information'],
            reason_counseling=self.example['AI_counselor']['CBT'][
                'reason_counseling'],
            history="Client: " + history
        )
        response = self.llm.generate(prompt)

        try:
            cbt_technique = response.split("Counseling")[0].replace("\n", "")
        except Exception as e:
            cbt_technique = None
            print(e)

        try:
            cbt_plan = response.split("Counseling")[1].split(":\n")[1]
        except Exception as e:
            cbt_plan = None
            print(e)

        if cbt_plan:
            return cbt_technique, cbt_plan
        else:
            error_file_path = Path(
                f"./invalid_response_{self.example[:10]}.txt")
            with open(error_file_path, "w", encoding="utf-8") as f:
                f.write(response)
            raise ValueError("Invalid response format from LLM")

    def extract_cbt_details(self, response):
        match = re.search(self.pattern, response, re.DOTALL | re.IGNORECASE)

        if not match:
            return None, None

        cbt_technique = match.group(1).strip()
        cbt_plan = match.group(2).strip()
        return cbt_technique, cbt_plan


class CounselorAgent(Agent):
    def __init__(self, llm_type):
        super().__init__(llm_type)
        prompt_text = self.load_prompt(f"agent_cactus_{llm_type}.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["history"],
            template=prompt_text)

    def generate(self, history):
        history = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['message']}"
                for message in history
            ]
        )
        prompt = self.prompt_template.format(history=history)
        return self.llm.generate(prompt)


class CactusCounselorAgent(CounselorAgent):
    def __init__(self, example, llm_type):
        super().__init__(llm_type)
        self.example = example
        self.cbt_technique = None
        self.cbt_plan = None
        self.llm_type = llm_type
        prompt_text = self.load_prompt(f"agent_cactus_{llm_type}.txt")
        self.prompt_template = PromptTemplate(
            input_variables=[
                "client_information",
                "reason_counseling",
                "cbt_plan",
                "history"
            ],
            template=prompt_text)

    def set_cbt(self, history):
        cbt_agent = CBTAgent(self.llm_type, self.example)
        self.cbt_technique, self.cbt_plan = cbt_agent.generate(history)

    def generate(self, history):
        history_text = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['message']}"
                for message in history
            ]
        )
        prompt = self.prompt_template.format(
            client_information=self.example['AI_counselor']['CBT'][
                'client_information'],
            reason_counseling=self.example['AI_counselor']['CBT'][
                'reason_counseling'],
            cbt_plan=self.cbt_plan,
            history=history_text,
        )

        response = self.llm.generate(prompt)

        if "'message':" in response:
            response = self.clean_message(response)

        response = self.extract_counselor_message(response)
        return response.strip()

    def clean_message(self, response):
        response = response.split("'message':")[1]
        response = response.split(", {")[0]
        response = response.replace("\"", "")
        response = response.replace("]", "")
        response = response.replace("}", "")
        return response

    def extract_counselor_message(self, response):
        response = response.split("Counselor:")[-1]
        response = response.replace("\n", "")
        response = response.replace("\\", "")
        response = response.replace("\"", "")
        return response


class Psych8kCounselorAgent(CounselorAgent):
    def __init__(self, llm_type):
        super().__init__(llm_type)
        prompt_text = self.load_prompt(f"agent_psych8k_{llm_type}.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["history"],
            template=prompt_text)

    def generate(self, history):
        history = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['message']}"
                for message in history
            ]
        )
        prompt = self.prompt_template.format(history=history)
        response = self.llm.generate(prompt)
        response = response.replace('Output:', '')
        response = response.replace('Counselor:', '')
        response = response.strip()
        return response


class SmileCounselorAgent(CounselorAgent):
    def __init__(self, llm_type):
        super().__init__(llm_type)
        prompt_text = self.load_prompt(f"agent_smile_{llm_type}.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["history"],
            template=prompt_text)


class LLMFactory:
    @staticmethod
    def get_llm(llm_type):
        if llm_type == "chatgpt":
            return ChatGPT()
        elif llm_type == "llama2":
            return LLama2()
        elif llm_type == "llama3":
            return LLama3()
        raise ValueError(f"Unsupported LLM type: {llm_type}")


class TherapySession:
    def __init__(self, example, counselor_type, counselor_llm_type, max_turns):
        self.counselor_type = counselor_type
        self.example = example
        self.client_agent = ClientAgent(example=example)
        self.counselor_agent = self._create_counselor_agent(
            counselor_type,
            counselor_llm_type)
        self.history = []
        self.max_turns = max_turns

    def _create_counselor_agent(self, counselor_type, llm_type):
        if counselor_type == "cactus":
            return CactusCounselorAgent(self.example, llm_type)
        elif counselor_type == "psych8k":
            return Psych8kCounselorAgent(llm_type)
        elif counselor_type == "smile":
            return SmileCounselorAgent(llm_type)
        else:
            raise ValueError(f"Unsupported counselor type: {counselor_type}")

    def _add_to_history(self, role, message):
        self.history.append({"role": role, "message": message})

    def _initialize_session(self):
        example_cbt = self.example['AI_counselor']['CBT']
        self._add_to_history("counselor",
                             example_cbt['init_history_counselor'])
        self._add_to_history("client", example_cbt['init_history_client'])
        if self.counselor_type == 'cactus':
            self.counselor_agent.set_cbt(example_cbt['init_history_client'])

    def _exchange_statements(self):

        for turn in range(self.max_turns):
            counselor_statement = self.counselor_agent.generate(self.history)
            counselor_statement = counselor_statement.replace('Counselor: ',
                                                              '')
            self._add_to_history("counselor", counselor_statement)

            client_statement = self.client_agent.generate(self.history)
            client_statement = client_statement.replace('Client: ', '')

            self._add_to_history("client", client_statement)

            if '[/END]' in client_statement:
                self.history[-1]['message'] = self.history[-1][
                    'message'].replace('[/END]', '')
                break

    def run_session(self):
        self._initialize_session()
        self._exchange_statements()
        return {
            "example": self.example,
            "cbt_technique": getattr(
                self.counselor_agent,
                'cbt_technique',
                None
            ),
            "cbt_plan": getattr(self.counselor_agent, 'cbt_plan', None),
            "history": self.history
        }


def run_therapy_session(index, example, output_dir,
                        counselor_type, llm_type, total, max_turns):
    output_dir = Path(output_dir)
    file_number = index + 1

    try:
        print(f"Generating example {file_number} out of {total}")

        therapy_session = TherapySession(
            example,
            counselor_type,
            llm_type,
            max_turns,
        )
        session_data = therapy_session.run_session()

        file_name = f"session_{file_number}.json"
        file_path = output_dir / file_name

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        error_file_name = f"error_{file_number}.txt"
        error_file_path = output_dir / error_file_name
        with open(error_file_path, "w", encoding="utf-8") as f:
            f.write("".join(traceback.format_exception(e)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run therapy sessions in parallel.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the JSON file containing client intake forms.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save the session results.")
    parser.add_argument("--num_processes", type=int, default=None,
                        help="Number of processes to use in the pool."
                             " Defaults to the number of CPU cores "
                             "if not specified.")
    parser.add_argument("--counselor_type", type=str, required=True,
                        choices=["cactus", "psych8k", "smile"],
                        help="Type of counselor to use.")
    parser.add_argument("--llm_type", type=str, required=True,
                        choices=["chatgpt", "llama2", "llama3"],
                        help="Type of LLM to use.")
    parser.add_argument("--max_turns", type=int, default=20,
                        help="Maximum number of turns for the session.")

    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(data)
    args_list = [(index, example, output_dir, args.counselor_type,
                  args.llm_type, total, args.max_turns)
                 for index, example in enumerate(data)]

    with multiprocessing.Pool(processes=args.num_processes) as pool:
        for i, _ in enumerate(pool.starmap(run_therapy_session, args_list)):
            print(f"Generating example {i} out of {total}")
