import argparse
import asyncio
import json
import random

from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from utils.config import load_prompt

TOTAL_COST = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="gpt-3.5-turbo or gpt-4")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--prompt_name", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0,
                        help="If you want to start from a specific index, set this argument")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_sample", type=int, default=None,
                        help="If you want to test your code by sampling a small number of data, you can set this argument.")
    parser.add_argument("--num_shot", type=int, default=None)
    ## generate args ##
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--stop_sequence", type=str, nargs='+', default=None)
    parser.add_argument("--sampling_num", type=int, default=1, help="The number of samples to generate per instance")
    args = parser.parse_args()

    return args


def get_dialogue_history(dialogue_history_list):
    dialogue_history_tmp = []
    for item in dialogue_history_list:
        if item['role'] == 'counselor':
            text = 'Counselor: ' + item['message']
        else:
            text = 'Client: ' + item['message']
        dialogue_history_tmp.append(text)

    dialogue_history = '\n'.join(dialogue_history_tmp)

    return dialogue_history


def prepare_model_input(prompt: str, data_path: str):
    '''
        input : prompt, data_path (str)
        output : all_model_data (list of dict)
    '''

    with open(data_path, "r", encoding="UTF-8") as f:
        data = json.load(f)

    all_model_data = []
    for d in tqdm(data):
        input_temp = dict()
        input_temp["idx"] = d["idx"]

        input_temp['model_input'] = prompt.format(**{
            'conversation': get_dialogue_history(d["dialogue"])
        })

        all_model_data.append(input_temp)

    return all_model_data


def load_and_prepare_data(args):
    prompt = load_prompt(args.prompt_name)
    print("Preparing model inputs...")
    all_model_data = prepare_model_input(
        prompt, args.input_path)
    return all_model_data


def sample_indices(all_model_inputs, num_sample):
    random.seed(0)
    cand_indices = list(range(len(all_model_inputs)))
    sampled_indices = random.sample(cand_indices, num_sample)
    return sampled_indices


def filter_data(all_model_data, num_sample):
    if num_sample:
        sampled_indices = sample_indices(all_model_data, num_sample)
        all_model_data = [all_model_data[i] for i in sampled_indices]
    return all_model_data


async def async_generate(args, llm, model_data, idx):
    global TOTAL_COST
    human_message = HumanMessage(content=model_data['model_input'])
    while True:
        try:
            response = await llm.agenerate([[human_message]])
            token_used = response.llm_output['token_usage']['total_tokens']

            if args.model_name == "gpt-3.5-turbo":
                TOTAL_COST += token_used / 1000 * 0.002
            elif args.model_name == "gpt-4":
                TOTAL_COST += token_used / 1000 * 0.06
            elif args.model_name == "gpt-4o":
                TOTAL_COST += token_used / 1000 * 0.02
            print(idx, TOTAL_COST)
            break

        except Exception as e:
            print(f"Exception occurred: {e}")

    await asyncio.sleep(2)
    result = {
        "idx": model_data["idx"],
        "score": response.generations[0][0].text,
    }

    return result


async def generate_concurrently(args, all_model_data, start_idx):
    llm = ChatOpenAI(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=100,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        n=args.sampling_num,
    )
    tasks = [async_generate(args, llm, model_data, i + start_idx)
             for i, model_data in enumerate(all_model_data)]

    await asyncio.sleep(2)
    return await tqdm_asyncio.gather(*tasks)


async def main(args):
    all_model_data = load_and_prepare_data(args)

    if args.num_sample:
        all_model_data = all_model_data[:args.num_sample]

    all_results = []
    batch_num = 30
    if len(all_model_data) - args.start_idx > batch_num:
        for start_idx in tqdm(range(args.start_idx, len(all_model_data), batch_num)):
            cur_model_data = all_model_data[start_idx:start_idx + batch_num]
            all_results.extend(await generate_concurrently(args, cur_model_data, start_idx))
            await asyncio.sleep(2)
    else:
        all_results = await generate_concurrently(args, all_model_data, args.start_idx)

    with open(args.save_dir, "w", encoding='UTF-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
