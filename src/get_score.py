import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_foler", type=str)
    parser.add_argument('--criteria_list', type=str, nargs='+', help='List of criteria')
    args = parser.parse_args()

    return args


def get_scoring(args):
    criteria_list = args.criteria_list[0].split(" ")

    total_results = {}
    for criteria in criteria_list:
        with open(f"{args.result_foler}/score_{criteria}.json", "r") as f:
            dataset = json.load(f)

        avg_score = 0
        for data in dataset:
            try:
                score = int(data["score"].split(",")[0])
            except Exception as e:
                print(e)
                score = 0
            avg_score += score

        total_results[criteria] = avg_score / len(dataset)

    for criteria in criteria_list:
        print(f"{criteria} : {round(total_results[criteria], 2)}")


if __name__ == "__main__":
    args = parse_args()
    get_scoring(args)
