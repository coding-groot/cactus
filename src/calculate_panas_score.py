import argparse
import json
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    return args


def load_data(input_path):
    with open(input_path, 'r') as json_file:
        data = json.load(json_file)

    return data


def calculate_score(data):
    criteria_list = [
        "Interested", "Excited", "Strong", "Enthusiastic", "Proud",
        "Alert", "Inspired", "Determined", "Attentive", "Active",
        "Distressed", "Upset", "Guilty", "Scared", "Hostile",
        "Irritable", "Ashamed", "Nervous", "Jittery", "Afraid"
    ]
    score_dict = {}

    for cri in criteria_list:
        score_dict[cri] = []

    score_per_attitude = {
        'positive': deepcopy(score_dict),
        'neutral': deepcopy(score_dict),
        'negative': deepcopy(score_dict)
    }

    for i in range(len(data)):
        score_lines = data[i]['prediction'].split('\n\n')
        for line in score_lines:
            criteria = line.split(', ')[0]
            score = int(line.split(', ')[-1])
            score_per_attitude[data[i]['attitude']][criteria].append(score)

    avg_score_per_attitude = {}
    for att in score_per_attitude.keys():
        avg_score_dict = {}
        for key in score_per_attitude[att].keys():
            avg_score_dict[key] = sum(score_per_attitude[att][key]) / len(score_per_attitude[att][key])

        positive_score = []
        for key in criteria_list[:10]:
            positive_score.append(avg_score_dict[key])

        negative_score = []
        for key in criteria_list[10:]:
            negative_score.append(avg_score_dict[key])

        avg_score_dict['positive_criteria'] = sum(positive_score) / len(positive_score)
        avg_score_dict['negative_criteria'] = sum(negative_score) / len(negative_score)

        avg_score_per_attitude[att] = avg_score_dict

    return avg_score_per_attitude


def save_file(args, avg_score_dict):
    with open(args.save_path, 'w') as json_file:
        json.dump(avg_score_dict, json_file)
    print("Saved the file to", args.save_path)


if __name__ == "__main__":
    args = parse_args()
    data = load_data(args.input_path)
    avg_score_dict = calculate_score(data)

    save_file(args, avg_score_dict)
