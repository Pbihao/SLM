import os
import json
from evaluate import load
import re
from tqdm import tqdm
import argparse

def accuracy_score(label, predict):
    # Calculate accuracy for multiple-choices questions and classification
    # For generation model, we only select the first word to compare
    label = re.sub(r'[^\w\s]', ' ', label)
    predict = re.sub(r'[^\w\s]', ' ', predict)
    assert len(label.split()) == 1, "label can only contain one word"
    if len(predict.split()) == 0:
        return 0
    label = label.split()[0]
    predict = predict.split()[0]
    return int(label == predict)


def score(result_path, task):
    with open(result_path) as f:
        results = json.load(f)
    
    score_path = os.path.join(os.path.split(result_path)[0], "score.json")
    if os.path.exists(score_path):
        raise ValueError("There is already score file")

    if task in ['history', 'finance']:
        score = 0
        for result in tqdm(results):
            label = result['label']
            output = result['output']
            score += accuracy_score(label, output)
        score = score / len(results)    
        score = dict(Accuracy=score)
    elif task in ['medical']:
        bertscore = load("bertscore")
        outputs = []
        labels = []
        for result in results:
            labels.append(result['label'])
            outputs.append(result['output'])
        score = bertscore.compute(predictions=outputs, references=labels, lang='en')
        for k in ['precision', 'recall', 'f1']:
            score["avg_{}".format(k)] = sum(score[k]) / len(results)
            score.pop(k)
    else:
        raise ValueError("No matched task")
    
    with open(score_path, 'w') as f:
        json.dump(score, f, indent=2)

    return score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="finance", type=str)
    parser.add_argument('--result_path', default=None)
    args = parser.parse_args()

    assert args.result_path is not None, "result_path should not be None"
    score(result_path=args.result_path, task=args.task)