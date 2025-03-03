import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_jsonl(file):
    """load a jsonl file and return a list of json objects"""
    with open(f'{file}', 'r') as json_file:
        json_list = list(json_file)
    return json_list


def preprocess_theseus_files(file):
    return


def load_theseus_files(filepath, subfolder):
    json_list_main = []
    for i in range(3):
        json_file = (filepath + f"{subfolder}{i}.jsonl")
        json_list = load_jsonl(json_file)
        json_list_main.extend(json_list)
    text_lists = []
    for json_str in json_list_main[1:]:
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            continue
        if "interacting_agents" not in result.keys():
            continue
        else:
            text_lists.append((result["interacting_agents"]["discussant"],
                               result["interacting_agents"]["discussant_llm"],
                               result["interacting_agents"]["discussant_opinion"],
                               result["interacting_agents"]["opponent"],
                               result["interacting_agents"]["opponent_llm"],
                               result["interacting_agents"]["opponent_opinion"],
                               result["opinion_variation_discussant"],
                               result["opinion_variation_opponent"],
                               result["opponent_statement"],
                               result["discussant_answer"], result["opponent_answer"]))
    df = pd.DataFrame(text_lists,
                      columns=['discussant', 'discussant_llm', "discussant_numeric_opinion", 'opponent',
                               'opponent_llm', "opponent_numeric_opinion",
                               'opinion_variation_discussant', 'opinion_variation_opponent',
                               'opponent_statement', 'discussant_answer', 'opponent_answer'])

    return df


def load_model(model, tokenizer):
    """load a model and tokenizer from huggingface"""
    model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, top_k=None)
    return classifier


def predict(tokenizer, model, text_data):
    output = []
    for text in text_data:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = model(**inputs)
            scores = logits[0][0]
            scores = torch.nn.Softmax(dim=0)(scores)
            _, ranking = torch.topk(scores, k=scores.shape[0])
            ranking = ranking.tolist()
        output_single = [f"{i + 1}) {model.config.id2label[ranking[i]]} {scores[ranking[i]]:.4f}" for i in
                         range(scores.shape[0])][0]
        output.append(output_single)
    return output
