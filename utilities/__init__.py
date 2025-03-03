import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


def load_jsonl(file):
    """load a jsonl file and return a list of json objects"""
    with open(f'{file}', 'r') as json_file:
        json_list = list(json_file)
    return json_list


def load_theseus_files(filepath, subfolder, exp_type):
    json_list_main = []
    for i in range(3):
        json_file = (filepath + f"{subfolder}{i}{exp_type}.jsonl")
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
                      columns=['discussant', 'discussant_llm', "discussant_opinion", 'opponent',
                               'opponent_llm', "opponent_opinion",
                               'opinion_variation_discussant', 'opinion_variation_opponent',
                               'opponent_statement', 'discussant_answer', 'opponent_answer'])

    return df


def load_model(model, tokenizer, device):
    print(device)
    """load a model and tokenizer from huggingface"""
    model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
    model.config.max_length = 800
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, top_k=None, device=device,
                          max_length=512, truncation=True)
    return model, tokenizer, classifier


def predict(tokenizer, model, text_data, device):
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


def remove_task_sentences(text):
    text = str(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered_sentences = [sentence for sentence in sentences if 'task:' not in sentence.lower()]
    filtered_text = ' '.join(filtered_sentences)
    return filtered_text


def remove_constraint(text):
    text = str(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered_sentences = [sentence for sentence in sentences if 'constraints:' not in sentence.lower()]
    filtered_text = ' '.join(filtered_sentences)
    return filtered_text


def remove_starting_opinion(text):
    keywords = ["After reading your argument my conclusions are:", "My original opinion was"]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered_sentences = [sentence for sentence in sentences if not any(keyword in sentence for keyword in keywords)]
    filtered_text = ' '.join(filtered_sentences)
    return filtered_text


def remove_opinions(text):
    keywords = ["<ACCEPT|REJECT|IGNORE>", "END", "[INST]"]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered_sentences = [sentence for sentence in sentences if not any(keyword in sentence for keyword in keywords)]
    filtered_text = ' '.join(filtered_sentences)
    return filtered_text


def remove_other_constraints(text):
    text = str(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered_sentences = [sentence for sentence in sentences if 'Write your response' not in sentence]
    filtered_text = ' '.join(filtered_sentences)
    return filtered_text


def cleaning(text):
    text = text.replace('\n', ' ')
    text = text.replace('**', '')
    text = text.replace('\r', '')
    text = text.replace('\t', ' ')
    text = remove_constraint(text)
    text = remove_task_sentences(text)
    text = remove_opinions(text)
    text = remove_other_constraints(text)
    text = remove_starting_opinion(text)
    return text
