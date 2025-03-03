from utilities import *

opinion_map = {
    "0": "strongly disagree",
    "1": "disagree",
    "2": "mildly disagree",
    "3": "neutral",
    "4": "mildly agree",
    "5": "agree",
    "6": "fully agree"
}

filepath = "c:/Users/Erica Cau/Documents/GitHub/Theseus/data/"
output_path = "c:/Users/Erica Cau/Documents/GitHub/Theseus/output/"
device = "cuda:0" if torch.cuda.is_available() else "cpu"


folders = ["theseus_different_llama3_", "theseus_same_llama3_", "theseus_different_mistral_", "theseus_same_mistral_"]
exp_type = ["_unbalanced", "_polarized", ""]

for subfolder in folders:
    for exp in exp_type:
        df = load_theseus_files(filepath, subfolder, exp)
        df_opponent = df.copy()
        df_answers = df.copy()
        df_opponent_answer = df.copy()
        print(subfolder, exp)

        print(df_opponent.shape, df_answers.shape, df_opponent_answer.shape)

        df_opponent = df_opponent.drop_duplicates(subset=['opponent_statement'])
        df_answers = df_answers.drop_duplicates(subset=['discussant_answer'])
        df_opponent_answer = df_opponent_answer.drop_duplicates(subset=['opponent_answer'])
        print("annotation of opponent statement")
        text_data = df_opponent["opponent_statement"].to_list()
        model_name = "q3fer/distilbert-base-fallacy-classification"
        tokenizer_name = "q3fer/distilbert-base-fallacy-classification"
        model, tokenizer, classifier = load_model(model_name, tokenizer_name, device)
        list_of_fallacies = predict(tokenizer, model, text_data, device)
        df_opponent["fallacies"] = list_of_fallacies
        df_opponent.to_csv(f"../output/fallacies_unique_opponent_{subfolder}{exp}.csv", index=False)
        
        print("annotation of discussant")
        text_data = df_answers["discussant_answer"].to_list()
        model, tokenizer, classifier = load_model(model_name, tokenizer_name, device)
        list_of_fallacies = predict(tokenizer, model, text_data, device)
        df_answers["fallacies_answers"] = list_of_fallacies
        df_answers.to_csv(f"../output/fallacies_unique_discussant_{subfolder}{exp}.csv", index=False)
        
        print("annotation of opponent answers")
        text_data = df_opponent_answer["opponent_answer"].to_list()
        model, tokenizer, classifier = load_model(model_name, tokenizer_name, device)
        list_of_fallacies = predict(tokenizer, model, text_data, device)
        df_opponent_answer["fallacies_opponent_answer"] = list_of_fallacies
        df_opponent_answer.to_csv(f"../output/fallacies_unique_opponent_answer_{subfolder}{exp}.csv", index=False)

