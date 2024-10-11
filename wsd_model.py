import torch
import pickle
import argparse
import os
import re


def train_time(train_data, bert_model, batch_size, hidden_layers_number):

    "{'word': word, 'word_id': word_id, 'sense_id': sense_id, 'examples': [[train_examples, dynasty],...]}"
    dynastys_dict = {'1':'3','2':'2','3':'1'} # Only used when evaluating 3 historical periods

    train_embeddings = {}
    for data in train_data:
        target_word = data['word'][0]

        word_id = data['word_id']
        sense_id = data['sense_id']
        examples = data['examples']
        context_vectors = []
        processed_examples = [example[0].replace('[', '').replace(']', '').replace(' ', '') for example in examples]
        dynastys = [dynastys_dict[example[1]] for example in examples]
        for i in range(0, len(examples), batch_size):
            batch_examples = processed_examples[i:i+batch_size]
            batch_dynastys = dynastys[i:i+batch_size]
            with torch.no_grad():
                target_embeddings = bert_model.embed_word(
                        batch_examples,
                        target_word,
                        time=batch_dynastys,
                        batch_size=batch_size,
                        hidden_layers_number=hidden_layers_number,
                    )
            torch.cuda.empty_cache()
            context_vectors.append(target_embeddings)

        context_vectors = torch.cat(context_vectors, dim=0)
        if context_vectors.ndim != 1:
            context_vectors = torch.mean(context_vectors, dim=0)
        if word_id not in train_embeddings:
            train_embeddings[word_id] = {}
        train_embeddings[word_id][sense_id] = context_vectors
        
    return train_embeddings
    

def evaluate_time(test_data, bert_model, train_embeddings, batch_size, hidden_layers_number):
    true_predict = 0
    all_examples = 0
    dynastys_dict = {'1':'3','2':'2','3':'1'} # Only used when evaluating 3 historical periods
    predict_test_data = []
    for data in test_data:
        
        target_word = data['word'][0]

        word_id = data['word_id']
        true_sense_id = data['sense_id']
        examples =  [example for example in data['examples'] if target_word in example[0][:128]]

        processed_examples = [example[0].replace('[', '').replace(']', '').replace(' ', '') for example in examples]
        dynastys = [dynastys_dict[example[1]] for example in examples]
        predicted_senses = []
        for i in range(0, len(examples), batch_size):
            batch_examples = processed_examples[i:i+batch_size]
            batch_dynastys = dynastys[i:i+batch_size]
            with torch.no_grad():
                word_embs = bert_model.embed_word(
                        batch_examples,
                        target_word,
                        time=batch_dynastys,
                        batch_size=batch_size,
                        hidden_layers_number=hidden_layers_number,
                    )
            if word_embs.ndim == 1:
                word_embs = torch.unsqueeze(word_embs, 0)
            torch.cuda.empty_cache()

            for word_emb in word_embs:
                all_examples += 1
                max_similarity = None
                predicted_sense = None

                for sense, embedding in train_embeddings[word_id].items():
                    similarity = torch.cosine_similarity(word_emb, embedding, dim=0)
                    if max_similarity is None or similarity > max_similarity:
                        max_similarity = similarity
                        predicted_sense = sense
                predicted_senses.append(predicted_sense)

                if predicted_sense == true_sense_id:
                    true_predict += 1
        assert len(predicted_senses) == len(examples)
        predict_test_data.append({'word': target_word, 'word_id': word_id, 'true_sense_id': true_sense_id, 'examples': [example + [predicted_sense] for example, predicted_sense in zip(examples, predicted_senses)]})

    return true_predict/all_examples, predict_test_data

def run(train_file, test_file, tokenizer, bert_model, device, hidden_layers_number):

    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)

    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)

    batch_size = 128

    train_embeddings = train_time(train_data, bert_model, batch_size, hidden_layers_number)
    print('train_done')
    acc_results, predict_test_data = evaluate_time(test_data, bert_model, train_embeddings, batch_size, hidden_layers_number)
    print('eval_done')

    return acc_results, predict_test_data


def str_to_bool(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Invalid value for --time. Valid values are 'True' or 'False'.")


def save_results(train_file, model_name, acc_results, predict_test_data):
    folder_path = f"eval_3/{model_name.split('/')[-1]}"
    if "checkpoint" in model_name:
        predict_path = f"predict_3/{model_name.split('/')[-2]}_{model_name.split('/')[-1].split('-')[-1]}"
    else:
        predict_path = f"predict_3/{model_name.split('/')[-1]}"

    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(predict_path, exist_ok=True)
    match = re.search(r'\d+', train_file[-10:])
    number = match.group()

    filename = f"{folder_path}/eval_data_{number}.txt"
    print(acc_results)
    with open(filename, "w") as file:
        file.write(str(acc_results))
    with open(f"{predict_path}/eval_data_{number}_predict.txt", "w") as file2:
        for i in predict_test_data:
            file2.write(str(i))
            file2.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, help="Path to train file")
    parser.add_argument("--test_file", type=str, help="Path to test file")
    parser.add_argument("--model_name", type=str, help="Path to model name")
    parser.add_argument("--hidden_layers_number", type=int, help="hidden_layers_number")
    args = parser.parse_args()

    train_file = args.train_file
    test_file = args.test_file
    model_name = args.model_name
    hidden_layers_number = args.hidden_layers_number

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import test_bert
    tester = test_bert.Tester(model_name, device=0)
    tokenizer = None
    for model in tester.bert_models:
        bert_model = model


    acc_results, predict_test_data = run(train_file, test_file, tokenizer, bert_model, device, hidden_layers_number)


    save_results(train_file, model_name, acc_results, predict_test_data)



