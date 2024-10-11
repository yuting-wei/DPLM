import multiprocessing
import os
import argparse

def run_process(train_file, test_file, model_name, hidden_layers_number):
    print(train_file)
    command = f"python wsd_model.py --train_file {train_file} --test_file {test_file} --model_name {model_name}  --hidden_layers_number {hidden_layers_number}"
    os.system(command)

def sort_key(file_name):
    file_num = int(file_name.split("_")[-1].split(".")[0])
    if file_num == 10:
        return float('inf')
    else:
        return file_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Path to model name")
    args = parser.parse_args()
    model_name = args.model_name    

    num_processes = 3

    train_files = []
    test_files = []
    for i in range(2, 3):
        train_files.append(f"eval_wsd_datasets/ccl_3/train_data_{i}.pkl")
        test_files.append(f"eval_wsd_datasets/ccl_3/test_data_{i}.pkl")    

    hidden_layers_number = 1
    print(model_name)
    pool = multiprocessing.Pool(processes=num_processes)
    for train_file, test_file in zip(train_files, test_files):
        pool.apply_async(run_process, args=(train_file, test_file, model_name, hidden_layers_number))
    pool.close()
    pool.join()

    import os

    folder_path = f"results/eval_3/{model_name.split('/')[-1]}"
    summary_file = f"results/eval_3/{model_name.split('/')[-2][-7:]}_{model_name.split('/')[-1]}_all.txt"

    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")], key=sort_key)

    with open(summary_file, "w") as f:
        for txt_file in txt_files:
            file_path = os.path.join(folder_path, txt_file)
            with open(file_path, "r") as txt:
                content = txt.read().strip()
                file_name = os.path.splitext(txt_file)[0]
                f.write(f"{content}\n")




