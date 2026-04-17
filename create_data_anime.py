from datasets import load_dataset
import random
from simple_chat import load_model, chat
import gc, torch

def build_chat_text(example, tokenizer):
    messages = [
        {
            "role": "system",
            "content": f"You are an anime character with the following personality: {example['trait']}."
        },
        {
            "role": "user",
            "content": example["question"]
        },
        {
            "role": "assistant",
            "content": example["dialogue"]
        },
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    return {"text": text}

if __name__ == "__main__":
    ds = load_dataset("parquet", data_files="./anime_dataset/data/train-00000-of-00001.parquet")
    # print(ds["train"][0])
    # print(len(ds["train"]))
    # print(ds.keys())
    # print(ds["train"].unique("trait")) # 'tsundere'傲娇, 'yandere'病娇, 'himedere'公主骄, 'genki'元气型, 'moe'萌系, 'bakadere'笨蛋骄

    if "model" not in globals():
        model, tokenizer = load_model()

    ds_map = ds.map(
        lambda example: build_chat_text(example, tokenizer), 
        remove_columns=["trait", "dialogue", "question"])
    ds_map.save_to_disk("anime_map")

    # print(ds_map) # 如果不remove_columns，会保留原来的列
        # DatasetDict({
        #     train: Dataset({
        #         features: ['trait', 'dialogue', 'question', 'text'],
        #         num_rows: 744
        #     })
        # })
    print(ds_map["train"][0])

    # # 清理内存。
    # 其他措施：小样本、存下ds_map到本地、不加载模型（多次加载会很占内存）、if "model" not in globals()
    del model, ds_map
    gc.collect()
    torch.cuda.empty_cache()
