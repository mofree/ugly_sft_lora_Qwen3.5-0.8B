import random
import json
import time
from accelerate import data_loader
import pandas as pd
from datasets import Dataset, load_dataset
from simple_chat import load_model
import gc, torch

# def generate_xlsx():
#     """第1步：从 city.txt 随机生成问答，输出 xlsx"""
#     Q_list = [
#         ('{city}{year}年{month}月{day}日的天气','%Y-%m-%d'),
#         ('{city}{year}年{month}月{day}号的天气','%Y-%m-%d'),
#         ('{city}{month}月{day}日的天气','%m-%d'),
#         ('{city}{month}月{day}号的天气','%m-%d'),

#         ('{year}年{month}月{day}日{city}的天气','%Y-%m-%d'),
#         ('{year}年{month}月{day}号{city}的天气','%Y-%m-%d'),
#         ('{month}月{day}日{city}的天气','%m-%d'),
#         ('{month}月{day}号{city}的天气','%m-%d'),

#         ('你们{year}年{month}月{day}日去{city}玩吗？','%Y-%m-%d'),
#         ('你们{year}年{month}月{day}号去{city}玩么？','%Y-%m-%d'),
#         ('你们{month}月{day}日去{city}玩吗？','%m-%d'),
#         ('你们{month}月{day}号去{city}玩吗？','%m-%d'),
#     ]

#     prompt_template = '''
# 给定一句话："%s"，请你按步骤要求工作。

# 步骤1：识别这句话中的城市和日期共2个信息
# 步骤2：根据城市和日期信息，生成JSON字符串，格式为{"city":城市,"date":日期}

# 请问，这个JSON字符串是：
# '''

#     with open('./city_date_dataset/city.txt', 'r', encoding='utf-8') as fp:
#         city_list = fp.readlines()
#         city_list = [line.strip().split(' ')[1] for line in city_list]

#     Q_arr = []
#     A_arr = []
#     # 生成一批"1月2号"、"1月2日"、"2023年1月2号", "2023年1月2日", "2023-02-02", "03-02"之类的话术, 教会它做日期转换
#     for i in range(1000):
#         Q = Q_list[random.randint(0, len(Q_list) - 1)]
#         city = city_list[random.randint(0, len(city_list) - 1)]
#         year = random.randint(1990, 2025)
#         month = random.randint(1, 12)
#         day = random.randint(1, 28)
#         time_str = '{}-{}-{}'.format(year, month, day)
#         date_field = time.strftime(Q[1], time.strptime(time_str, '%Y-%m-%d'))
#         Q = Q[0].format(city=city, year=year, month=month, day=day)  # 问题
#         A = json.dumps({'city': city, 'date': date_field}, ensure_ascii=False)  # 回答

#         Q_arr.append(prompt_template % (Q,))
#         A_arr.append(A)

#     df = pd.DataFrame({'Prompt': Q_arr, 'Completion': A_arr})
#     df.to_excel('./city_date_dataset/train.xlsx', index=False)
#     print(f"xlsx 已生成，共 {len(df)} 条")

def build_chat_text(example, tokenizer):
    """将每条数据套上 chat template，变成模型可训练的 text"""
    messages = [
        {"role": "user", "content": example["Prompt"]},
        {"role": "assistant", "content": example["Completion"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

# def convert_xlsx_to_dataset():
#     """第2步：xlsx → chat template → HuggingFace Dataset"""
#     # 读取 xlsx
#     df = pd.read_excel('./city_date_dataset/train.xlsx')
#     ds = Dataset.from_pandas(df)

#     # 加载 tokenizer
#     _, tokenizer = load_model()

#     # 转换为 chat template 格式
#     ds_map = ds.map(
#         lambda example: build_chat_text(example, tokenizer),
#         remove_columns=["Prompt", "Completion"]
#     )

#     # 保存到本地，供 sft_train 加载
#     ds_map.save_to_disk('./city_date_dataset/city_date_map')
#     print(f"数据集已保存，共 {len(ds_map)} 条")
#     print("示例数据：")
#     print(ds_map[0])

#     # 清理
#     del tokenizer
#     gc.collect()
#     torch.cuda.empty_cache()

def generate_json():
    """第1步：从 city.txt 随机生成问答，输出 xlsx"""
    Q_list = [
        ('{city}{year}年{month}月{day}日的天气','%Y-%m-%d'),
        ('{city}{year}年{month}月{day}号的天气','%Y-%m-%d'),
        ('{city}{month}月{day}日的天气','%m-%d'),
        ('{city}{month}月{day}号的天气','%m-%d'),

        ('{year}年{month}月{day}日{city}的天气','%Y-%m-%d'),
        ('{year}年{month}月{day}号{city}的天气','%Y-%m-%d'),
        ('{month}月{day}日{city}的天气','%m-%d'),
        ('{month}月{day}号{city}的天气','%m-%d'),

        ('你们{year}年{month}月{day}日去{city}玩吗？','%Y-%m-%d'),
        ('你们{year}年{month}月{day}号去{city}玩么？','%Y-%m-%d'),
        ('你们{month}月{day}日去{city}玩吗？','%m-%d'),
        ('你们{month}月{day}号去{city}玩吗？','%m-%d'),
    ]

    prompt_template = '''
给定一句话："%s"，请你按步骤要求工作。

步骤1：识别这句话中的城市和日期共2个信息
步骤2：根据城市和日期信息，生成JSON字符串，格式为{"city":城市,"date":日期}

请问，这个JSON字符串是：
'''

    with open('./city_date_dataset/city.txt', 'r', encoding='utf-8') as fp:
        city_list = fp.readlines()
        city_list = [line.strip().split(' ')[1] for line in city_list]

    Q_arr = []
    A_arr = []
    # 生成一批"1月2号"、"1月2日"、"2023年1月2号", "2023年1月2日", "2023-02-02", "03-02"之类的话术, 教会它做日期转换
    for i in range(1000):
        Q = Q_list[random.randint(0, len(Q_list) - 1)]
        city = city_list[random.randint(0, len(city_list) - 1)]
        year = random.randint(1990, 2025)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        time_str = '{}-{}-{}'.format(year, month, day)
        date_field = time.strftime(Q[1], time.strptime(time_str, '%Y-%m-%d'))
        Q = Q[0].format(city=city, year=year, month=month, day=day)  # 问题
        A = json.dumps({'city': city, 'date': date_field}, ensure_ascii=False)  # 回答

        Q_arr.append(prompt_template % (Q,))
        A_arr.append(A)

    with open('./city_date_dataset/train.jsonl', 'w', encoding='utf-8') as f:
        for Q, A in zip(Q_arr, A_arr):
            f.write(json.dumps({
                "Prompt": Q,
                "Completion": A,
            }, ensure_ascii=False) + "\n")

def convert_jsonl_to_dataset():
    ds = load_dataset("json", data_files="./city_date_dataset/train.jsonl")
    print(f"数据集已加载，共 {len(ds['train'])} 条")
    print("示例数据：")
    print(ds['train'][0])

    # 加载 tokenizer
    _, tokenizer = load_model()

    # 转换为 chat template 格式
    ds_map = ds.map(
        lambda example: build_chat_text(example, tokenizer),
        remove_columns=["Prompt", "Completion"]
    )

    # 保存到本地，供 sft_train 加载
    ds_map.save_to_disk('./city_date_dataset/city_date_map2_fromJson')
    print(f"数据集已保存，共 {len(ds_map['train'])} 条")
    print("示例数据：")
    print(ds_map['train'][0])

    # 清理
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # generate_xlsx()      # 第1步：生成 xlsx
    # convert_xlsx_to_dataset()  # 第2步：转为可训练数据集

    # generate_json()
    convert_jsonl_to_dataset()