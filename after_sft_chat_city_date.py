"""
使用 LoRA Adapter 的 Qwen3.5-0.8B 对话脚本。
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 基础模型路径
BASE_MODEL_PATH = "./Qwen/Qwen3___5-0___8B"
# LoRA Adapter 路径（选择你训练好的 checkpoint）
ADAPTER_PATH = "./city_date_train_output/checkpoint-1000"  # 【改】

def load_model_sft():
    """加载基础模型和 LoRA Adapter"""
    print("正在加载模型和Adapter...")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # 加载 LoRA Adapter 并合并到基础模型
    print("正在加载 LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload()  # 合并权重，得到完整模型
    
    print("模型和 Adapter 加载完成!")
    return model, tokenizer

def chat(model, tokenizer, prompt):
    """对话函数"""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 应用聊天模板
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成回复
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 解码并提取回复
    response = outputs[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(response, skip_special_tokens=True).rstrip('\n')
    return reply

# 新增用来对比的基础模型
def load_model_base():
    """加载模型和分词器"""
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        device_map="auto"  # 自动选择设备
    )
    print("模型加载完成!")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model_base()
    
    print("\n" + "="*50)
    print("Qwen3.5-0.8B (带 LoRA) 对话助手已就绪！")
    print("="*50 + "\n")

    prompt_template='''
给定一句话：“%s”，请你按步骤要求工作。

步骤1：识别这句话中的城市和日期共2个信息
步骤2：根据城市和日期信息，生成JSON字符串，格式为{"city":城市,"date":日期}

请问，这个JSON字符串是：
'''
    
    Q_list=['2020年4月16号三亚下雨么？','青岛3-15号天气预报','5月6号下雪么，城市是威海','青岛2023年12月30号有雾霾么?','我打算6月1号去北京旅游，请问天气怎么样？','你们打算1月3号坐哪一趟航班去上海？','小明和小红是8月8号在上海结婚么?',
        '一起去东北看冰雕么，大概是1月15号左右，我们3个人一起']

    for Q in Q_list:
        prompt = prompt_template%(Q,)
        A = chat(model, tokenizer, prompt)
        print(f"Q: {Q}\nA: {A}\n")
