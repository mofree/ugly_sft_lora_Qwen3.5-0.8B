"""
Qwen3.5-0.8B 模型调用示例。
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型路径
MODEL_PATH = "./Qwen/Qwen3___5-0___8B"

def load_model():
    """加载模型和分词器"""
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto"  # 自动选择设备
    )
    print("模型加载完成!")
    return model, tokenizer

def chat(model, tokenizer, prompt):
    """简单的对话函数"""
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
    reply = tokenizer.decode(response, skip_special_tokens=True)
    return reply

def main():
    """主函数"""
    model, tokenizer = load_model()
    
    prompt = """请在我问的问题中提取出城市名和日期，并按照如下格式：
城市名：xx
日期：xx

现在我的问题是：乌鲁木齐3月23日天气。请直接按照格式回答："""

    response = chat(model, tokenizer, prompt)
    print(response)

if __name__ == "__main__":
    main()
