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
    
    print("\n" + "="*50)
    print("Qwen3.5-0.8B 对话助手已就绪！")
    print("输入 'quit' 或 'exit' 退出程序")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见!")
                break
            
            if not user_input:
                continue
                
            response = chat(model, tokenizer, user_input)
            print(f"Qwen: {response}")
            
        except KeyboardInterrupt:
            print("\n\n已退出!")
            break

if __name__ == "__main__":
    main()
