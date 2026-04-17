# 上面代码似乎无效，必须在powershell本环境中设置$env:PYTHONUTF8=1

from simple_chat import load_model
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
import gc, torch # 清理内存
from trl import SFTTrainer
from transformers import TrainingArguments

def clear_memory(model, ds_map):
    del model, ds_map
    gc.collect()
    torch.cuda.empty_cache()

ds_map = load_from_disk("./city_date_dataset/city_date_map2_fromJson") # create_data_city_date.py中存的处理好的数据
model, tokenizer = load_model()

lora_config = LoraConfig(
    r=16, # 秩
    lora_alpha=32, # 缩放系数
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
# print(model)
# print(model.print_trainable_parameters())
# trainable params: 1,081,344 || all params: 753,474,368 || trainable%: 0.1435

training_args = TrainingArguments(
    output_dir="./city_date_train_output",
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./city_date_train_output/logs",
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds_map["train"],
    args=training_args,
)

trainer.train()

# model.save_pretrained("./city_date_lora") # 保存的是adapter_model和adapter_config，和checkpoint里的一样

# clear_memory(model, ds_map)
