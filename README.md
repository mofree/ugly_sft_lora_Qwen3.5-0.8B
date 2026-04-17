# ugly_sft_lora_Qwen3.5-0.8B



## 项目说明
qwen3.5 0.8b sft LoRA微调，以实现动漫风格人物回答，由于模型本身能力可能也不错，且数据集为全英文，所以可能微调效果不显著。所以又执行了一个 城市、日期 字段提取的微调任务，在微调数据集中固定了输出为json的格式，且日期严格使用-分割，微调后的模型能很好遵循输出要求。

## 脚本说明

| 文件 | 功能 |
|---|---|
| `simple_chat.py` | 加载基础模型，提供交互式对话 |
| `ugly_chat.py` | 早期对话脚本（未套 chat template） |
| `create_data_anime.py` | 动漫角色数据集 → chat template → HuggingFace Dataset |
| `create_data_city_date.py` | 城市日期数据集生成（jsonl）→ chat template → HuggingFace Dataset |
| `sft_train_anime.py` | 动漫角色 LoRA 微调训练 |
| `sft_train_city_date.py` | 城市日期提取 LoRA 微调训练 |
| `after_sft_chat_anime.py` | 加载动漫 LoRA Adapter 进行对话 |
| `after_sft_chat_city_date.py` | 加载城市日期 LoRA Adapter 进行对话 |

## 依赖

见 `pip_list.txt`

## 参考代码

`ref_anime_sft_code/` 和 `ref_city_date_sft_code/` 中为他人程序代码，仅供参考

## 参考链接
* Qwen3.5-0.8B：https://huggingface.co/Qwen/Qwen3.5-0.8B
* 用 SFT + LoRA 为模型注入动漫人格：https://www.bilibili.com/video/BV1UaPmzrESw/?spm_id_from=333.337.search-card.all.click&vd_source=e84e9da84516bd527398f1c8242cd87c
* 动漫风格数据集：https://huggingface.co/datasets/maomao88/anime-waifu-personality-chat-with-questions
* 通义千问1.8B大模型微调，实现天气预报功能：https://www.bilibili.com/video/BV16a4y1z7LY/?spm_id_from=333.337.search-card.all.click&vd_source=e84e9da84516bd527398f1c8242cd87c
  上述任务对应的github仓库，但代码与视频有所区别：https://github.com/owenliang/qwen-sft/tree/main
* Qwen3.5 0.8B/2B/4B/9B 小模型本地部署指南，微调教程：https://zhuanlan.zhihu.com/p/2012668704182260051
