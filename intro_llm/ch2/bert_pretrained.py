import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json

from datasets import concatenate_datasets, load_dataset
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline

from itertools import chain


bookcorpus = load_dataset("bookcorpus", "plain_text", split="train")
wiki = load_dataset("wikipedia", "20220301.en", split="train")

wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

dataset = concatenate_datasets([bookcorpus, wiki])

d = dataset.train_test_split(test_size=0.1)

def dataset_to_text(dataset, output_filename="data.txt"):
    with open(output_filename, "w", encoding="utf-8") as f:
        for t in dataset["text"]:
            print(t, file=f)

# 将训练集保存为 train.txt
dataset_to_text(d["train"], "train.txt")

# 将测试集保存为 test.txt
dataset_to_text(d["test"], "test.txt")


special_tokens = [
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]
files = ["train.txt"]
vocab_size = 30_522
max_length = 512
truncate_longer_samples = False

tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
tokenizer.enable_truncation(max_length=max_length)

model_path = "pretrained-bert"

if not os.path.isdir(model_path):
  os.mkdir(model_path)
tokenizer.save_model(model_path)
with open(os.path.join(model_path, "config.json"), "w") as f:
  tokenizer_cfg = {
      "do_lower_case": True,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "model_max_length": max_length,
      "max_len": max_length,
  }
  json.dump(tokenizer_cfg, f)

tokenizer = BertTokenizerFast.from_pretrained(model_path)

def encode_with_truncation(examples):
    """使用词元分析对句子进行处理并截断的映射函数（Mapping function）"""
    return tokenizer(examples["text"], truncation=True, padding="max_length",
                     max_length=max_length, return_special_tokens_mask=True)

def encode_without_truncation(examples):
    """使用词元分析对句子进行处理且不截断的映射函数（Mapping function）"""
    return tokenizer(examples["text"], return_special_tokens_mask=True)

encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

train_dataset = d["train"].map(encode, batched=True)
test_dataset = d["test"].map(encode, batched=True)

if truncate_longer_samples:
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
    test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])


def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result


if not truncate_longer_samples:
    train_dataset = train_dataset.map(group_texts, batched=True,
                                      desc=f"Grouping texts in chunks of {max_length}")
    test_dataset = test_dataset.map(group_texts, batched=True,
                                    desc=f"Grouping texts in chunks of {max_length}")
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)

training_args = TrainingArguments(
    output_dir=model_path,          # 输出目录，用于保存模型检查点
    evaluation_strategy="steps",    # 每隔 `logging_steps` 步进行一次评估
    overwrite_output_dir=True,
    num_train_epochs=10,            # 训练时的轮数，可以根据需要进行调整
    per_device_train_batch_size=10, # 训练批量大小，可以根据 GPU 内存容量将其设置得尽可能大
    gradient_accumulation_steps=8,  # 在更新权重之前累积梯度
    per_device_eval_batch_size=64,  # 评估批量大小
    logging_steps=1000,             # 每隔 1000 步进行一次评估，记录并保存模型检查点
    save_steps=1000,
    # load_best_model_at_end=True,  # 是否在训练结束时加载最佳模型（根据损失）
    # save_total_limit=3,           # 如果磁盘空间有限，则可以限制只保存 3 个模型权重
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-10000"))
tokenizer = BertTokenizerFast.from_pretrained(model_path)

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

examples = [
  "Today's most trending hashtags on [MASK] is Donald Trump",
  "The [MASK] was cloudy yesterday, but today it's rainy.",
]
for example in examples:
  for prediction in fill_mask(example):
    print(f"{prediction['sequence']}, confidence: {prediction['score']}")
  print("="*50)