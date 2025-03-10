import os
import numpy as np
import random
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import json
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import tiktoken
gpt2_tokenizer = tiktoken.get_encoding("gpt2")
cwd = os.getcwd() # 当前工作目录
project_name = "LawMind" # 项目名称
random.seed(42)
np.random.seed(42)

# 训练数据集 维基百科/百度百科/fine_web_edu
common_cn_data_files = ["llmcorpus/CN/wiki/wikipedia-cn-20230720-filtered.json"]
common_eng_data_files = ["llmcorpus/ENG/tokenized_finewebedu/train_0.bin"]

def read_corpus(mode="train_tokenizer"):
    files = common_eng_data_files+common_cn_data_files
    for file_path in files:
        if file_path in common_cn_data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in json.load(f):
                    if mode=="train_tokenizer":
                        yield line['completion']
                    if mode=="pretrain":
                        yield line['completion']+"<｜end▁of▁sentence｜>"
        if file_path in common_eng_data_files:
            idx,step =0, 10000
            npt = np.memmap(file_path, dtype=np.uint16, mode='r')
            while step*idx<len(npt):
                idx+=1
                txt = gpt2_tokenizer.decode(npt[step*idx:step*idx+step])
                if mode=="train_tokenizer":
                    yield txt.replace("<|endoftext|>","")
                if mode=="pretrain":
                    yield txt.replace("<|endoftext|>","<｜end▁of▁sentence｜>")

def train_tokenizer():
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token
    special_tokens = ["<｜UNK｜>", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"]

    # 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=16384,
        special_tokens=special_tokens,  # 确保这三个token被包含
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 读取文本数据
    texts = read_corpus()

    # 训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    assert tokenizer.token_to_id("<｜UNK｜>") == 0
    assert tokenizer.token_to_id("<｜begin▁of▁sentence｜>") == 1
    assert tokenizer.token_to_id("<｜end▁of▁sentence｜>") == 2

    # 保存tokenizer
    tokenizer_dir = "LawMind/model/LawMind_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("LawMind/model/LawMind_tokenizer")

    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<｜UNK｜>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<｜begin▁of▁sentence｜>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<｜end▁of▁sentence｜>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<｜begin▁of▁sentence｜>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<｜end▁of▁sentence｜>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<｜UNK｜>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<｜UNK｜>",
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<｜begin▁of▁sentence｜>system\\n' + system_message + '<｜end▁of▁sentence｜>\\n' }}{% else %}{{ '<｜begin▁of▁sentence｜>system\\n你是 LawMind，是一个有用的人工智能助手。<｜end▁of▁sentence｜>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<｜begin▁of▁sentence｜>user\\n' + content + '<｜end▁of▁sentence｜>\\n<｜begin▁of▁sentence｜>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<｜end▁of▁sentence｜>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")

def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("LawMind/model/LawMind_tokenizer")

    messages = [
        {"role": "system", "content": "你是一个优秀的法律助手"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自LawMind项目'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(new_prompt)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))

    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print('decoder和原始文本是否一致：', response == new_prompt)


def encode_corpus():
    lawmind_tokenizer = AutoTokenizer.from_pretrained("LawMind/model/LawMind_tokenizer")
    shard_size, shard_idx = int(1e8),0
    filename = os.path.join(cwd, f'{project_name}/dataset/ds_{shard_idx}.bin')
    arr = np.memmap(filename=filename,dtype=np.uint16,mode="w+",shape=(shard_size,))
    token_count = 0
    # 读取文本,以tokenid的形式分块存储
    print("preparing corpus")
    texts = read_corpus("pretrain")
    print("processing start")
    for text in texts:
        ids = lawmind_tokenizer.encode(text)
        if token_count+len(ids)<shard_size:
            arr[token_count:token_count+len(ids)]=ids
            token_count+=len(ids)
        else:
            # 写入数据
            arr.flush()
            print(f"{filename} process is completed")
            # 更新参数开辟新块
            shard_idx+=1
            filename = os.path.join(cwd, f'{project_name}/dataset/ds_{shard_idx}.bin')
            arr = np.memmap(filename=filename,dtype=np.uint16,mode="w+",shape=(shard_size,))
            token_count = 0
            # 存储上一个未存储的文本
            arr[token_count:token_count+len(ids)]=ids
            token_count+=len(ids)
        

    

def main():
    # train_tokenizer()
    eval_tokenizer()
    encode_corpus()


if __name__ == '__main__':
    main()
