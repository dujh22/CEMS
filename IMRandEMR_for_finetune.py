#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os # 导入os模块，用于操作系统功能，如文件路径
os.environ['CUDA_VISIBLE_DEVICES'] = "5" # 设置CUDA_VISIBLE_DEVICES环境变量，指定使用的GPU编号

# -------------------------------------------------

import re

# 用于计算中英混杂比率（IMR）和中英混杂率（EMR）的函数--------------------------------------

def is_chinese(char):
    """检查一个字符是否为中文。"""
    return re.match(r'[\u4e00-\u9fff]', char)

def calculate_imr(responses):
    """计算单个响应的中英混杂比率，并返回所有响应的最大、最小和平均比率。"""
    ratios = []
    for response in responses:
        chinese_chars = sum(is_chinese(char) is not None for char in response)
        total_chars = len(response)
        ratio = chinese_chars / total_chars if total_chars > 0 else 0
        ratios.append(ratio)
    return max(ratios), min(ratios), sum(ratios) / len(ratios) if ratios else 0

def calculate_emr_char(responses):
    """计算所有响应的中英混杂比率。"""
    total_chinese_chars = sum(sum(is_chinese(char) is not None for char in response) for response in responses)
    total_chars = sum(len(response) for response in responses)
    return total_chinese_chars / total_chars if total_chars > 0 else 0

def calculate_emr_case(responses):
    """计算所有响应的中英混杂率。"""
    total_chinese_cases = sum(any(is_chinese(char) is not None for char in response) for response in responses)
    return total_chinese_cases / len(responses) if responses else 0

# 加载模型---------------------------------------------------------------------------


from pathlib import Path # 导入路径操作库
from typing import Annotated, Union # 导入类型注解支持库

import typer # 导入命令行应用构建库
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM # 从peft库导入模型类
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
) # 从transformers库导入模型和分词器类

ModelType = Union[PreTrainedModel, PeftModelForCausalLM] # 定义模型类型的联合类型
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast] # 定义分词器类型的联合类型

# 定义函数解析路径
def _resolve_path(path: Union[str, Path]) -> Path:
    # 将路径展开为绝对路径
    return Path(path).expanduser().resolve()

# 定义加载模型和分词器的函数
def load_model_and_tokenizer(model_dir: Union[str, Path]) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir) # 解析模型目录路径
    # 如果adapter配置文件存在
    if (model_dir / 'adapter_config.json').exists():
        # 加载Peft模型
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        # 获取分词器目录
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        # 加载transformers模型
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        # 设置分词器目录为模型目录
        tokenizer_dir = model_dir
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    return model, tokenizer # 返回模型和分词器


model_dir = "/workspace/dujh22/ce_finetune/output_single/checkpoint-3000/"
model, tokenizer = load_model_and_tokenizer(model_dir) # 加载模型和分词器
    
# -----------------------------------------------------------------------------------
import json
import concurrent.futures
import os
from tqdm import tqdm
import time

# 为了处理异常并在遇到失败时重试最多十次，我们可以在process_entry函数中添加异常处理逻辑。具体地说，将在调用GLM接口的部分添加一个循环，该循环最多尝试十次。如果所有尝试都失败，则捕获异常并允许代码继续执行，跳过当前的数据点。也将添加一些日志输出，以便跟踪哪些数据点被跳过。
def process_entry(data):

    # 构造messages
    messages = data.get('conversations', [])
    if messages[-1]['role'] == 'assistant':
        messages = messages[:-1]
    # 每2个对话一组构造prompt
    prompt = ""
    for i in range(0, len(messages), 2):
        prompt += f"User: {messages[i]['content']}\n"
        if i + 1 < len(messages):
            prompt += f"Assistant: {messages[i + 1]['content']}\n"
        else:
            prompt += "Assistant: "

    attempts = 0
    while attempts < 10:
        try:
            # 调用上述模型
            response, _ = model.chat(tokenizer, prompt) # 使用模型和分词器生成响应
            # 添加GLM响应到数据并返回
            messages.append({'role': 'assistant', 'content': response})
            return messages
        except Exception as e:
            print(f"在处理 {messages[0]} 时遇到异常：{e}")
            attempts += 1
            print(f"重试次数 {attempts}/10")
    
    print(f"跳过数据点 {messages[0]}，因为尝试了10次都失败了。")
    # 返回一个修改过的版本，其中包含错误信息，而不是简单地跳过，以便在输出文件中记录这一点。
    messages.append({'role': 'assistant', 'content': "Error"})
    return messages


def process_json_concurrently(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data_list = [json.loads(line) for line in infile]

    total = len(data_list)

    # 使用tqdm创建进度条
    with tqdm(total=total, desc="正在处理") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(process_entry, data) for data in data_list]
            results = []
            for future in concurrent.futures.as_completed(futures):
                # 每完成一个任务，进度条更新一次
                results.append(future.result())
                pbar.update(1)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        responses = []
        for result in results:
            if result[-1]['content'] == "Error":
                continue
            elif result[-1]['role'] == 'assistant':
                responses.append(result[-1]['content'])

        # 计算中英混杂比率（IMR）和中英混杂率（EMR）
        max_imr, min_imr, avg_imr = calculate_imr(responses)
        emr_char = calculate_emr_char(responses)
        emr_case = calculate_emr_case(responses)

        # 将IMR和EMR添加到输出文件
        outfile.write(f"max_imr: {max_imr}\n")
        outfile.write(f"min_imr: {min_imr}\n")
        outfile.write(f"avg_imr: {avg_imr}\n")
        outfile.write(f"emr_char: {emr_char}\n")
        outfile.write(f"emr_case: {emr_case}\n")

        # 输出存在中文回复的response
        for result in results:
            # outfile.write(f"{result}\n")
            if result[-1]['content'] == "Error":
                continue
            elif result[-1]['role'] == 'assistant':
                chinese_chars = sum(is_chinese(char) is not None for char in result[-1]['content'])
                if chinese_chars > 0:
                    outfile.write(f"{result}\n")
          
def main():
    # 注意ce_data_for_temp是小数据集测试本脚本的可用性
    # input_dir = '/workspace/dujh22/ce_finetune/data/ce_data_for_temp'
    input_dir = '/workspace/dujh22/ce_finetune/data/ce_data_fix'
    output_dir = '/workspace/dujh22/ce_finetune/data/ce_data_exam'

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有json文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            input_file_path = os.path.join(input_dir, file_name)
            output_file_path = os.path.join(output_dir, file_name)

            print(f"正在处理文件：{input_file_path}")
            process_json_concurrently(input_file_path, output_file_path)
            print(f"处理完成，输出文件：{output_file_path}")

if __name__ == "__main__":
    main()
