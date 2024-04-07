#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 在Unix/Linux环境中指定脚本解释器为python,指定文件编码为UTF-8
# 本代码是使用transformers和PEFT模型（用于因果语言模型的模型）的Python脚本，可以通过命令行接口加载模型和分词器，以生成基于给定提示的响应。

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

app = typer.Typer(pretty_exceptions_show_locals=False) # 创建typer应用


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

@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='')],
        out_dir: Annotated[str, typer.Option(help='')],
):
    model, tokenizer = load_model_and_tokenizer(model_dir) # 加载模型和分词器

    # 把加载原模型和lora模型后做合并，并保存
    merged_model = model.merge_and_unload() # 合并模型
    merged_model.save_pretrained(out_dir, safe_serialization=True) # 保存合并后的模型
    tokenizer.save_pretrained(out_dir) # 保存分词器


# 如果脚本直接运行
if __name__ == '__main__':
    app() # 运行typer应用
