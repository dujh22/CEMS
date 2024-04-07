# -*- coding: utf-8 -*-
# 设置文件编码为utf-8

import subprocess
# subprocess.run(["pip", "install", "-r", "/workspace/dujh22/ce_finetune/requirements.txt"])
# subprocess.run(["pip", "install", "-r", "/workspace/dujh22/ce_finetune/requirements.txt"])
# subprocess.run(["pip", "install", "-r", "/workspace/dujh22/ce_finetune/requirements.txt"])
# subprocess.run(["pip", "install", "-r", "/workspace/dujh22/ce_finetune/requirements.txt"])
# subprocess.run(["pip", "install", "-r", "/workspace/dujh22/ce_finetune/requirements.txt"])
# subprocess.run(["pip", "install", "-r", "/workspace/dujh22/ce_finetune/requirements.txt"])
# subprocess.run(["pip", "install", "-r", "/workspace/dujh22/ce_finetune/requirements.txt"])
# subprocess.run(["pip", "install", "-r", "/workspace/dujh22/ce_finetune/requirements.txt"])

def install_dependencies(retries=20):
    """尝试安装依赖，最多重试 retries 次"""
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(["pip", "install", "-r", "/workspace/dujh22/ce_finetune/requirements.txt"], check=True)
            print(f"依赖安装成功，在第{attempt}次尝试后。")
            return True  # 成功安装，退出函数
        except subprocess.CalledProcessError:
            print(f"尝试{attempt}次安装依赖失败，正在重试...")

    print("依赖安装多次尝试失败，脚本终止执行。")
    return False  # 多次尝试后仍然失败

# 调用函数尝试安装依赖
if install_dependencies(retries=3):
    print("继续执行后续操作...")
    # 在这里添加您的后续操作代码
else:
    print("由于依赖安装失败，后续操作不会执行。")

install_dependencies()
# 导入部分------------------------------------------------------------------

# 导入必要的库和模块
import dataclasses as dc  # 导入dataclasses模块，用于创建数据类
import functools  # 导入functools模块，提供了一系列高阶函数，比如@lru_cache装饰器
from collections.abc import Callable, Mapping, Sequence  # 导入抽象基类，用于类型提示
from pathlib import Path  # 导入Path类，用于路径操作
from typing import Annotated, Any, Optional, Union  # 导入类型提示模块

# 导入第三方库
import jieba  # 导入分词库
import numpy as np  # 导入NumPy库
import ruamel.yaml as yaml  # 导入yaml处理库，用于处理YAML文件
import torch  # 导入PyTorch库
import typer  # 导入Typer库，用于创建命令行接口
from datasets import Dataset, DatasetDict, NamedSplit, Split, load_dataset  # 导入hugging face的datasets库
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # 导入nltk库，用于计算BLEU分数
# peft是一个自定义的模块，用于加载peft模型
from peft import (
    PeftConfig,
    PeftModelForCausalLM,
    get_peft_config,
    get_peft_model
)
from rouge_chinese import Rouge  # 导入中文Rouge评分库
from torch import nn  # 导入PyTorch的nn模块，用于构建神经网络
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments, AutoConfig,
) # 导入transformers库中的类和函数，这些分别用于加载模型、加载tokenizer、评估预测、生成配置、加载预训练模型、加载预训练tokenizer、加载预训练tokenizer（快速版本）、加载Seq2Seq训练参数、加载预训练模型配置
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq  # 导入transformers库中的DataCollatorForSeq2Seq类，用于处理批次数据

from transformers import Seq2SeqTrainer as _Seq2SeqTrainer  # 导入transformers库中的Seq2SeqTrainer类
import os  # 导入os模块，用于操作系统功能，如文件路径

# 定义模型和分词器的类型别名
ModelType = Union[PreTrainedModel, PeftModelForCausalLM] # 定义ModelType为PreTrainedModel和PeftModelForCausalLM的联合类型
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast] # 定义TokenizerType为PreTrainedTokenizer和PreTrainedTokenizerFast的联合类型
app = typer.Typer(pretty_exceptions_show_locals=False) # 创建Typer应用实例，用于构建命令行接口

# 数据处理相关类------------------------------------------------------------------
# 接下来的部分定义了两个数据处理相关的类：DataCollatorForSeq2Seq 和 Seq2SeqTrainer。这些类继承并扩展了transformers库中的相应功能，以适应特定的数据处理需求。

class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    # 定义DataCollatorForSeq2Seq类，用于在训练和评估时处理输入数据的批处理
    def __call__(self, features, return_tensors=None):
        # 定义__call__方法，使得DataCollatorForSeq2Seq的实例可以像函数一样被调用
        # features参数是输入的特征，return_tensors指定返回的张量类型
        output_ids = (
            [feature['output_ids'] for feature in features]
            if 'output_ids' in features[0].keys()
            else None
        )
        # 如果features中含有'output_ids'键，则提取这些id，否则设为None
        if output_ids is not None:
            # 计算输出序列的最大长度
            max_output_length = max(len(out) for out in output_ids)
            # 如果设置了pad_to_multiple_of，则将最大长度调整到该倍数
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
            # 对每个feature的输出id进行填充，使长度统一
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                        max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)
        # 调用父类的__call__方法，完成剩余的处理
        return super().__call__(features, return_tensors)


class Seq2SeqTrainer(_Seq2SeqTrainer):
    # 定义Seq2SeqTrainer类，用于执行序列到序列模型的训练和评估
    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys=None,
            **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 此方法在进行预测时调用，用于执行一步预测操作
        if self.args.predict_with_generate:
            output_ids = inputs.pop('output_ids')
        input_ids = inputs['input_ids']
        # 调用父类的prediction_step方法，进行预测
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )
        # 调整生成的tokens，使之与输入对齐
        generated_tokens = generated_tokens[:, input_ids.size()[1]:]
        if self.args.predict_with_generate:
            labels = output_ids
        return loss, generated_tokens, labels

# 这部分主要涉及到数据处理和预测步骤的自定义实现，例如如何处理输入输出数据、如何执行预测等。数据整理器DataCollatorForSeq2Seq在数据准备阶段调用，负责将数据批处理到统一的长度，而Seq2SeqTrainer在训练和预测时被调用，对模型的预测步骤进行自定义操作。
    



# 辅助函数和数据处理配置类------------------------------------------------------------------
# 这部分定义了一些辅助函数和用于配置数据处理的类，例如解析路径、进行简单的数据检查、加载数据集等。

def _resolve_path(path: Union[str, Path]) -> Path:
    # 定义_resolve_path函数，用于解析给定的路径字符串或Path对象为绝对路径
    # path参数是输入的路径字符串或Path对象
    # 返回值是解析后的Path对象
    return Path(path).expanduser().resolve()


def _sanity_check(
        input_ids: Sequence[int],
        output_ids: Sequence[int],
        tokenizer: PreTrainedTokenizer,
):
    # 定义_sanity_check函数，用于对给定的输入和输出id序列进行简单的检查
    # input_ids是输入id序列，output_ids是输出id序列，tokenizer是分词器实例
    print('--> Sanity check')
    for in_id, out_id in zip(input_ids, output_ids):
        if in_id == 0:
            continue
        if in_id in tokenizer.tokenizer.index_special_tokens:
            in_text = tokenizer.tokenizer.index_special_tokens[in_id]
        else:
            in_text = tokenizer.decode([in_id])
        print(f'{repr(in_text):>20}: {in_id} -> {out_id}')


@functools.cache
def _get_yaml_parser() -> yaml.YAML:
    # 创建并返回一个配置为安全和纯Python的YAML解析器
    parser = yaml.YAML(typ='safe', pure=True)
    parser.indent(mapping=2, offset=2, sequence=4)
    parser.default_flow_style = False
    return parser


@dc.dataclass
class DataConfig(object):
    # 定义数据配置的数据类，用于存储数据相关的配置信息
    train_file: str
    val_file: Optional[str] = None
    test_file: Optional[str] = None

    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:
        # 根据训练文件的后缀名推断数据格式
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        # 将数据文件映射为NamedSplit枚举到文件路径的字典
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            )
            if data_file is not None
        }


@dc.dataclass
class FinetuningConfig(object):
    # 定义模型微调配置的数据类
    data_config: DataConfig

    max_input_length: int
    max_output_length: int

    training_args: Seq2SeqTrainingArguments = dc.field(
        default=Seq2SeqTrainingArguments(output_dir='./output')
    )
    peft_config: Optional[PeftConfig] = None

    def __post_init__(self):
        # 初始化后进行的额外设置，如确保评估策略的一致性
        if not self.training_args.do_eval or self.data_config.val_file is None:
            # skips the evaluation stage when `do_eval` or `eval_file` is not provided
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = 'no'
            self.data_config.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = (
                    self.training_args.per_device_eval_batch_size
                    or self.training_args.per_device_train_batch_size
            )

    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        # 从字典创建FinetuningConfig对象的工厂方法
        # 处理training_args和data_config的嵌套字典
        training_args = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments
        ):
            gen_config = training_args.get('generation_config')
            # TODO: a bit hacky
            if not isinstance(gen_config, GenerationConfig):
                training_args['generation_config'] = GenerationConfig(
                    **gen_config
                )
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):
            kwargs['data_config'] = DataConfig(**data_config)

        peft_config = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs['peft_config'] = get_peft_config(peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        # 从YAML文件加载配置
        path = _resolve_path(path)
        kwargs = _get_yaml_parser().load(path)
        return cls.from_dict(**kwargs)


def _load_datasets(
        data_dir: Path,
        data_format: str,
        data_files: dict[NamedSplit, str],
        num_proc: Optional[int],
) -> DatasetDict:
    # 定义_load_datasets函数，用于根据给定的数据格式和文件，加载数据集
    if data_format in ('.csv', '.json', '.jsonl'):
        dataset_dct = load_dataset(
            data_format[1:],
            data_dir=data_dir,
            data_files=data_files,
            num_proc=num_proc,
        )
    else:
        err_msg = f"Cannot load dataset in the '{data_format}' format."
        raise NotImplementedError(err_msg)

    return dataset_dct


class DataManager(object):
    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc

        self._dataset_dct = _load_datasets(
            _resolve_path(data_dir),
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        return self._dataset_dct.get(split, None)

    def get_dataset(
            self,
            split: NamedSplit,
            process_fn: Callable[[dict[str, Any]], dict[str, Any]],
            batched: bool = True,
            remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:
            return

        if remove_orig_columns:
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None
        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
        )


def print_model_size(model: PreTrainedModel):
    print("--> Model")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> model has {total_params / 1e6}M params\n")

# 用于处理批次数据
def process_batch(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_tools = batch.get('tools', None) # 获取tools字段，如果没有则返回None
    batched_conv = batch['conversations'] # 获取conversations字段
    batched_input_ids = [] # 用于存储input_ids
    batched_labels = [] # 用于存储labels

    if batched_tools is None:
        batched_tools = [None] * len(batched_conv) # 如果tools字段为空，则将batched_tools设置为None
        # *len(batched_conv)表示将batched_tools设置为一个长度为len(batched_conv)的列表，列表中的元素都是None

    # 遍历tools和conversations
    for tools, conv in zip(batched_tools, batched_conv):
        input_ids, loss_masks = [
            tokenizer.get_command('[gMASK]'), 
            tokenizer.get_command('sop'),
        ], [False, False]
        # get_command方法返回一个列表，列表中的元素是对应的token_id

        if tools is not None:
            raise NotImplementedError()

        for message in conv:
            if message['role'] in ('system', 'user'):
                loss_mask_val = False
            else:
                loss_mask_val = True

            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
                new_input_ids = tokenizer.build_single_message(
                    message['role'], '', message['content']
                )
                new_loss_masks = [loss_mask_val] * len(new_input_ids)

            input_ids += new_input_ids
            loss_masks += new_loss_masks

        input_ids.append(tokenizer.eos_token_id)
        loss_masks = [False, *loss_masks]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        max_length = max_input_length + max_output_length + 1
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])
    return {'input_ids': batched_input_ids, 'labels': batched_labels}

# 定义process_batch_eval函数，用于处理评估时的批次数据
def process_batch_eval(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_tools = batch.get('tools', None)
    batched_conv = batch['conversations']
    batched_input_ids = []
    # To avoid computing loss, we do not provide the `labels` field in the input dictionary.
    batched_output_ids = []

    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        input_ids = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ]

        if tools is not None:
            raise NotImplementedError()

        for message in conv:
            if len(input_ids) >= max_input_length:
                break
            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
                new_input_ids = tokenizer.build_single_message(
                    message['role'], '', message['content']
                )
                if message['role'] == 'assistant':
                    output_prompt, output_ids = (
                        new_input_ids[:1],
                        new_input_ids[1:],
                    )
                    output_ids.append(tokenizer.eos_token_id)
                    batched_input_ids.append(
                        input_ids[:max_input_length] + output_prompt[:1]
                    )
                    batched_output_ids.append(output_ids[:max_output_length])
                input_ids += new_input_ids
    return {'input_ids': batched_input_ids, 'output_ids': batched_output_ids}


# 定义_prepare_model_for_training函数，用于准备模型进行训练，包括参数类型转换等
# TODO: Not sure if this is necessary, can set it to half
def _prepare_model_for_training(model: nn.Module, use_cpu: bool):
    for param in model.parameters():
        if param.requires_grad or use_cpu:
	    # if train with cpu, cast all params to fp32 instead of trainable ones.
            param.data = param.data.to(torch.float32)


def load_tokenizer_and_model(
        model_dir: str, # 模型目录路径或huggingface.co上的模型ID
        peft_config: Optional[PeftConfig] = None,  # PEFT配置，可选，用于特定的模型调整如前缀调优或LoRA
) -> tuple[PreTrainedTokenizer, nn.Module]:
    # 从指定目录加载分词器，信任远程代码
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # 如果指定了PEFT配置
    if peft_config is not None:
        # 如果使用的是前缀调优
        if peft_config.peft_type.name == "PREFIX_TUNING":
            # 加载模型配置，并根据PEFT配置调整
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            config.pre_seq_len = peft_config.num_virtual_tokens
            config.use_cache = False
            # 加载预训练模型
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                config=config,
            )
        # 如果使用的是LoRA
        if peft_config.peft_type.name == "LORA":
            # 直接加载预训练模型，不使用缓存
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                empty_init=False,
                use_cache=False
            )
            # 应用PEFT模型调整
            model = get_peft_model(model, peft_config)
            # 打印可训练参数
            model.print_trainable_parameters()
    else:
        # 如果没有指定PEFT配置，直接加载预训练模型
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            empty_init=False,
            use_cache=False
        )
    # 打印模型大小
    print_model_size(model)
    # 返回分词器和模型
    return tokenizer, model

# 从评估预测中提取预测ID和标签ID
def compute_metrics(eval_preds: EvalPrediction, tokenizer: PreTrainedTokenizer):
    batched_pred_ids, batched_label_ids = eval_preds

    # 初始化度量字典
    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
    # 遍历每一组预测ID和标签ID
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        # 解码预测文本和标签文本
        pred_txt = tokenizer.decode(pred_ids).strip()
        label_txt = tokenizer.decode(label_ids).strip()
        # 使用jieba进行分词
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        # 初始化Rouge评估器
        rouge = Rouge()
        # 计算Rouge得分
        # 原来是：scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        try:
            scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        except: # 如果出现异常，跳过当前数据点
            continue
        # 将得分添加到度量字典
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v['f'] * 100, 4))
        # 计算BLEU-4得分
        metrics_dct['bleu-4'].append(
            sentence_bleu(
                [label_tokens],
                pred_tokens,
                smoothing_function=SmoothingFunction().method3,
            )
        )
    # 返回平均得分
    return {k: np.mean(v) for k, v in metrics_dct.items()}


# Annotated是一个类型提示，它的作用是为函数参数添加额外的信息，比如help文档等
@app.command() # app.command()装饰器将main函数注册为一个命令行命令
def main(
    data_dir: str = typer.Argument(
            default='/workspace/dujh22/ce_finetune/data/ce_data_fix/',
            help='数据目录的路径。'
        ), # 数据目录
    model_dir: str = typer.Argument(
            default='/workspace/dujh22/models/chatglm3/',
            help='指定在huggingface.co上托管的预训练模型配置的模型id的字符串，或包含模型配置文件的目录的路径。'
        ), # 模型目录或ID
    config_file: str = typer.Argument(
            default='/workspace/dujh22/ce_finetune/configs/lora_multimachine.yaml',
            help='配置文件的路径。'
        ), # 配置文件
    auto_resume_from_checkpoint: str = typer.Option(
            default='',
            help='如果输入为"yes"，则自动使用最新的保存检查点。对于特定检查点，使用它们的标识符（例如，"12", "15"）。输入"no"以重新开始训练。'
        ) # 自动从检查点恢复
):
    # 从文件加载微调配置
    ft_config = FinetuningConfig.from_file(config_file) # 从配置文件中加载finetuning配置
    # 加载分词器和模型
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config) # 加载tokenizer和model
    # 创建数据管理器
    data_manager = DataManager(data_dir, ft_config.data_config) # 加载数据管理器

    # 加载训练数据集
    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    print('train_dataset:', train_dataset)
    # 加载验证数据集
    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if val_dataset is not None:
        print('val_dataset:', val_dataset)
    # 加载测试数据集
    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if test_dataset is not None:
        print('test_dataset:', test_dataset)

    # checks encoded dataset
    # _sanity_check(
    #     train_dataset[0]["input_ids"], train_dataset[0]["labels"], tokenizer
    # )

    # 准备模型训练，设置模型为fp32精度
    _prepare_model_for_training(model, ft_config.training_args.use_cpu)

    # 设置训练参数
    ft_config.training_args.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )
    ft_config.training_args.generation_config.eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command('<|user|>'),
        tokenizer.get_command('<|observation|>'),
    ]
    model.gradient_checkpointing_enable() # 启用梯度检查点
    model.enable_input_require_grads() # 启用输入的梯度计算

    # 创建序列到序列的训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset.select(list(range(50))),
        tokenizer=tokenizer,
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )

    # 判断是否自动从检查点恢复训练
    # Determine whether to continue training without breakpoints or if it is empty, then start training again directly
    if auto_resume_from_checkpoint.upper() == "" or auto_resume_from_checkpoint is None:
        trainer.train()
    else:
        # 自动恢复训练的逻辑
        output_dir = ft_config.training_args.output_dir
        dirlist = os.listdir(output_dir)
        checkpoint_sn = 0
        for checkpoint_str in dirlist:
            if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:
                checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                if checkpoint > checkpoint_sn:
                    checkpoint_sn = checkpoint
        if auto_resume_from_checkpoint.upper() == "YES":
            if checkpoint_sn > 0:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
                checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                print("resume checkpoint from  checkpoint-" + str(checkpoint_sn))
                trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                trainer.train()
        else:
            if auto_resume_from_checkpoint.isdigit():
                if int(auto_resume_from_checkpoint) > 0:
                    checkpoint_sn = int(auto_resume_from_checkpoint)
                    model.gradient_checkpointing_enable()
                    model.enable_input_require_grads()
                    checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                    print("resume checkpoint from  checkpoint-" + str(checkpoint_sn))
                    trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                print(auto_resume_from_checkpoint,
                      "The specified checkpoint sn(" + auto_resume_from_checkpoint + ") has not been saved. Please search for the correct chkeckpoint in the model output directory")

    # 如果有测试数据集，执行预测
    if test_dataset is not None:
        trainer.predict(test_dataset)


if __name__ == '__main__':
    app() # app是一个typer.Typer()对象，调用app()就是调用typer.Typer()对象的__call__方法
