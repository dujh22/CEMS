import json
from typing import Union
from pathlib import Path

# resolve_path用于将输入的路径转换为绝对路径
def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()
    # expanduser()用于扩展路径"~"为用户主目录的相对路径
    # resolve()用于将路径转换为绝对路径
    # 例如：Path('~/data').expanduser().resolve() -> PosixPath('/home/username/data')


def _mkdir(dir_name: Union[str, Path]):
    dir_name = _resolve_path(dir_name)
    if not dir_name.is_dir():
        dir_name.mkdir(parents=True, exist_ok=False)


def convert_adgen(data_dir: Union[str, Path], save_dir: Union[str, Path]):
    def _convert(in_file: Path, out_file: Path):
        _mkdir(out_file.parent)
        with open(in_file, encoding='utf-8') as fin:
            with open(out_file, 'wt', encoding='utf-8') as fout:
                data = json.load(fin)  # 读取json文件
                for item in data:
                    # 初始化对话列表，从prompt开始
                    conversations = []
                    # 处理history部分，如果存在
                    for i in range(0, len(item.get('history', [])), 2):  # 以2为步长进行迭代
                        # 检查是否存在对应的response
                        if i+1 < len(item['history']):  # 确保有对应的assistant回复
                            user_content = item['history'][i]['prompt']
                            assistant_content = item['history'][i+1]['response']
                            # 添加用户输入
                            conversations.append({'role': 'user', 'content': user_content})
                            # 添加助手回复
                            conversations.append({'role': 'assistant', 'content': assistant_content})
                        else:
                            # 如果没有对应的assistant回复，则忽略这个用户输入
                            conversations = []
                            break  # 或者continue，取决于你想如何处理这种情况

                    # 添加当前对话，其中用户的部分是item['prompt']，助手的部分是gpt3point5turbo_modify_response
                    conversations.append({'role': 'user', 'content': item['prompt']})
                    conversations.append({'role': 'assistant', 'content': item.get('gpt3point5turbo_modify_response', '')})
                    
                    sample = {'conversations': conversations}
                    fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

    data_dir = _resolve_path(data_dir)
    save_dir = _resolve_path(save_dir)

    train_file = data_dir / 'train.json'
    if train_file.is_file(): # 判断文件是否存在
        out_file = save_dir / train_file.relative_to(data_dir)
        _convert(train_file, out_file)

    dev_file = data_dir / 'dev.json'
    if dev_file.is_file(): # 判断文件是否存在
        out_file = save_dir / dev_file.relative_to(data_dir)
        _convert(dev_file, out_file)

    test_file = data_dir / 'test.json'
    if test_file.is_file(): # 判断文件是否存在
        out_file = save_dir / test_file.relative_to(data_dir)
        _convert(test_file, out_file)


jsonl_file_path = 'F://code//github//ce_finetune//data//ce_data_split'
output_dir = 'F://code//github//ce_finetune//data//ce_data_fix'
convert_adgen(jsonl_file_path, output_dir)