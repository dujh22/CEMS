import json
import random
import os

def split_data(jsonl_file_path, output_dir, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    # 确保比例之和为1
    assert train_ratio + dev_ratio + test_ratio == 1, "The sum of ratios must be 1."
    
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取JSONL文件
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]
    
    # 随机打乱数据
    random.shuffle(data)
    
    # 计算划分点
    total_size = len(data)
    train_end = int(total_size * train_ratio)
    dev_end = train_end + int(total_size * dev_ratio)
    
    # 划分数据集
    train_data = data[:train_end]
    dev_data = data[train_end:dev_end]
    test_data = data[dev_end:]
    
    # 保存数据集
    train_path = os.path.join(output_dir, 'train.json')
    dev_path = os.path.join(output_dir, 'dev.json')
    test_path = os.path.join(output_dir, 'test.json')

    with open(train_path, 'w', encoding='utf-8') as file:
        json.dump(train_data, file, ensure_ascii=False, indent=4)
    with open(dev_path, 'w', encoding='utf-8') as file:
        json.dump(dev_data, file, ensure_ascii=False, indent=4)
    with open(test_path, 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)

# 使用
jsonl_file_path = 'F://code//github//ce_finetune//data//ce_data//target_file.jsonl'
output_dir = 'F://code//github//ce_finetune//data//ce_data_split'
split_data(jsonl_file_path, output_dir)