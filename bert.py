import os
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import json
from tqdm.notebook import tqdm
import time
'''
# 初始化 BERT 分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese').to('cuda' if torch.cuda.is_available() else 'cpu')
print(model)
'''
from transformers import BertTokenizer, BertModel, BertConfig
import torch
bert_model_dir = 'D:/project/WWW2021-master/chinese_wwm_pytorch'

# 初始化 BERT 分词器和模型
tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
config = BertConfig.from_pretrained(bert_model_dir + '/bert_config.json')
model = BertModel.from_pretrained(bert_model_dir, config=config).to('cuda' if torch.cuda.is_available() else 'cpu')
print(model)
# 测试一下

def bert_embed(text, max_length=256):
    # 对文本进行编码，并限制最大长度
    inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # 返回BERT模型的最后一层隐藏状态
    return outputs.last_hidden_state.squeeze(0)
def process_batch(data_batch, batch_index, save_dir, max_comments=5000):
    processed_batch_list = []

    # 对每个批次的数据进行处理
    for data in data_batch:
        # 计算内容的嵌入表示
        content_embedding = bert_embed(data['content']).to('cuda')
        #print(f'Content embedding shape: {content_embedding.shape}')

        # 检查评论数量，如果超过5000则仅使用前5000条
        comments = data['comments'][:max_comments]

        # 计算评论的嵌入表示
        comments_embeddings = torch.stack([bert_embed(comment).to('cuda') for comment in comments])
        #print(f'Comments embeddings shape: {comments_embeddings.shape}')

        # 计算平均池化和最大池化特征
        mean_pooling = torch.mean(comments_embeddings, dim=0)
        #print(f'Mean pooling shape: {mean_pooling.shape}')
        max_pooling = torch.max(comments_embeddings, dim=0).values
        #print(f'Max pooling shape: {max_pooling.shape}')

        # 计算语义差特征
        semantic_gap_mean = content_embedding - mean_pooling
        #print(f'Semantic gap mean shape: {semantic_gap_mean.shape}')
        semantic_gap_max = content_embedding - max_pooling
        #print(f'Semantic gap max shape: {semantic_gap_max.shape}')

        # 连接所有特征形成最终特征
        final_feature = torch.cat([content_embedding, mean_pooling, max_pooling, semantic_gap_mean, semantic_gap_max])
        #print(f'Final feature shape: {final_feature.shape}')
        
        processed_batch_list.append(final_feature)

    # 将处理后的批次数据保存到文件
    batch_file_name = f'batch_{batch_index}.npy'
    batch_file_path = os.path.join(save_dir, batch_file_name)

    # Move the stacked tensor to CPU before converting to NumPy
    np.save(batch_file_path, torch.stack(processed_batch_list).cpu().numpy())

    # 清理内存
    torch.cuda.empty_cache()
    del processed_batch_list
    return batch_file_name



def merge_batches(file_list, output_file_path):
    # 合并所有批次文件中的数据
    batch_data = [np.load(file) for file in file_list]
    merged_data = np.concatenate(batch_data, axis=0)
    # 将合并后的数据保存到一个文件
    np.save(output_file_path, merged_data)

# 设置批处理大小
batch_size = 1
save_dir = './data'
# 创建保存目录
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
datasets_ch = ['Weibo-20']
for dataset in datasets_ch:
    print(f'\n\n{"-"*20} [{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] Processing the dataset: {dataset} {"-"*20}\n')
    # 指定数据集目录
    data_dir = os.path.join('../../dataset', dataset)
    output_dir = os.path.join(save_dir, dataset)
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    processed_data_dir = os.path.join(output_dir, 'processed_bertwwm')
    # 创建处理后数据的保存目录
    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)
    # 加载数据集的训练、验证和测试部分
    split_datasets = {
        t: json.load(open(os.path.join(data_dir, f'{t}.json'), 'r', encoding='utf-8'))
        for t in ['train','test','val']
    }

    for split, data in split_datasets.items():
        # 为每个数据分割设置批处理文件目录
        batch_dir = os.path.join(processed_data_dir, split)
        if not os.path.exists(batch_dir):
            os.mkdir(batch_dir)

        file_list = []
        # 处理数据并保存为批次文件
        for batch_index in tqdm(range(0, len(data), batch_size), desc=f"Processing {split} dataset"):
            data_batch = data[batch_index:batch_index + batch_size]
            batch_file_name = process_batch(data_batch, batch_index // batch_size, batch_dir)
            file_list.append(os.path.join(batch_dir, batch_file_name))

        # 合并所有批次文件为一个单独的文件
        final_file_path = os.path.join(processed_data_dir, f'{split}.npy')
        merge_batches(file_list, final_file_path)
