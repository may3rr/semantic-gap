import os
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import json
from tqdm.notebook import tqdm
import time

# 初始化 BERT 分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese').to('cuda' if torch.cuda.is_available() else 'cpu')

def bert_embed(text, max_length=256): 
    inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.squeeze(0)


def process_batch(data_batch, batch_index, save_dir, max_comments=5000):
    processed_batch_list = []

    for data in data_batch:
        content_embedding = bert_embed(data['content']).to('cuda')

        comments = data['comments'][:max_comments]
        comments_embeddings = torch.stack([bert_embed(comment).to('cuda') for comment in comments])

        mean_pooling = torch.mean(comments_embeddings, dim=0)
        max_pooling = torch.max(comments_embeddings, dim=0).values

        semantic_gap_mean = content_embedding - mean_pooling
        semantic_gap_max = content_embedding - max_pooling

        final_feature = torch.cat([content_embedding, mean_pooling, max_pooling, semantic_gap_mean, semantic_gap_max])
        processed_batch_list.append(final_feature)

    batch_file_name = f'batch_{batch_index}.npy'
    batch_file_path = os.path.join(save_dir, batch_file_name)

    np.save(batch_file_path, torch.stack(processed_batch_list).cpu().numpy())

    torch.cuda.empty_cache()
    del processed_batch_list
    return batch_file_name


def merge_batches(file_list, output_file_path):
    batch_data = [np.load(file) for file in file_list]
    merged_data = np.concatenate(batch_data, axis=0)
    np.save(output_file_path, merged_data)

batch_size = 1
save_dir = './data'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

datasets_ch = ['Weibo-20']
for dataset in datasets_ch:
    print(f'\n\n{"-"*20} [{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] Processing the dataset: {dataset} {"-"*20}\n')
    data_dir = os.path.join('../../dataset', dataset)
    output_dir = os.path.join(save_dir, dataset)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    processed_data_dir = os.path.join(output_dir, 'processed_new')
    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)

    split_datasets = {
        t: json.load(open(os.path.join(data_dir, f'{t}.json'), 'r', encoding='utf-8'))
        for t in ['train','test','val']
    }

    for split, data in split_datasets.items():
        batch_dir = os.path.join(processed_data_dir, split)
        if not os.path.exists(batch_dir):
            os.mkdir(batch_dir)

        file_list = []
        for batch_index in tqdm(range(0, len(data), batch_size), desc=f"Processing {split} dataset"):
            data_batch = data[batch_index:batch_index + batch_size]
            batch_file_name = process_batch(data_batch, batch_index // batch_size, batch_dir)
            file_list.append(os.path.join(batch_dir, batch_file_name))

        final_file_path = os.path.join(processed_data_dir, f'{split}.npy')
        merge_batches(file_list, final_file_path)