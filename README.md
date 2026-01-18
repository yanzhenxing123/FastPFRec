# FastPFRec: A Fast Personalized Federated Recommendation with Secure Sharing

FastPFRec 是一个面向图推荐的个性化联邦推荐实现，包含基于 trusted nodes 的两层聚合与FastGNN，用于实现更安全的参数共享。

## Quick Start

训练 FastPFRec（以 yelp 为例）：

```bash
python main.py --model FastPFRec --dataset=yelp 
```

常用参数：
- `--model`: 模型名（默认 `FastPFRec`）
- `--dataset`: `kindle/yelp/gowalla/gowalla_real`（内部会映射到 `*_test` 目录）
- `--emb`: embedding 维度（默认 `64`）
- `--trusted_nodes_num`: trusted nodes 数量（默认 `10`）

运行后日志会写入 `./logs/`。

## Datasets

默认数据路径格式：
- `./dataset/<dataset_name>/train.txt`
- `./dataset/<dataset_name>/valid.txt`
- `./dataset/<dataset_name>/test.txt`

其中 `--dataset=yelp` 会映射到 `./dataset/yelp_test/`（其他 dataset 同理）。
