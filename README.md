# FastPFRec: A Fast Personalized Federated Recommendation with Secure Sharing

FastPFRec is a graph-based personalized federated recommender, featuring two-level aggregation via trusted nodes and a FastGNN encoder for more secure model sharing.

## Quick Start

Train FastPFRec (example: yelp):

```bash
python main.py --model FastPFRec --dataset=yelp 
```

Common arguments:
- `--model`: model name (default: `FastPFRec`)
- `--dataset`: `kindle/yelp/gowalla/gowalla_real` (internally mapped to `*_test`)
- `--emb`: embedding dimension (default: `64`)
- `--trusted_nodes_num`: number of trusted nodes (default: `10`)

Logs are written to `./logs/`.

## Datasets

Default dataset layout:
- `./dataset/<dataset_name>/train.txt`
- `./dataset/<dataset_name>/valid.txt`
- `./dataset/<dataset_name>/test.txt`

`--dataset=yelp` is mapped to `./dataset/yelp_test/` (same for other datasets).
