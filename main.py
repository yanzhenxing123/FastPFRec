import argparse
import os
import sys
import time

from SELFRec import SELFRec
from util.conf import ModelConf


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


if __name__ == '__main__':

    graph_baselines = ['LightGCN', 
                       'MF', 
                       'FedGNN',
                        'FedMF', 
                        'PerFedRec', 
                        'PerFedRec_plus',
                       "FastPFRec"]
    ssl_graph_models = ['SGL', 'SimGCL', 'XSimGCL']  # Self-Supervised
    sequential_baselines = []
    ssl_sequential_models = []

    print('=' * 80)
    print('Graph-Based Baseline Models:')
    print('   '.join(graph_baselines))
    print('-' * 100)
    print('Self-Supervised  Graph-Based Models:')
    print('   '.join(ssl_graph_models))
    print('=' * 80)
    print('Sequential Baseline Models:')
    print('   '.join(sequential_baselines))
    print('-' * 100)
    print('Self-Supervised Sequential Models:')
    print('   '.join(ssl_sequential_models))
    print('=' * 80)

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model', type=str, default='FastPFRec')
    parser.add_argument("--dataset", type=str, default='yelp', help='kindle/yelp/gowalla/gowalla_real')
    parser.add_argument('--emb', type=str, default='64')  # emb_size
    parser.add_argument('--pretrain_epoch', type=str, default='5')
    parser.add_argument('--noise_scale', type=str, default='0.1')
    parser.add_argument('--clip_value', type=str, default='0.5')
    parser.add_argument('--pretrain_noise', type=str, default='0.1')
    parser.add_argument('--pretrain_nclient', type=str, default='256')
    parser.add_argument('--trusted_nodes_num', type=str, default='10',
                        help='Number of trusted nodes for federated aggregation (only used by FastPFRec / secure variants)')
    args = parser.parse_args()

    # log file
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_path = os.path.join(log_dir, f'{args.model}_{args.dataset}_{timestamp}.log')
    log_file = open(log_path, 'a', buffering=1)

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    model = args.model

    print(f"Using model {model}")

    s = time.time()
    if model in graph_baselines or model in ssl_graph_models or model in sequential_baselines or model in ssl_sequential_models:
        conf = ModelConf('./conf/' + model + '.conf')
    else:
        print('Wrong model name!')
        exit(-1)

    if args.dataset == 'kindle':
        args.dataset = 'kindle_test'
    elif args.dataset == 'yelp':
        args.dataset = 'yelp_test'
    elif args.dataset == 'gowalla':
        args.dataset = 'gowalla_test'
    elif args.dataset == 'gowalla_real':
        args.dataset = 'gowalla'

    conf.__setitem__('training.set', f'./dataset/{args.dataset}/train.txt')
    conf.__setitem__('valid.set', f'./dataset/{args.dataset}/valid.txt')
    conf.__setitem__('test.set', f'./dataset/{args.dataset}/test.txt')
    conf.__setitem__('embedding.size', args.emb)
    conf.__setitem__('noise_scale', args.noise_scale)
    conf.__setitem__('clip_value', args.clip_value)
    conf.__setitem__('pretrain_noise', args.pretrain_noise)
    conf.__setitem__('pretrain_nclient', args.pretrain_nclient)
    conf.__setitem__('pretrain_epoch', args.pretrain_epoch)
    conf.__setitem__('trusted_nodes_num', args.trusted_nodes_num)

    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
