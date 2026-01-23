import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.graph_recommender import GraphRecommender
from base.torch_interface import TorchGraphInterface
from data.augmentor import GraphAugmentor
from util.conf import OptionConf
from util.loss_torch import *
from util.sampler import *


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedReAvg(w, N_client):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, N_client):
            w_avg[k] += w[1][k]
        w_avg[k] = torch.div(w_avg[k], N_client)
    return w_avg


class ItemModule(nn.Module):
    def __init__(self, data, emb_size):
        self.data = data  # Interaction
        self.latent_size = emb_size  # 64
        super(ItemModule, self).__init__()
        initializer = nn.init.xavier_uniform_
        self.item_emebedding = nn.ParameterDict({
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),  # (7597, 64)
        })

    def forward(self):
        return self.item_emebedding


class FastPFRec(GraphRecommender):
    def __init__(self, conf, training_set, test_set, valid_set):
        """
        self.data: Interaction
        :param conf:
        :param training_set:
        :param test_set:
        :param valid_set:
        """
        super(FastPFRec, self).__init__(conf, training_set, test_set, valid_set)
        # 優先使用 CUDA，其次 Apple MPS，最後 CPU
        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"
        self.device = torch.device(device_name)

        print("Using device:", self.device)

        args = OptionConf(self.config['FastPFRec'])



        self.n_layers = int(args['-n_layer'])  # 2
        pretrain_noise = float(conf['pretrain_noise'])  # 0.1
        self.model = FastGNN_Encoder(
            self.data, self.emb_size, self.n_layers, pretrain_noise
        )  # emb_size: 6，对比学习
        self.msg += conf['training.set']
        self.dataset_name = conf['training.set']
        self.pretrain_epoch = conf['pretrain_epoch']  # 5
        self.noise_scale = float(conf['noise_scale'])  # 0.1
        self.clip_value = float(conf['clip_value'])  # 0.5
        self.pretrain_nclient = int(conf['pretrain_nclient'])  # 256
        self.trusted_nodes_num = int(conf['trusted_nodes_num'])
        # 异常检测相关参数（可选，如果配置文件中没有则使用默认值）
        self.anomaly_detection_enabled = conf['anomaly_detection_enabled'] if conf.contain('anomaly_detection_enabled') else True  # 默认启用

        self.anomaly_ratio_threshold = (
            float(conf['anomaly_ratio_threshold'])
            if conf.contain('anomaly_ratio_threshold')
            else 0.2  # 默认值
        )
        self.anomaly_dist_threshold = (
            float(conf['anomaly_dist_threshold'])
            if conf.contain('anomaly_dist_threshold')
            else 3.5  # 默认值
        )
        self.msg += ('pretrain_epoch:' + conf['pretrain_epoch'] + '\n')
        self.msg += ('noise_scale:' + (conf['noise_scale']) + '\n')
        self.msg += ('clip_value:' + (conf['clip_value']) + '\n')
        self.msg += ('pretrain_noise:' + (conf['pretrain_noise']) + '\n')
        self.msg += ('pretrain_nclient:' + (conf['pretrain_nclient']) + '\n')
        self.item_global_model = ItemModule(self.data, self.emb_size)
        print(self.msg)

    def evenly_split_list(self, lst, num_sublists: int):
        """
        将一个列表均匀分成指定数量的子列表，用于构造 trusted nodes。
        """
        sublists = [[] for _ in range(num_sublists)]
        for i, item in enumerate(lst):
            sublists[i % num_sublists].append(item)
        return sublists

    def _detect_node_anomaly(self, node_client_params, reference_params, 
                             ratio_threshold=None, dist_threshold=None):
        """
        基于稳健统计的离群检测，用于检测 trusted node 内的异常客户端参数。
        
        Args:
            node_client_params: List[Dict[str, torch.Tensor]]，该节点内所有客户端的参数差异列表
            reference_params: Dict[str, torch.Tensor]，参考模型参数（通常是 global model）
            ratio_threshold: float，离群比例阈值，默认使用 self.anomaly_ratio_threshold
            dist_threshold: float，距离阈值，默认使用 self.anomaly_dist_threshold
            
        Returns:
            bool: True 表示检测到异常，False 表示正常
        """
        if not node_client_params or len(node_client_params) == 0:
            return False
        
        if ratio_threshold is None:
            ratio_threshold = self.anomaly_ratio_threshold
        if dist_threshold is None:
            dist_threshold = self.anomaly_dist_threshold
        
        with torch.no_grad():
            # 计算每个客户端参数与参考模型的距离
            dists = []
            for client_params in node_client_params:
                # 计算所有参数键的归一化 L2 距离，然后取平均
                # 这样可以避免大参数主导距离计算
                normalized_dists = []
                for key in reference_params:
                    if key in client_params:
                        param_diff = client_params[key] - reference_params[key]
                        # 计算相对距离：L2范数 / (参考参数范数 + epsilon)
                        ref_norm = torch.norm(reference_params[key]).item() + 1e-8
                        dist = torch.norm(param_diff).item() / ref_norm
                        normalized_dists.append(dist)
                
                # 如果没有任何匹配的键，跳过该客户端
                if len(normalized_dists) == 0:
                    continue
                
                # 使用平均归一化距离
                total_dist = sum(normalized_dists) / len(normalized_dists)
                dists.append(total_dist)
            
            if len(dists) == 0:
                return False
            
            dists = torch.tensor(dists)
            
            # 使用稳健统计量（median 和 MAD）进行标准化
            median = torch.median(dists)
            mad = torch.median(torch.abs(dists - median)) + 1e-6
            robust_z = 0.6745 * (dists - median) / mad
            
            # 计算离群比例
            outlier_ratio = (torch.abs(robust_z) > dist_threshold).float().mean().item()
            
            return outlier_ratio > ratio_threshold

    def pre_training(self, model):
        """
        做对比学习
        :return:
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate * 50)
        self.msg += '\npretrain\n'
        for epoch in range(int(self.pretrain_epoch)):  # 5
            user_list = list(self.data.user.keys())
            random.shuffle(user_list)  # shuffle 取前256个
            select_user_list = user_list[:self.pretrain_nclient]
            not_select_user_list = user_list[self.pretrain_nclient:]
            select_user_list_num = [self.data.user[_] for _ in select_user_list]  # [user_id]
            not_select_user_list_num = [self.data.user[_] for _ in not_select_user_list]  # [user_id]
            self.cl_rate = 1
            # 计算对比学习损失 ☆
            cl_loss = self.cl_rate * self.cal_cl_loss(self.data)
            optimizer.zero_grad()
            cl_loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.user_emb, self.item_emb = model.get_emb()
            self.fast_evaluation(epoch)  # 评价，global model

    def _aggregate_with_trusted_nodes(self, original_params, noisy_diff):
        """
        使用 trusted nodes 进行两层聚合，包含异常检测和隔离功能：
        1）客户端 -> trusted nodes（包含异常检测）
        2）trusted nodes -> 服务器（只聚合正常节点）
        
        Args:
            original_params: Dict[str, torch.Tensor]，原始全局模型参数
            noisy_diff: List[Dict[str, torch.Tensor]]，所有客户端的参数差异列表
            
        Returns:
            Dict[str, torch.Tensor]: 聚合后的全局模型参数
        """
        trusted_nodes_num = self.trusted_nodes_num
        trusted_nodes = self.evenly_split_list(noisy_diff, trusted_nodes_num)

        # 第 1 层：每个 trusted node 内部先做一次 FedAvg，并进行异常检测
        final_state_dict_li = [{} for _ in range(trusted_nodes_num)]
        isolated_node_indices = []  # 记录被隔离的节点索引
        
        for i in range(trusted_nodes_num):
            clients_on_trusted_node = trusted_nodes[i]
            if len(clients_on_trusted_node) == 0:
                continue
            
            # 异常检测：如果启用且检测到异常，则隔离该节点
            if self.anomaly_detection_enabled:
                is_anomalous = self._detect_node_anomaly(
                    node_client_params=clients_on_trusted_node,
                    reference_params=original_params,
                    ratio_threshold=self.anomaly_ratio_threshold,
                    dist_threshold=self.anomaly_dist_threshold
                )
                
                if is_anomalous:
                    isolated_node_indices.append(i)
                    if len(isolated_node_indices) <= 3:  # 只打印前3个，避免日志过多
                        print(f"Warning: Trusted node {i} detected as anomalous and isolated. "
                              f"Affected clients: {len(clients_on_trusted_node)}")
                    continue  # 跳过该节点，不参与聚合
            
            # 正常节点：进行聚合
            for key in original_params:
                combined = torch.sum(
                    torch.stack([diff[key] for diff in clients_on_trusted_node], dim=0),
                    dim=0
                ) / len(clients_on_trusted_node)
                final_state_dict_li[i][key] = combined

        # 第 2 层：trusted nodes 之間再做一次 FedAvg（只使用正常节点）
        final_state_dict = {}
        for key in original_params:
            # 只选择未被隔离的节点
            valid_nodes = []
            for idx, node in enumerate(final_state_dict_li):
                if idx not in isolated_node_indices and key in node:
                    valid_nodes.append(node[key])
            
            if len(valid_nodes) == 0:
                # 如果所有节点都被隔离，使用原始参数
                final_state_dict[key] = original_params[key]
            else:
                # 聚合正常节点的参数
                combined = torch.sum(
                    torch.stack(valid_nodes, dim=0),
                    dim=0
                ) / len(valid_nodes)
                final_state_dict[key] = combined
        
        if self.anomaly_detection_enabled and len(isolated_node_indices) > 0:
            total_clients_affected = sum(len(trusted_nodes[i]) for i in isolated_node_indices)
            print(f"Anomaly detection summary: {len(isolated_node_indices)}/{trusted_nodes_num} "
                  f"nodes isolated, {total_clients_affected} clients affected")
        
        return final_state_dict

    def train(self):
        model = self.model.to(self.device)
        model_para_list = []
        N_client = 256
        self.N_client = N_client  # 256
        loc, scale = 0., 0.1
        scale = self.noise_scale
        clip_value = self.clip_value
        self.local_model = None

        ######## 1. 自监督学习预训练 #######
        Pretraining = int(self.pretrain_epoch) > 0
        if Pretraining:
            self.pre_training(model)

        ######## 2. 联邦学习推荐训练 #######
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lRate * N_client
        )
        self.loss_list = []
        self.ndcg_list = []

        for epoch in range(self.maxEpoch):  # 500
            original_params = copy.deepcopy(model.state_dict())  # 原始的global model
            self.local_model = {}
            losses = []
            if epoch == 0:
                user_list = list(self.data.user.keys())  # [user_id]
                random.shuffle(user_list)
                select_user_list = user_list[:N_client]  # 每次选256个做训练数据
                not_select_user_list = user_list[N_client:]
            else:
                select_user_list = self.select_user_list
                not_select_user_list = self.not_select_user_list
            select_user_list_num = [self.data.user[_] for _ in select_user_list]
            not_select_user_list_num = [self.data.user[_] for _ in not_select_user_list]

            dropped_adj, dropped_adj_ten = self.get_client_mat(not_select_user_list_num)  # user-item matrix


            for n, batch in enumerate(next_batch_pairwise_fl_pse(self.data, self.batch_size, select_user_list)):
                """
                batch: [users] [pos_items] [neg_items]
                """
                model_ini = copy.deepcopy(model.state_dict())  # 保存没有训练过的模型
                user_idx, pos_idx, neg_idx = batch

                if epoch > 0 and epoch % 10 == 0:  # 每经过10个epoch进行用户和物品的完全卷积
                    rec_user_emb, rec_item_emb = model(perturbed=False, pretraining=True) 
                else:
                    rec_user_emb, rec_item_emb = model(perturbed=False, pretraining=False)  

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]  # user_emb是全部相同的

                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(
                    self.reg, user_emb,
                    pos_item_emb,
                    neg_item_emb
                ) / self.batch_size  # 256
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 100 == 0 and n > 0:  #
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
                losses.append(batch_loss.item())

                model_aft = copy.deepcopy(model.state_dict())  # 训练过的
                model_para_list += [model_aft]

                self.local_model[user_idx[0]] = model_aft  # 保存local model

                model.load_state_dict(model_ini)  # 清空模型

            print('Avg Loss:', sum(losses) / len(losses))
            self.loss_list.append(sum(losses) / len(losses))
            model_params_list = model_para_list
            params_diff = []
            for state_dict in model_params_list:  # 一轮训练过的模型
                diff = {}
                for key in original_params:
                    diff[key] = state_dict[key] - original_params[key]
                    diff[key] = torch.clamp(diff[key], min=-clip_value, max=clip_value)  # 阈值
                    diff[key] = diff[key] + original_params[key]
                params_diff.append(diff)  # 相当于限制了local和global之间的diff，得到信的local
            noisy_diff = params_diff
            ######## 3. Trusted Nodes + FedAvg 得到 global 模型 #######
            final_state_dict = self._aggregate_with_trusted_nodes(original_params, noisy_diff)
            model.load_state_dict(final_state_dict)  # 最终的 global 模型

            model_para_list = []
            ######## 4. LDP #######
            add_noise = True
            if add_noise:
                i_random_noise = torch.tensor(np.random.laplace(loc=loc, scale=scale, size=(
                    N_client, rec_item_emb.shape[0], rec_item_emb.shape[1])))
                i_random_noise = torch.mean(i_random_noise, dim=0).float().to(self.device)
                model.add_noise_(i_random_noise)

            with torch.no_grad():
                self.user_emb, self.item_emb = model.get_emb()
            if epoch > 0 and epoch % 5 == 0:  # 每5轮measure一次
                print('########################### evaluate global_model ###########################')
                measure = self.fast_evaluation(epoch)
                print('########################### evaluate global_model done~###########################')
                measure_ndcg = measure[-1].split(':')[-1]
                self.ndcg_list.append(measure_ndcg)

                print('########################### evaluate local_model ###########################')
                self.fast_evaluation(epoch, model_type='local_model')
                print('########################### evaluate local_model done~###########################')



            user_candidate_list = list(set(self.data.user.keys()) - set(select_user_list))
            random.shuffle(user_candidate_list)
            select_user_list = user_candidate_list[:N_client]
            not_select_user_list = list(set(self.data.user.keys()) - set(select_user_list))
            self.select_user_list = select_user_list
            self.not_select_user_list = not_select_user_list

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = copy.deepcopy(self.model.get_emb())
            self.best_local_model = copy.deepcopy(self.local_model)

    def get_client_mat(self, drop_client_list):
        """
        获取客户端的交互矩阵
        :param drop_client_list:
        :return:
        """
        dropped_mat = None
        dropped_mat_ = GraphAugmentor.client_select_drop(self.data.interaction_mat, drop_client_list)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat_)
        return dropped_mat_, TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).to(self.device)

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()

    def predict_local(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            if u in self.local_model:
                user_emb = self.local_model[u]['embedding_dict.user_emb']
                item_emb = self.local_model[u]['embedding_dict.item_emb']
                score = torch.matmul(user_emb[u], item_emb.transpose(0, 1))
                return score.cpu().numpy()
            else:
                return None

    def cal_cl_loss(self, idx):
        """
        计算对比学习损失 ☆
        :param idx:
        :return:
        """
        # 1. 选出对比学习的user
        cl_sampple = self.N_client
        user_list = list(self.data.user.keys())
        random.shuffle(user_list)
        select_user_list = user_list[:cl_sampple]
        select_user_list_num = [self.data.user[_] for _ in select_user_list]
        item_num = idx.item_num
        rand_item_num = random.sample([_ for _ in range(item_num)], cl_sampple)
        u_idx = torch.unique(torch.Tensor(select_user_list_num).type(torch.long)).to(self.device)  # 选一些
        i_idx = torch.unique(torch.Tensor(rand_item_num).type(torch.long)).to(self.device)
        # 2. 前向传播计算损失
        user_view_1, item_view_1 = self.model(perturbed=True,
                                              pretraining=True)  # torch.Size([5224, 64]) torch.Size([7597, 64])
        user_view_2, item_view_2 = self.model(perturbed=True,
                                              pretraining=True)  # torch.Size([5224, 64]) torch.Size([7597, 64])
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)  # user_emb 对比损失
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)  # item_emb 对比损失
        return user_cl_loss + item_cl_loss

    def contrastive_augment(self, _mat):
        self.drop_rate = 0.1
        dropped_mat = None
        dropped_mat = GraphAugmentor.node_dropout(_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()


class FastGNN_Encoder(nn.Module):
    """
    local GCN encoder
    """

    def __init__(self, data, emb_size, n_layers, pretrain_noise):
        super(FastGNN_Encoder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"

        self.data = data  # Interaction
        self.latent_size = emb_size  # 64
        self.layers = n_layers  # 2
        # user-item 二部图的归一化邻接矩阵（用于完整 GCN）
        self.norm_adj = data.norm_adj
        self.pretrain_noise = float(pretrain_noise)  # 0.1

        self.embedding_dict = self._init_model()  # {'user_emb': , 'item_emb': ]}
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(self.device)

        # 基于交互矩阵构造 user-user 图：
        # A_uu = A_ui * A_ui^T，然后做对称归一化，得到 (U, U) 的稀疏矩阵
        ui_mat = self.data.interaction_mat  # (|U|, |I|) 稀疏矩阵
        uu_mat = ui_mat * ui_mat.T          # (|U|, |U|)
        try:
            uu_mat.setdiag(0)              # 去掉 self-loop，防止度数过大
        except Exception:
            pass
        uu_mat.eliminate_zeros()
        uu_norm = self.data.normalize_graph_mat(uu_mat)
        self.sparse_uu_norm = TorchGraphInterface.convert_sparse_mat_to_tensor(uu_norm).to(self.device)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),  # (5224, 64)
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),  # (7597, 64)
        })
        return embedding_dict

    def get_emb(self):
        return self.embedding_dict['user_emb'].data, self.embedding_dict['item_emb'].data

    def add_noise_(self, noise):
        self.embedding_dict['item_emb'].data = self.embedding_dict['item_emb'].data + noise

    def forward(self, perturbed=False, perturbed_adj=None, pretraining=False):
        self.eps = self.pretrain_noise  # 0.1
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.layers):  # 2 times
            if perturbed_adj is not None:
                if isinstance(perturbed_adj, list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)  # (12821, 12821) * (12821, 64)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj,
                                                 ego_embeddings)  # (12821, 12821) * (12821, 64)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).to(self.device)
                ego_embeddings += F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)  # 两层做平均
        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.data.user_num, self.data.item_num]
        )

        # 根据pretraining标志决定item embedding的返回方式
        if not pretraining:
            item_all_embeddings = self.embedding_dict['item_emb']  # 模型的新改进：使用原始item embedding
        return user_all_embeddings, item_all_embeddings

    def forward_uu(self, perturbed=False, perturbed_adj=None, pretraining=False):
        """
        for abalation
        只更新 user_emb 时，使用 user-user 图卷积：
        - 图：A_uu = A_ui A_ui^T 归一化后的 (U, U) 矩阵
        - 传播：H 层 A_uu * user_emb
        """
        self.eps = self.pretrain_noise  # 0.1
        # 情况 1：pretraining=True，完整 GCN（u-i + i-u），更新 user 和 item
        if pretraining:
            ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
            all_embeddings = []
            for k in range(self.layers):  # 2 次传播
                if perturbed_adj is not None:
                    if isinstance(perturbed_adj, list):
                        ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                    else:
                        ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
                if perturbed:
                    random_noise = torch.rand_like(ego_embeddings).to(self.device)
                    ego_embeddings += F.normalize(random_noise, dim=-1) * self.eps
                all_embeddings.append(ego_embeddings)
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = torch.mean(all_embeddings, dim=1)
            user_all_embeddings, item_all_embeddings = torch.split(
                all_embeddings, [self.data.user_num, self.data.item_num]
            )
            return user_all_embeddings, item_all_embeddings

        # 情况 2：只更新 user_emb 时，使用 user-user 图卷积：
        # - 图：A_uu = A_ui A_ui^T 归一化后的 (U, U) 矩阵
        # - 传播：H 层 A_uu * user_emb
        ego_user = self.embedding_dict['user_emb']
        all_user_embeddings = []
        for k in range(self.layers):
            ego_user = torch.sparse.mm(self.sparse_uu_norm, ego_user)  # (U, U) * (U, d)
            if perturbed:
                random_noise = torch.rand_like(ego_user).to(self.device)
                ego_user += F.normalize(random_noise, dim=-1) * self.eps
            all_user_embeddings.append(ego_user)

        all_user_embeddings = torch.stack(all_user_embeddings, dim=1)
        user_all_embeddings = torch.mean(all_user_embeddings, dim=1)
        item_all_embeddings = self.embedding_dict['item_emb']  # item 不经过 GCN 更新
        return user_all_embeddings, item_all_embeddings