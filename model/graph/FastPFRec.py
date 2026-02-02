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

class ItemModule(nn.Module):
    def __init__(self, data, emb_size):
        self.data = data
        self.latent_size = emb_size
        super(ItemModule, self).__init__()
        initializer = nn.init.xavier_uniform_
        self.item_emebedding = nn.ParameterDict({
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })

    def forward(self):
        return self.item_emebedding


class FastPFRec(GraphRecommender):
    def __init__(self, conf, training_set, test_set, valid_set):
        """FastPFRec recommender."""
        super(FastPFRec, self).__init__(conf, training_set, test_set, valid_set)
        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"
        self.device = torch.device(device_name)

        print("Using device:", self.device)

        args = OptionConf(self.config['FastPFRec'])



        self.n_layers = int(args['-n_layer'])
        pretrain_noise = float(conf['pretrain_noise'])
        self.model = FastGNN_Encoder(
            self.data, self.emb_size, self.n_layers, pretrain_noise
        )
        self.msg += conf['training.set']
        self.dataset_name = conf['training.set']
        self.pretrain_epoch = conf['pretrain_epoch']
        self.noise_scale = float(conf['noise_scale'])
        self.clip_value = float(conf['clip_value'])
        self.pretrain_nclient = int(conf['pretrain_nclient'])
        self.trusted_nodes_num = int(conf['trusted_nodes_num'])
        self.anomaly_detection_enabled = conf['anomaly_detection_enabled'] if conf.contain('anomaly_detection_enabled') else True
        self.local_global_blend_alpha = (
            float(conf['local_global_blend_alpha'])
            if conf.contain('local_global_blend_alpha')
            else 0
        )

        self.anomaly_ratio_threshold = (
            float(conf['anomaly_ratio_threshold'])
            if conf.contain('anomaly_ratio_threshold')
            else 0.2
        )
        self.anomaly_dist_threshold = (
            float(conf['anomaly_dist_threshold'])
            if conf.contain('anomaly_dist_threshold')
            else 3.5
        )
        self.msg += ('pretrain_epoch:' + conf['pretrain_epoch'] + '\n')
        self.msg += ('noise_scale:' + (conf['noise_scale']) + '\n')
        self.msg += ('clip_value:' + (conf['clip_value']) + '\n')
        self.msg += ('pretrain_noise:' + (conf['pretrain_noise']) + '\n')
        self.msg += ('pretrain_nclient:' + (conf['pretrain_nclient']) + '\n')
        self.item_global_model = ItemModule(self.data, self.emb_size)
        print(self.msg)

    def evenly_split_list(self, lst, num_sublists: int):
        """Split a list into evenly sized sublists."""
        sublists = [[] for _ in range(num_sublists)]
        for i, item in enumerate(lst):
            sublists[i % num_sublists].append(item)
        return sublists

    def _detect_node_anomaly(self, node_client_params, reference_params, 
                             ratio_threshold=None, dist_threshold=None):
        """Detect anomalies within a trusted node."""
        if not node_client_params or len(node_client_params) == 0:
            return False
        
        if ratio_threshold is None:
            ratio_threshold = self.anomaly_ratio_threshold
        if dist_threshold is None:
            dist_threshold = self.anomaly_dist_threshold
        
        with torch.no_grad():
            dists = []
            for client_params in node_client_params:
                normalized_dists = []
                for key in reference_params:
                    if key in client_params:
                        param_diff = client_params[key] - reference_params[key]
                        ref_norm = torch.norm(reference_params[key]).item() + 1e-8
                        dist = torch.norm(param_diff).item() / ref_norm
                        normalized_dists.append(dist)
                
                if len(normalized_dists) == 0:
                    continue
                
                total_dist = sum(normalized_dists) / len(normalized_dists)
                dists.append(total_dist)
            
            if len(dists) == 0:
                return False
            
            dists = torch.tensor(dists)
            
            median = torch.median(dists)
            mad = torch.median(torch.abs(dists - median)) + 1e-6
            robust_z = 0.6745 * (dists - median) / mad
            
            outlier_ratio = (torch.abs(robust_z) > dist_threshold).float().mean().item()
            
            return outlier_ratio > ratio_threshold

    def pre_training(self, model):
        """Run contrastive pretraining."""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate * 50)
        self.msg += '\npretrain\n'
        for epoch in range(int(self.pretrain_epoch)):
            user_list = list(self.data.user.keys())
            random.shuffle(user_list)
            select_user_list = user_list[:self.pretrain_nclient]
            not_select_user_list = user_list[self.pretrain_nclient:]
            select_user_list_num = [self.data.user[_] for _ in select_user_list]
            not_select_user_list_num = [self.data.user[_] for _ in not_select_user_list]
            self.cl_rate = 1
            cl_loss = self.cl_rate * self.cal_cl_loss(self.data)
            optimizer.zero_grad()
            cl_loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.user_emb, self.item_emb = model.get_emb()
            self.fast_evaluation(epoch)

    def _aggregate_with_trusted_nodes(self, original_params, noisy_diff):
        """Aggregate with trusted nodes and anomaly isolation."""
        trusted_nodes_num = self.trusted_nodes_num
        trusted_nodes = self.evenly_split_list(noisy_diff, trusted_nodes_num)

        final_state_dict_li = [{} for _ in range(trusted_nodes_num)]
        isolated_node_indices = []
        
        for i in range(trusted_nodes_num):
            clients_on_trusted_node = trusted_nodes[i]
            if len(clients_on_trusted_node) == 0:
                continue
            
            if self.anomaly_detection_enabled:
                is_anomalous = self._detect_node_anomaly(
                    node_client_params=clients_on_trusted_node,
                    reference_params=original_params,
                    ratio_threshold=self.anomaly_ratio_threshold,
                    dist_threshold=self.anomaly_dist_threshold
                )
                
                if is_anomalous:
                    isolated_node_indices.append(i)
                    if len(isolated_node_indices) <= 3:
                        print(f"Warning: Trusted node {i} detected as anomalous and isolated. "
                              f"Affected clients: {len(clients_on_trusted_node)}")
                    continue
            
            for key in original_params:
                combined = torch.sum(
                    torch.stack([diff[key] for diff in clients_on_trusted_node], dim=0),
                    dim=0
                ) / len(clients_on_trusted_node)
                final_state_dict_li[i][key] = combined

        final_state_dict = {}
        for key in original_params:
            valid_nodes = []
            for idx, node in enumerate(final_state_dict_li):
                if idx not in isolated_node_indices and key in node:
                    valid_nodes.append(node[key])
            
            if len(valid_nodes) == 0:
                final_state_dict[key] = original_params[key]
            else:
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
        self.N_client = N_client
        loc, scale = 0., 0.1
        scale = self.noise_scale
        clip_value = self.clip_value
        self.local_model = None
        if not hasattr(self, 'local_model_cache'):
            self.local_model_cache = {}

        Pretraining = int(self.pretrain_epoch) > 0
        if Pretraining:
            self.pre_training(model)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lRate * N_client
        )
        self.loss_list = []
        self.ndcg_list = []

        for epoch in range(self.maxEpoch):
            original_params = copy.deepcopy(model.state_dict())
            self.local_model = {}
            losses = []
            if epoch == 0:
                user_list = list(self.data.user.keys())
                random.shuffle(user_list)
                select_user_list = user_list[:N_client]
                not_select_user_list = user_list[N_client:]
            else:
                select_user_list = self.select_user_list
                not_select_user_list = self.not_select_user_list
            select_user_list_num = [self.data.user[_] for _ in select_user_list]
            not_select_user_list_num = [self.data.user[_] for _ in not_select_user_list]

            dropped_adj, dropped_adj_ten = self.get_client_mat(not_select_user_list_num)


            for n, batch in enumerate(next_batch_pairwise_fl_pse(self.data, self.batch_size, select_user_list)):
                """batch: [users] [pos_items] [neg_items]"""
                
                model_ini = copy.deepcopy(model.state_dict())
                user_idx, pos_idx, neg_idx = batch
                user_id = user_idx[0]
                if self.local_global_blend_alpha > 0 and user_id in self.local_model_cache:
                    local_state = self.local_model_cache[user_id]
                    alpha = self.local_global_blend_alpha
                    blended_state = {}
                    for key in model_ini:
                        if key in local_state:
                            blended_state[key] = alpha * local_state[key] + (1 - alpha) * model_ini[key]
                        else:
                            blended_state[key] = model_ini[key]
                    model.load_state_dict(blended_state)

                if epoch > 0 and epoch % 10 == 0:
                    rec_user_emb, rec_item_emb = model(perturbed=False, pretraining=True) 
                else:
                    rec_user_emb, rec_item_emb = model(perturbed=False, pretraining=False)  

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]

                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(
                    self.reg, user_emb,
                    pos_item_emb,
                    neg_item_emb
                ) / self.batch_size
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
                losses.append(batch_loss.item())

                model_aft = copy.deepcopy(model.state_dict())
                model_para_list += [model_aft]

                self.local_model[user_id] = model_aft
                self.local_model_cache[user_id] = model_aft

                model.load_state_dict(model_ini)

            print('Avg Loss:', sum(losses) / len(losses))
            self.loss_list.append(sum(losses) / len(losses))
            model_params_list = model_para_list
            params_diff = []
            for state_dict in model_params_list:
                diff = {}
                for key in original_params:
                    diff[key] = state_dict[key] - original_params[key]
                    diff[key] = torch.clamp(diff[key], min=-clip_value, max=clip_value)
                    diff[key] = diff[key] + original_params[key]
                params_diff.append(diff)
            noisy_diff = params_diff
            final_state_dict = self._aggregate_with_trusted_nodes(original_params, noisy_diff)
            model.load_state_dict(final_state_dict)

            model_para_list = []
            add_noise = True
            if add_noise:
                i_random_noise = torch.tensor(np.random.laplace(loc=loc, scale=scale, size=(
                    N_client, rec_item_emb.shape[0], rec_item_emb.shape[1])))
                i_random_noise = torch.mean(i_random_noise, dim=0).float().to(self.device)
                model.add_noise_(i_random_noise)

            with torch.no_grad():
                self.user_emb, self.item_emb = model.get_emb()
            if epoch > 0 and epoch % 5 == 0:
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
        """Return a dropped interaction matrix for selected clients."""
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
        """Compute the contrastive loss."""
        cl_sampple = self.N_client
        user_list = list(self.data.user.keys())
        random.shuffle(user_list)
        select_user_list = user_list[:cl_sampple]
        select_user_list_num = [self.data.user[_] for _ in select_user_list]
        item_num = idx.item_num
        rand_item_num = random.sample([_ for _ in range(item_num)], cl_sampple)
        u_idx = torch.unique(torch.Tensor(select_user_list_num).type(torch.long)).to(self.device)
        i_idx = torch.unique(torch.Tensor(rand_item_num).type(torch.long)).to(self.device)
        user_view_1, item_view_1 = self.model(perturbed=True,
                                              pretraining=True)
        user_view_2, item_view_2 = self.model(perturbed=True,
                                              pretraining=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        return user_cl_loss + item_cl_loss

    def contrastive_augment(self, _mat):
        self.drop_rate = 0.1
        dropped_mat = None
        dropped_mat = GraphAugmentor.node_dropout(_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()


class FastGNN_Encoder(nn.Module):
    """Local GCN encoder."""

    def __init__(self, data, emb_size, n_layers, pretrain_noise):
        super(FastGNN_Encoder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"

        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.pretrain_noise = float(pretrain_noise)

        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(self.device)

        ui_mat = self.data.interaction_mat
        uu_mat = ui_mat * ui_mat.T
        try:
            uu_mat.setdiag(0)
        except Exception:
            pass
        uu_mat.eliminate_zeros()
        uu_norm = self.data.normalize_graph_mat(uu_mat)
        self.sparse_uu_norm = TorchGraphInterface.convert_sparse_mat_to_tensor(uu_norm).to(self.device)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def get_emb(self):
        return self.embedding_dict['user_emb'].data, self.embedding_dict['item_emb'].data

    def add_noise_(self, noise):
        self.embedding_dict['item_emb'].data = self.embedding_dict['item_emb'].data + noise

    def forward(self, perturbed=False, perturbed_adj=None, pretraining=False):
        self.eps = self.pretrain_noise
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj, list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj,
                                                 ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).to(self.device)
                ego_embeddings += F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.data.user_num, self.data.item_num]
        )

        if not pretraining:
            item_all_embeddings = self.embedding_dict['item_emb']
        return user_all_embeddings, item_all_embeddings

    def forward_uu(self, perturbed=False, perturbed_adj=None, pretraining=False):
        """User-user convolution path."""
        self.eps = self.pretrain_noise
        if pretraining:
            ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
            all_embeddings = []
            for k in range(self.layers):
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

        ego_user = self.embedding_dict['user_emb']
        all_user_embeddings = []
        for k in range(self.layers):
            ego_user = torch.sparse.mm(self.sparse_uu_norm, ego_user)
            if perturbed:
                random_noise = torch.rand_like(ego_user).to(self.device)
                ego_user += F.normalize(random_noise, dim=-1) * self.eps
            all_user_embeddings.append(ego_user)

        all_user_embeddings = torch.stack(all_user_embeddings, dim=1)
        user_all_embeddings = torch.mean(all_user_embeddings, dim=1)
        item_all_embeddings = self.embedding_dict['item_emb']
        return user_all_embeddings, item_all_embeddings
