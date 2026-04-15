import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataloader import DataSet
from lightGCN import LightGCN

torch.set_num_threads(8)


class MEMBER(nn.Module):
    def __init__(self, args, dataset: DataSet, expert_type='visited'):
        super(MEMBER, self).__init__()
        self.dataset = dataset
        self.device = args.device
        self.layers = args.layers
        self.layers_sg = args.layers_sg
        self.con_s = args.con_s
        self.con_us = args.con_us
        self.gen = args.gen
        self.expert_type = expert_type
        self.l2 = args.decay
        self.s_temp = args.temp_s
        self.us_temp = args.temp_us

        self.lambda_s = args.lambda_s
        self.lambda_us = args.lambda_us

        self.neg_edge = args.neg_edge
        self.dropout = args.dropout

        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.inter_matrix = dataset.inter_matrix
        self.all_inter_matrix = dataset.all_inter_matrix

        self.test_users = list(dataset.test_interacts.keys())
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size

        self.user_embedding_glo = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding_glo = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)

        self.user_embedding_loc = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding_loc = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)

        self.aux_graphs = nn.ModuleList(
            [
                LightGCN(
                    self.device,
                    self.layers,
                    self.n_users + 1,
                    self.n_items + 1,
                    self.inter_matrix[idx],
                )
                for idx in range(len(self.inter_matrix) - 1)
            ]
        )
        self.tar_graph = LightGCN(
            self.device,
            self.layers,
            self.n_users + 1,
            self.n_items + 1,
            self.inter_matrix[-1],
        )
        glo_layers = self.layers_sg if self.expert_type == 'visited' else self.layers
        self.glo_graph = LightGCN(
            self.device,
            glo_layers,
            self.n_users + 1,
            self.n_items + 1,
            self.all_inter_matrix,
        )

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self._load_model()

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def bpr_loss(self, p_score, n_score):
        gamma = 1e-10
        loss = -torch.log(gamma + torch.sigmoid(p_score - n_score))
        return loss.mean()

    def con_loss(self, pos, aug, temp):
        sampled_indices = torch.randperm(pos.shape[0], device=pos.device)[:1024]

        pos = pos[sampled_indices, :]
        aug = aug[sampled_indices, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1)
        ttl_score = torch.matmul(pos, aug.permute(1, 0))

        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.sum(torch.exp(ttl_score / temp), axis=1)

        return -torch.mean(torch.log(pos_score / ttl_score))

    def gen_loss(self, user, item, adj, batch_size=1024):
        num_neg_samples = self.neg_edge
        adj = adj.tocoo()

        if adj.nnz == 0:
            return torch.tensor(0.0, device=self.device)

        coo_indices = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long, device=self.device)
        num_pos_samples = coo_indices.size(1)

        if batch_size < num_pos_samples:
            sampled_indices = torch.randint(0, num_pos_samples, (batch_size,), device=self.device)
        else:
            sampled_indices = torch.arange(0, num_pos_samples, device=self.device)
            batch_size = num_pos_samples

        pos_user_indices = coo_indices[0][sampled_indices]
        pos_item_indices = coo_indices[1][sampled_indices]

        pos_scores = torch.sigmoid((user[pos_user_indices] * item[pos_item_indices]).sum(dim=1))

        neg_user_indices = torch.randint(0, user.size(0), (batch_size * num_neg_samples,), device=self.device)
        neg_item_indices = torch.randint(0, item.size(0), (batch_size * num_neg_samples,), device=self.device)

        neg_scores = torch.sigmoid((user[neg_user_indices] * item[neg_item_indices]).sum(dim=1))

        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([pos_labels, neg_labels])
        return F.binary_cross_entropy(all_scores, all_labels)

    def _compute_local_graph_embeddings(self, embeddings_loc):
        behavior_embeddings = [graph(embeddings_loc) for graph in self.aux_graphs]
        behavior_embeddings.append(self.tar_graph(embeddings_loc))
        return behavior_embeddings

    def _split_behavior_embeddings(self, behavior_embeddings):
        user_embeddings = []
        item_embeddings = []
        for embeddings in behavior_embeddings:
            user_embedding, item_embedding = torch.split(
                embeddings,
                [self.n_users + 1, self.n_items + 1],
            )
            user_embeddings.append(user_embedding)
            item_embeddings.append(item_embedding)
        return user_embeddings, item_embeddings

    def _compute_unvisited_gen_loss(self, user_embeddings, item_embeddings):
        losses = []
        for higher_idx in range(1, len(user_embeddings)):
            for lower_idx in range(higher_idx):
                losses.append(
                    self.gen_loss(
                        user_embeddings[higher_idx],
                        item_embeddings[higher_idx],
                        self.inter_matrix[lower_idx],
                    )
                )

        if not losses:
            return torch.tensor(0.0, device=self.device)

        return torch.stack(losses).mean()

    def _compute_shared_embeddings(self):
        embeddings_loc = torch.cat([self.user_embedding_loc.weight, self.item_embedding_loc.weight], dim=0)
        embeddings_glo = torch.cat([self.user_embedding_glo.weight, self.item_embedding_glo.weight], dim=0)

        behavior_embeddings = self._compute_local_graph_embeddings(embeddings_loc)
        user_embeddings, item_embeddings = self._split_behavior_embeddings(behavior_embeddings)

        glo_embeddings = self.glo_graph(embeddings_glo)
        self.user_glo_embedding, self.item_glo_embedding = torch.split(
            glo_embeddings,
            [self.n_users + 1, self.n_items + 1],
        )

        self.user_loc_embedding = torch.stack(user_embeddings, dim=0).mean(dim=0)
        self.item_loc_embedding = torch.stack(item_embeddings, dim=0).mean(dim=0)

        return embeddings_loc, embeddings_glo, user_embeddings, item_embeddings

    def forward(self, batch_data):
        embeddings_loc, embeddings_glo, user_embeddings, item_embeddings = self._compute_shared_embeddings()

        if self.expert_type == 'unvisited':
            glo_embeddings_aug = self.glo_graph_aug(embeddings_glo)
            user_glo_embedding_aug, item_glo_embedding_aug = torch.split(
                glo_embeddings_aug,
                [self.n_users + 1, self.n_items + 1],
            )

            c_loss_user = self.con_loss(self.user_glo_embedding, user_glo_embedding_aug, self.us_temp)
            c_loss_item = self.con_loss(self.item_glo_embedding, item_glo_embedding_aug, self.us_temp)
            c_loss = (c_loss_user + c_loss_item) / 2

            bce_loss = self._compute_unvisited_gen_loss(user_embeddings, item_embeddings)
            loss = self.gen * bce_loss + self.con_us * c_loss
        else:
            loc_embeddings_aug = self.buy_graph_aug(embeddings_loc)
            user_loc_embedding_aug, item_loc_embedding_aug = torch.split(
                loc_embeddings_aug,
                [self.n_users + 1, self.n_items + 1],
            )

            c_loss_user = self.con_loss(self.user_loc_embedding, user_loc_embedding_aug, self.s_temp)
            c_loss_item = self.con_loss(self.item_loc_embedding, item_loc_embedding_aug, self.s_temp)
            c_loss = (c_loss_user + c_loss_item) / 2
            bce_loss = torch.tensor(0.0, device=self.device)
            loss = self.gen * bce_loss + self.con_s * c_loss

        return loss

    def full_predict(self, users):
        self._compute_shared_embeddings()

        self.user_emb_loc = self.user_loc_embedding[users.long()]
        self.user_emb_glo = self.user_glo_embedding[users.long()]

        scores_loc = torch.matmul(self.user_emb_loc, self.item_loc_embedding.transpose(0, 1))
        scores_glo = torch.matmul(self.user_emb_glo, self.item_glo_embedding.transpose(0, 1))

        if self.expert_type == 'visited':
            return self.lambda_s * scores_glo + (1 - self.lambda_s) * scores_loc
        return self.lambda_us * scores_glo + (1 - self.lambda_us) * scores_loc
