import copy
import time
import os

import torch
import numpy as np
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import pickle

import torch.nn.functional as F
from dataloader import DataSet
from metrics import metrics_dict
from lightGCN_aug import LightGCN
import numpy as np
import torch.nn as nn

class Trainer(object):

    def __init__(self, model_visited, model_unvisited, dataset: DataSet, args):
        self.model_visited = model_visited
        self.model_unvisited = model_unvisited
        self.dataset = dataset
        self.behaviors = args.behaviors
        self.topk = args.topk
        self.alpha = args.alpha
        self.metrics = args.metrics
        self.learning_rate = args.lr
        self.weight_decay = args.decay

        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.min_epoch = args.min_epoch
        self.epochs = args.epochs
        self.model_path = args.model_path
        self.model_name = args.model_name
        self.embedding_size = args.embedding_size
        
        self.device = args.device
        self.TIME = args.TIME
        self.setting = args.setting
        self.mask_validation = bool(getattr(args, 'mask_validation', False))

        self._eval_exclusion_cache = {}

        self.optimizer_visited = self.get_optimizer(self.model_visited)
        self.optimizer_unvisited = self.get_optimizer(self.model_unvisited)
        
        self._initialize_masks()
        
    def _initialize_masks(self):
        aux_interactions = sum(self.dataset.inter_matrix[:-1])
        
        aux_interactions_dense = torch.tensor(aux_interactions.toarray(), device=self.device, dtype=torch.bool)

        # Construct masks
        self.visited_mask = aux_interactions_dense  
        self.unvisited_mask = ~aux_interactions_dense  
        self.behavior_eval_matrices = {
            behavior: self.dataset.inter_matrix[idx].tocsr()
            for idx, behavior in enumerate(self.behaviors[:-1])
        }

    def _normalize_eval_setting(self, setting):
        aliases = {
            'basic': 'basic',
            'visited': 'seen',
            'seen': 'seen',
            'unvisited': 'unseen',
            'unseen': 'unseen',
        }
        return aliases.get(setting, setting)

    def _strip_user_activity_segment(self, setting):
        if setting in {'warm', 'cold'}:
            return 'basic'

        for segment in ('warm', 'cold'):
            prefix = f'{segment}_'
            if setting.startswith(prefix):
                return setting[len(prefix):]

        return setting

    def _strip_item_popularity_segment(self, setting):
        if setting in {
            'item_warm',
            'item_cold',
            'warm_item',
            'cold_item',
            'item_popular',
            'item_unpopular',
            'popular_item',
            'unpopular_item',
            'popular',
            'unpopular',
        }:
            return 'basic'

        for segment in (
            'item_warm',
            'item_cold',
            'warm_item',
            'cold_item',
            'item_popular',
            'item_unpopular',
            'popular_item',
            'unpopular_item',
            'popular',
            'unpopular',
        ):
            prefix = f'{segment}_'
            if setting.startswith(prefix):
                return setting[len(prefix):]

        return setting

    def _mask_scores_with_batch_mask(self, scores, mask_batch):
        # return scores * mask_batch.to(dtype=scores.dtype)
        return scores.masked_fill(~mask_batch.bool(), float('-inf'))

    def _get_behavior_mask_batch(self, behavior, user_samples):
        if behavior not in self.behavior_eval_matrices:
            raise ValueError(f'Unsupported behavior for fine-grained evaluation: {behavior}')

        batch_mask = self.behavior_eval_matrices[behavior][user_samples.detach().cpu().numpy()].toarray()
        return torch.tensor(batch_mask, device=self.device, dtype=torch.bool)

    def _get_eval_exclusion_items(self, split_name):
        normalized_split = split_name.lower()
        if normalized_split in self._eval_exclusion_cache:
            return self._eval_exclusion_cache[normalized_split]

        exclusion_items = {
            user_id: set(int(item_id) for item_id in items)
            for user_id, items in self.dataset.train_behavior_dict[self.behaviors[-1]].items()
        }

        if self.mask_validation and normalized_split == 'test':
            for user_id, items in self.dataset.validation_split_interacts['basic'].items():
                exclusion_items.setdefault(user_id, set()).update(int(item_id) for item_id in items)

        exclusion_items = {
            user_id: list(item_ids)
            for user_id, item_ids in exclusion_items.items()
        }
        self._eval_exclusion_cache[normalized_split] = exclusion_items
        return exclusion_items

    def _combine_scores_for_eval(self, scores_visited_raw, scores_unvisited_raw, user_samples, setting):
        normalized_setting = self._normalize_eval_setting(
            self._strip_user_activity_segment(
                self._strip_item_popularity_segment(setting)
            )
        )

        if normalized_setting == 'basic':
            return self.apply_masks(
                scores_visited_raw,
                scores_unvisited_raw,
                user_samples,
                torch.arange(self.dataset.item_count + 1, device=self.device).unsqueeze(0).expand(user_samples.size(0), -1),
            )

        if normalized_setting == 'seen':
            visited_mask_batch = self.visited_mask[user_samples]
            return self._mask_scores_with_batch_mask(scores_visited_raw, visited_mask_batch)

        if normalized_setting == 'unseen':
            unvisited_mask_batch = self.unvisited_mask[user_samples]
            return self._mask_scores_with_batch_mask(scores_unvisited_raw, unvisited_mask_batch)

        if normalized_setting.endswith('_seen'):
            behavior = normalized_setting[:-5]
            behavior_mask_batch = self._get_behavior_mask_batch(behavior, user_samples)
            return self._mask_scores_with_batch_mask(scores_visited_raw, behavior_mask_batch)

        if normalized_setting.endswith('_unseen'):
            behavior = normalized_setting[:-7]
            behavior_mask_batch = self._get_behavior_mask_batch(behavior, user_samples)
            return self._mask_scores_with_batch_mask(scores_unvisited_raw, ~behavior_mask_batch)

        raise ValueError(f'Unsupported evaluation setting: {setting}')
        
    
    def get_optimizer(self, model):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        return optimizer
    
    def clear_parameter(self, model):
        model.storage_user_embeddings = None
        model.storage_item_embeddings = None
        
    
    def bpr_loss(self, p_score, n_score):
        self.gamma = 1e-10
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        loss = loss.mean()

        return loss
    
    @logger.catch()
    def train_model(self):
        train_dataset_loader = DataLoader(dataset=self.dataset.behavior_dataset(),
                                          batch_size=self.batch_size,
                                          shuffle=True)
        
        test_metric_dict = None
        
        for epoch in range(self.epochs):

            self.model_unvisited.glo_graph_aug = LightGCN(self.model_unvisited.device, self.model_unvisited.layers, self.model_unvisited.n_users + 1, self.model_unvisited.n_items + 1, self.dataset.aug_inter_matrix, self.model_unvisited.dropout)
            
            self.model_visited.buy_graph_aug = LightGCN(self.model_visited.device, self.model_visited.layers, self.model_visited.n_users + 1, self.model_visited.n_items + 1, self.dataset.buy_seen_matrix, self.model_visited.dropout)
            
            self.model_visited.train()
            self.model_unvisited.train()
            
            self._train_pre_epoch(train_dataset_loader, epoch, self.model_visited, self.optimizer_visited)
            self._train_pre_epoch(train_dataset_loader, epoch, self.model_unvisited, self.optimizer_unvisited)
            
            self._train_fine_epoch(train_dataset_loader, epoch, self.model_visited, self.model_unvisited, self.optimizer_visited, self.optimizer_unvisited)

            test_metric_dict = self.evaluate_combine(epoch)

        # Save final checkpoints for both experts and optimizers.
        self.save_model(self.model_visited, 'visited_model')
        self.save_model(self.model_unvisited, 'unvisited_model')
        self.save_model(self.optimizer_visited, 'visited_optimizer')
        self.save_model(self.optimizer_unvisited, 'unvisited_optimizer')

        if test_metric_dict is not None:
            logger.info(f"final test result is:  %s" % test_metric_dict.__str__())
    
    def apply_masks(self, scores_visited, scores_unvisited, user_indices, item_indices):

        visited_mask_batch = torch.gather(self.visited_mask[user_indices], 1, item_indices)
        unvisited_mask_batch = torch.gather(self.unvisited_mask[user_indices], 1, item_indices)
        
        visited_mask_batch = visited_mask_batch.to(dtype=scores_visited.dtype) #+ epsilon
        unvisited_mask_batch = unvisited_mask_batch.to(dtype=scores_unvisited.dtype) #+ epsilon
        
        # Apply masks
        visited_scores = scores_visited * visited_mask_batch
        unvisited_scores = scores_unvisited * unvisited_mask_batch

        return visited_scores + unvisited_scores
    
    def emb_loss(self, embeddings):
        emb_loss = 0
        
        for embedding in embeddings:
            tmp = torch.norm(embedding, p=2)
            tmp = tmp / embedding.shape[0]
            emb_loss += tmp
        
        return emb_loss
        
    def _train_pre_epoch(self, behavior_dataset_loader, epoch, model, optimizer):
        start_time = time.time()
        behavior_dataset_iter = (
            tqdm(
                enumerate(behavior_dataset_loader),
                total=len(behavior_dataset_loader),
                desc=f"\033[1;35m Train {epoch + 1:>5}\033[0m"
            )
        )
        total_loss = 0.0
        batch_no = 0
        for batch_index, batch_data in behavior_dataset_iter:
            batch_data = batch_data.to(self.device)
            optimizer.zero_grad()
            loss = model(batch_data)  # Use the specific model
            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.item()
            batch_no = batch_index + 1

        total_loss = total_loss / batch_no

        epoch_time = time.time() - start_time
        logger.info('epoch %d %.2fs Train loss is [%.4f] ' % (epoch + 1, epoch_time, total_loss))

        self.clear_parameter(self.model_visited)
        self.clear_parameter(self.model_unvisited)
        
    def _train_fine_epoch(self, behavior_dataset_loader, epoch, model_visited, model_unvisited, optimizer_visited, optimizer_unvisited):
        start_time = time.time()
        behavior_dataset_iter = (
            tqdm(
                enumerate(behavior_dataset_loader),
                total=len(behavior_dataset_loader),
                desc=f"\033[1;35m Train {epoch + 1:>5}\033[0m"
            )
        )
        total_loss = 0.0
        batch_no = 0
        
        for batch_index, batch_data in behavior_dataset_iter:
            batch_data = batch_data.to(self.device)
            pair_samples = batch_data[:, -1, :-1] 
            mask = torch.any(pair_samples != 0, dim=-1) 
            pair_samples = pair_samples[mask] 
                        
            optimizer_visited.zero_grad()
            optimizer_unvisited.zero_grad()
            
            bpr_loss = 0
            
            if pair_samples.shape[0] > 0:
                user_samples = pair_samples[:, 0].long()
                item_samples = pair_samples[:, 1:].long()
                u_emb_glo = model_visited.user_glo_embedding[user_samples].unsqueeze(1)
                i_emb_glo = model_visited.item_glo_embedding[item_samples]
                score_point_glo_visited = torch.sum((u_emb_glo * i_emb_glo), dim=-1)
                
                u_emb_loc = model_visited.user_loc_embedding[user_samples].unsqueeze(1)
                i_emb_loc = model_visited.item_loc_embedding[item_samples]
                score_point_loc_visited = torch.sum((u_emb_loc * i_emb_loc), dim = -1)
                
                bpr_scores_visited_raw = model_visited.lambda_s*score_point_glo_visited + (1-model_visited.lambda_s)*score_point_loc_visited 
                bpr_scores_visited = self.apply_masks(bpr_scores_visited_raw, torch.zeros_like(bpr_scores_visited_raw), user_samples, item_samples) 
                
                u_emb_glo_unseen = model_unvisited.user_glo_embedding[user_samples].unsqueeze(1)
                i_emb_glo_unseen = model_unvisited.item_glo_embedding[item_samples]
                score_point_glo_unvisited = torch.sum((u_emb_glo_unseen * i_emb_glo_unseen), dim=-1)
                
                u_emb_loc_unseen = model_unvisited.user_loc_embedding[user_samples].unsqueeze(1)
                i_emb_loc_unseen = model_unvisited.item_loc_embedding[item_samples]
                score_point_loc_unvisited = torch.sum((u_emb_loc_unseen * i_emb_loc_unseen), dim = -1)
                
                bpr_scores_unvisited_raw = model_unvisited.lambda_us*score_point_glo_unvisited + (1-model_unvisited.lambda_us)*score_point_loc_unvisited
                bpr_scores_unvisited = self.apply_masks(torch.zeros_like(bpr_scores_unvisited_raw), bpr_scores_unvisited_raw, user_samples, item_samples) 
                
                bpr_scores = bpr_scores_visited + bpr_scores_unvisited

            p_scores, n_scores = torch.chunk(bpr_scores, 2, dim=-1)
            bpr_loss = self.bpr_loss(p_scores, self.alpha*n_scores)
            
            emb_loss = (self.emb_loss((model_visited.user_glo_embedding, model_visited.item_glo_embedding)) + self.emb_loss((model_visited.user_loc_embedding, model_visited.item_loc_embedding))) / 2
            
            loss = bpr_loss + 0.001 * emb_loss

            loss.backward(retain_graph=True)
            
            optimizer_visited.step()
            optimizer_unvisited.step()
            
            total_loss += loss.item() 
            batch_no = batch_index + 1

        total_loss = total_loss / batch_no

        epoch_time = time.time() - start_time
        logger.info('epoch %d %.2fs Train loss is [%.4f] ' % (epoch + 1, epoch_time, total_loss))

        self.clear_parameter(self.model_visited)
        self.clear_parameter(self.model_unvisited)

        
    def evaluate_combine(self, epoch):
        start_time = time.time()
        eval_dataset, eval_interacts, eval_gt_length = self.dataset.get_eval_bundle('test', self.setting)
        test_metric_dict = self.evaluate(
            epoch,
            self.test_batch_size,
            eval_dataset,
            eval_interacts,
            eval_gt_length,
            setting=self.setting,
            split_name='test',
        )
        epoch_time = time.time() - start_time
        logger.info(
            f"test ({self.setting}) %d cost time %.2fs, result: %s " % (epoch + 1, epoch_time, test_metric_dict.__str__()))

        return test_metric_dict
    
    @logger.catch()
    @torch.no_grad()
    def evaluate(self, epoch, test_batch_size, dataset, gt_interacts, gt_length, setting = 'basic', split_name='test'):
        data_loader = DataLoader(dataset=dataset, batch_size=test_batch_size)
        self.model_visited.eval()
        self.model_unvisited.eval()
        
        start_time = time.time()
        iter_data = (
            tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                desc=f"\033[1;35mEvaluate \033[0m"
            )
        )
        topk_list = []
        exclusion_items = self._get_eval_exclusion_items(split_name)
        for batch_index, batch_data in iter_data:
            batch_data = batch_data.to(self.device)
            start = time.time()
            user_samples = batch_data.long()
            
            scores_visited_raw = self.model_visited.full_predict(batch_data)
            scores_unvisited_raw = self.model_unvisited.full_predict(batch_data)
            combined_scores = self._combine_scores_for_eval(
                scores_visited_raw,
                scores_unvisited_raw,
                user_samples,
                setting,
            )
            
            for index, user in enumerate(batch_data):
                user_score = combined_scores[index]
                items = exclusion_items.get(str(user.item()), None)
                if items is not None:
                    user_score[items] = -np.inf
                _, topk_idx = torch.topk(user_score, max(self.topk), dim=-1)
                gt_items = gt_interacts[str(user.item())]
                mask = np.isin(topk_idx.to('cpu'), gt_items)
                topk_list.append(mask)

        topk_list = np.array(topk_list)
        metric_dict = self.calculate_result(topk_list, gt_length)
        return metric_dict

    def calculate_result(self, topk_list, gt_len):
        if len(gt_len) == 0:
            metric_dict = {}
            for topk in self.topk:
                for metric in self.metrics:
                    metric_dict[f'{metric}@{topk}'] = 0.0
            return metric_dict

        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_list, gt_len)
            result_list.append(result)
            
        result_list = np.stack(result_list, axis=0).mean(axis=1)
    
        metric_dict = {}
        for topk in self.topk:
            for metric, value in zip(self.metrics, result_list):
                key = '{}@{}'.format(metric, topk)
                metric_dict[key] = np.round(value[topk - 1], 4)

        return metric_dict

    def save_model(self, model_or_optimizer, tag):
        if model_or_optimizer is None:
            logger.warning(f'Skip saving checkpoint because {tag} is None')
            return

        os.makedirs(self.model_path, exist_ok=True)
        file_name = f'{self.model_name}_{tag}_{self.TIME}.pth'
        torch.save(model_or_optimizer.state_dict(), os.path.join(self.model_path, file_name))
