import argparse
import os
import random
import json
import torch
import scipy.sparse as sp

from torch.utils.data import Dataset, DataLoader
import numpy as np


class TestDate(Dataset):
    def __init__(self, user_count, item_count, samples=None):
        self.user_count = user_count
        self.item_count = item_count
        self.samples = samples

    def __getitem__(self, idx):
        return int(self.samples[idx])

    def __len__(self):
        return len(self.samples)
    
    
class BehaviorDate(Dataset):
    def __init__(self, user_count, item_count, pos_sampling, neg_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.pos_sampling = pos_sampling
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors
        self.neg_count = neg_count

    def __getitem__(self, idx):
        total = []
        pos = self.pos_sampling[idx]
        u_id = pos[0]

        buy_inter = self.behavior_dict[self.behaviors[-1]].get(str(u_id), None) 

        if buy_inter is None: 
            signal = [0, 0, 0, 0]
        else:
            p_item = random.choice(buy_inter) 
            n_item = random.randint(1, self.item_count)
            while np.isin(n_item, buy_inter):
                n_item = random.randint(1, self.item_count)
            signal = [pos[0], p_item, n_item, 0] 

        total.append(signal) 
        
        return np.array(total)

    def __len__(self):
        return len(self.pos_sampling)


class DataSet(object):

    def __init__(self, args):

        self.behaviors = args.behaviors
        self.path = args.data_path
        self.neg_count = args.neg_count
        self.user_activity_split_type = getattr(args, 'user_activity_split_type', 'top_ratio')
        self.user_activity_warm_ratio = float(getattr(args, 'user_activity_warm_ratio', 0.2))
        self.user_activity_pareto_target = float(getattr(args, 'user_activity_pareto_target', 0.8))
        self.user_activity_min_warm_users = int(getattr(args, 'user_activity_min_warm_users', 1))
        self.item_popularity_split_type = getattr(args, 'item_popularity_split_type', 'top_ratio')
        self.item_popularity_warm_ratio = float(getattr(args, 'item_popularity_warm_ratio', 0.2))
        self.item_popularity_pareto_target = float(getattr(args, 'item_popularity_pareto_target', 0.8))
        self.item_popularity_min_warm_items = int(getattr(args, 'item_popularity_min_warm_items', 1))
        self._eval_bundle_cache = {}
        self._user_activity_summary_cache = {}
        self._item_popularity_summary_cache = {}
        
        self.__get_count()
        self.__get_pos_sampling()
        self.__get_behavior_items()
        self.__validate_user_activity_args()
        self.__validate_item_popularity_args()
        self.__get_validation_dict()
        self.__get_test_dict()
        self.__get_sparse_interact_dict()
        
        self.validation_gt_length = self._get_gt_length(self.validation_interacts)
        self.validation_gt_length_seen = self._get_gt_length(self.validation_interacts_seen)
        self.validation_gt_length_unseen = self._get_gt_length(self.validation_interacts_unseen)
        
        self.test_gt_length = self._get_gt_length(self.test_interacts)
        self.test_gt_length_seen = self._get_gt_length(self.test_interacts_seen)
        self.test_gt_length_unseen = self._get_gt_length(self.test_interacts_unseen)

    def __get_count(self):
        with open(os.path.join(self.path, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']

    def _load_json_dict(self, filename, required=True):
        filepath = os.path.join(self.path, filename)
        if not os.path.exists(filepath):
            if required:
                raise FileNotFoundError(f'Missing required file: {filepath}')
            return {}

        with open(filepath, encoding='utf-8') as f:
            return json.load(f)

    def _get_gt_length(self, interacts):
        return np.array([len(x) for _, x in interacts.items()], dtype=int)

    def _load_train_buy_user_counts(self):
        buy_counts = np.zeros(self.user_count + 1, dtype=np.int64)
        buy_dict_path = os.path.join(self.path, 'buy_dict.txt')
        if os.path.exists(buy_dict_path):
            with open(buy_dict_path, encoding='utf-8') as f:
                buy_dict = json.load(f)

            for user_id, items in buy_dict.items():
                user_id = int(user_id)
                if 0 <= user_id <= self.user_count:
                    buy_counts[user_id] = len(items)
            return buy_counts

        buy_path = os.path.join(self.path, 'buy.txt')
        with open(buy_path, encoding='utf-8') as f:
            for line in f:
                user_id, _ = map(int, line.strip('\n').strip().split())
                if 0 <= user_id <= self.user_count:
                    buy_counts[user_id] += 1

        return buy_counts

    def _load_train_buy_item_counts(self):
        buy_counts = np.zeros(self.item_count + 1, dtype=np.int64)
        buy_path = os.path.join(self.path, 'buy.txt')

        with open(buy_path, encoding='utf-8') as f:
            for line in f:
                _, item_id = map(int, line.strip('\n').strip().split())
                if 0 <= item_id <= self.item_count:
                    buy_counts[item_id] += 1

        return buy_counts

    def _resolve_warm_user_count(self, candidate_user_count):
        if candidate_user_count == 0:
            return 0

        if self.user_activity_split_type == 'pareto':
            warm_count = candidate_user_count - int(round(candidate_user_count * self.user_activity_pareto_target))
        elif self.user_activity_split_type == 'top_ratio':
            warm_count = int(round(candidate_user_count * self.user_activity_warm_ratio))
        else:
            raise ValueError(
                f'Unsupported user_activity_split_type: {self.user_activity_split_type}. '
                'Use top_ratio or pareto.'
            )

        if self.user_activity_min_warm_users > 0:
            warm_count = max(warm_count, self.user_activity_min_warm_users)
        warm_count = min(warm_count, candidate_user_count)

        return warm_count

    def _select_warm_users(self, candidate_users):
        if len(candidate_users) == 0:
            return np.array([], dtype=int)

        candidate_users = np.array(candidate_users, dtype=int)
        candidate_counts = self.train_buy_user_counts[candidate_users]
        sort_order = np.lexsort((candidate_users, -candidate_counts))
        sorted_candidate_users = candidate_users[sort_order]
        warm_count = self._resolve_warm_user_count(len(sorted_candidate_users))
        
        if warm_count == 0:
            return np.array([], dtype=int)
        
        return sorted_candidate_users[:warm_count]

    def __validate_user_activity_args(self):
        if self.user_activity_split_type not in {'top_ratio', 'pareto'}:
            raise ValueError('user_activity_split_type must be top_ratio or pareto.')
        if not 0 < self.user_activity_warm_ratio <= 1:
            raise ValueError('user_activity_warm_ratio must be in (0, 1].')
        if not 0 < self.user_activity_pareto_target <= 1:
            raise ValueError('user_activity_pareto_target must be in (0, 1].')
        if self.user_activity_min_warm_users < 0:
            raise ValueError('user_activity_min_warm_users must be >= 0.')

        self.train_buy_user_counts = self._load_train_buy_user_counts()

    def __validate_item_popularity_args(self):
        if self.item_popularity_split_type not in {'top_ratio', 'pareto'}:
            raise ValueError('item_popularity_split_type must be top_ratio or pareto.')
        if not 0 < self.item_popularity_warm_ratio <= 1:
            raise ValueError('item_popularity_warm_ratio must be in (0, 1].')
        if not 0 < self.item_popularity_pareto_target <= 1:
            raise ValueError('item_popularity_pareto_target must be in (0, 1].')
        if self.item_popularity_min_warm_items < 0:
            raise ValueError('item_popularity_min_warm_items must be >= 0.')

        self.train_buy_item_counts = self._load_train_buy_item_counts()
        self.train_buy_item_ids = np.flatnonzero(self.train_buy_item_counts).astype(int)
        self.train_buy_item_ids = self.train_buy_item_ids[self.train_buy_item_ids > 0]

    def _build_user_activity_segments(self, interacts):
        candidate_users = np.array([int(user_id) for user_id in interacts.keys()], dtype=int)
        warm_users = self._select_warm_users(candidate_users)
        warm_user_set = set(int(user_id) for user_id in warm_users.tolist())
        cold_user_set = set(int(user_id) for user_id in candidate_users.tolist()) - warm_user_set
        warm_interactions = int(self.train_buy_user_counts[list(warm_user_set)].sum()) if warm_user_set else 0
        total_interactions = int(self.train_buy_user_counts[candidate_users].sum()) if len(candidate_users) > 0 else 0

        summary = {
            'split_type': self.user_activity_split_type,
            'warm_ratio': self.user_activity_warm_ratio,
            'pareto_target': self.user_activity_pareto_target,
            'min_warm_users': self.user_activity_min_warm_users,
            'candidate_user_count': int(len(candidate_users)),
            'warm_user_count': int(len(warm_user_set)),
            'cold_user_count': int(len(cold_user_set)),
            'warm_user_share': round(
                len(warm_user_set) / len(candidate_users), 4
            ) if len(candidate_users) > 0 else 0.0,
            'warm_train_buy_count': warm_interactions,
            'cold_train_buy_count': int(total_interactions - warm_interactions),
            'warm_train_buy_share': round(
                warm_interactions / total_interactions, 4
            ) if total_interactions > 0 else 0.0,
        }
        segments = {
            'warm': warm_user_set,
            'cold': cold_user_set,
        }
        summary.update(self._summarize_user_activity_segment('warm', warm_users))
        summary.update(self._summarize_user_activity_segment('cold', np.array(sorted(cold_user_set), dtype=int)))

        return segments, summary

    def _summarize_user_activity_segment(self, segment_name, user_ids):
        user_ids = np.array(user_ids, dtype=int)
        if len(user_ids) == 0:
            return {
                f'{segment_name}_min_train_buy_count': 0,
                f'{segment_name}_max_train_buy_count': 0,
                f'{segment_name}_avg_train_buy_count': 0.0,
            }

        segment_counts = self.train_buy_user_counts[user_ids]
        return {
            f'{segment_name}_min_train_buy_count': int(segment_counts.min()),
            f'{segment_name}_max_train_buy_count': int(segment_counts.max()),
            f'{segment_name}_avg_train_buy_count': round(float(segment_counts.mean()), 4),
        }

    def _summarize_item_popularity_segment(self, segment_name, item_ids):
        item_ids = np.array(item_ids, dtype=int)
        if len(item_ids) == 0:
            return {
                f'{segment_name}_min_train_buy_count': 0,
                f'{segment_name}_max_train_buy_count': 0,
                f'{segment_name}_avg_train_buy_count': 0.0,
            }

        segment_counts = self.train_buy_item_counts[item_ids]
        return {
            f'{segment_name}_min_train_buy_count': int(segment_counts.min()),
            f'{segment_name}_max_train_buy_count': int(segment_counts.max()),
            f'{segment_name}_avg_train_buy_count': round(float(segment_counts.mean()), 4),
        }

    def _split_user_activity_setting(self, setting):
        if setting in {'warm', 'cold'}:
            return setting, 'basic'

        return None, setting

    def _resolve_warm_item_count(self, candidate_item_count):
        if candidate_item_count == 0:
            return 0

        warm_count = int(round(candidate_item_count * self.item_popularity_warm_ratio))
        if self.item_popularity_min_warm_items > 0:
            warm_count = max(warm_count, self.item_popularity_min_warm_items)
        warm_count = min(warm_count, candidate_item_count)

        return warm_count

    def _select_warm_items(self, candidate_items):
        if len(candidate_items) == 0:
            return np.array([], dtype=int)

        candidate_items = np.array(candidate_items, dtype=int)
        candidate_counts = self.train_buy_item_counts[candidate_items]
        sort_order = np.lexsort((candidate_items, -candidate_counts))
        sorted_candidate_items = candidate_items[sort_order]
        sorted_candidate_counts = candidate_counts[sort_order]

        if self.item_popularity_split_type == 'pareto':
            total_interactions = int(sorted_candidate_counts.sum())
            if total_interactions > 0:
                cumulative_share = np.cumsum(sorted_candidate_counts) / total_interactions
                warm_count = int(
                    np.searchsorted(cumulative_share, self.item_popularity_pareto_target, side='left') + 1
                )
            else:
                warm_count = self._resolve_warm_item_count(len(sorted_candidate_items))
        elif self.item_popularity_split_type == 'top_ratio':
            warm_count = self._resolve_warm_item_count(len(sorted_candidate_items))
        else:
            raise ValueError(
                f'Unsupported item_popularity_split_type: {self.item_popularity_split_type}. '
                'Use top_ratio or pareto.'
            )

        if self.item_popularity_min_warm_items > 0:
            warm_count = max(warm_count, self.item_popularity_min_warm_items)
        warm_count = min(warm_count, len(sorted_candidate_items))

        if warm_count == 0:
            return np.array([], dtype=int)

        return sorted_candidate_items[:warm_count]

    def _build_item_popularity_segments(self):
        candidate_item_array = np.array(self.train_buy_item_ids, dtype=int)
        warm_items = self._select_warm_items(candidate_item_array)
        warm_item_set = set(int(item_id) for item_id in warm_items.tolist())
        cold_item_set = set(int(item_id) for item_id in candidate_item_array.tolist()) - warm_item_set
        warm_interactions = int(self.train_buy_item_counts[list(warm_item_set)].sum()) if warm_item_set else 0
        total_interactions = int(self.train_buy_item_counts[candidate_item_array].sum()) if len(candidate_item_array) > 0 else 0

        summary = {
            'split_type': self.item_popularity_split_type,
            'warm_ratio': self.item_popularity_warm_ratio,
            'pareto_target': self.item_popularity_pareto_target,
            'min_warm_items': self.item_popularity_min_warm_items,
            'split_scope': 'global_train_buy_items',
            'split_definition': (
                'Items with at least one train-buy interaction are ranked by global train-buy count. '
                'The top group is popular/warm and the rest is unpopular/cold.'
            ),
            'candidate_item_count': int(len(candidate_item_array)),
            'warm_item_count': int(len(warm_item_set)),
            'cold_item_count': int(len(cold_item_set)),
            'warm_item_share': round(
                len(warm_item_set) / len(candidate_item_array), 4
            ) if len(candidate_item_array) > 0 else 0.0,
            'warm_train_buy_count': warm_interactions,
            'cold_train_buy_count': int(total_interactions - warm_interactions),
            'warm_train_buy_share': round(
                warm_interactions / total_interactions, 4
            ) if total_interactions > 0 else 0.0,
        }
        cold_items = np.array(sorted(cold_item_set), dtype=int)
        summary.update(self._summarize_item_popularity_segment('warm', warm_items))
        summary.update(self._summarize_item_popularity_segment('cold', cold_items))
        summary.update({
            'popular_item_count': summary['warm_item_count'],
            'unpopular_item_count': summary['cold_item_count'],
            'popular_item_share': summary['warm_item_share'],
            'popular_train_buy_count': summary['warm_train_buy_count'],
            'unpopular_train_buy_count': summary['cold_train_buy_count'],
            'popular_train_buy_share': summary['warm_train_buy_share'],
            'popular_min_train_buy_count': summary['warm_min_train_buy_count'],
            'popular_max_train_buy_count': summary['warm_max_train_buy_count'],
            'popular_avg_train_buy_count': summary['warm_avg_train_buy_count'],
            'unpopular_min_train_buy_count': summary['cold_min_train_buy_count'],
            'unpopular_max_train_buy_count': summary['cold_max_train_buy_count'],
            'unpopular_avg_train_buy_count': summary['cold_avg_train_buy_count'],
        })
        segments = {
            'warm': warm_item_set,
            'cold': cold_item_set,
            'popular': warm_item_set,
            'unpopular': cold_item_set,
        }

        return segments, summary

    def _get_item_popularity_cache_key(self, interacts=None):
        return 'global_train_buy_items'

    def _split_item_popularity_setting(self, setting):
        item_only_aliases = {
            'item_warm': 'warm',
            'item_cold': 'cold',
            'warm_item': 'warm',
            'cold_item': 'cold',
            'item_popular': 'warm',
            'item_unpopular': 'cold',
            'popular_item': 'warm',
            'unpopular_item': 'cold',
            'popular': 'warm',
            'unpopular': 'cold',
        }
        if setting in item_only_aliases:
            return item_only_aliases[setting], 'basic'

        for prefix, segment in (
            ('item_warm_', 'warm'),
            ('item_cold_', 'cold'),
            ('warm_item_', 'warm'),
            ('cold_item_', 'cold'),
            ('item_popular_', 'warm'),
            ('item_unpopular_', 'cold'),
            ('popular_item_', 'warm'),
            ('unpopular_item_', 'cold'),
            ('popular_', 'warm'),
            ('unpopular_', 'cold'),
        ):
            if setting.startswith(prefix):
                return segment, setting[len(prefix):]

        return None, setting

    def _filter_interacts_by_item_segment(self, interacts, segment):
        cache_key = self._get_item_popularity_cache_key(interacts)
        if cache_key not in self._item_popularity_summary_cache:
            self._item_popularity_summary_cache[cache_key] = self._build_item_popularity_segments()

        item_popularity_segments, _ = self._item_popularity_summary_cache[cache_key]

        if segment not in item_popularity_segments:
            raise ValueError(f'Unsupported item popularity segment: {segment}')

        item_ids = item_popularity_segments[segment]
        filtered_interacts = {}
        for user_id, items in interacts.items():
            filtered_items = [item_id for item_id in items if int(item_id) in item_ids]
            if filtered_items:
                filtered_interacts[user_id] = filtered_items

        return filtered_interacts

    def _filter_interacts_by_user_segment(self, interacts, segment):
        cache_key = tuple(sorted(interacts.keys()))
        if cache_key not in self._user_activity_summary_cache:
            self._user_activity_summary_cache[cache_key] = self._build_user_activity_segments(interacts)

        user_activity_segments, _ = self._user_activity_summary_cache[cache_key]

        if segment not in user_activity_segments:
            raise ValueError(f'Unsupported user activity segment: {segment}')

        user_ids = user_activity_segments[segment]
        return {
            user_id: items
            for user_id, items in interacts.items()
            if int(user_id) in user_ids
        }

    def get_user_activity_summary(self, split_name='test'):
        split_name = split_name.lower()
        if split_name == 'test':
            interacts = self.test_split_interacts['basic']
        elif split_name == 'validation':
            interacts = self.validation_split_interacts['basic']
        else:
            raise ValueError(f'Unsupported split: {split_name}')

        cache_key = tuple(sorted(interacts.keys()))
        if cache_key not in self._user_activity_summary_cache:
            self._user_activity_summary_cache[cache_key] = self._build_user_activity_segments(interacts)

        _, summary = self._user_activity_summary_cache[cache_key]
        return summary

    def get_item_popularity_summary(self, split_name='test'):
        split_name = split_name.lower()
        if split_name not in {'test', 'validation'}:
            raise ValueError(f'Unsupported split: {split_name}')

        cache_key = self._get_item_popularity_cache_key()
        if cache_key not in self._item_popularity_summary_cache:
            self._item_popularity_summary_cache[cache_key] = self._build_item_popularity_segments()

        _, summary = self._item_popularity_summary_cache[cache_key]
        return summary

    def _load_split_dict_bundle(self, split_name):
        split_bundle = {
            'basic': self._load_json_dict(f'{split_name}_dict.txt'),
            'seen': self._load_json_dict(f'{split_name}_seen_dict.txt', required=False),
            'unseen': self._load_json_dict(f'{split_name}_unseen_dict.txt', required=False),
        }

        for behavior in self.behaviors[:-1]:
            split_bundle[f'{behavior}_seen'] = self._load_json_dict(
                f'{split_name}_{behavior}_seen_dict.txt',
                required=False,
            )
            split_bundle[f'{behavior}_unseen'] = self._load_json_dict(
                f'{split_name}_{behavior}_unseen_dict.txt',
                required=False,
            )

        return split_bundle
    
    def __get_pos_sampling(self):
        with open(os.path.join(self.path, 'buy.txt'), encoding='utf-8') as f:
            data = f.readlines()
            arr = []
            for line in data:
                line = line.strip('\n').strip().split()
                arr.append([int(x) for x in line])
            self.pos_sampling = arr

    def __get_behavior_items(self):
        """
        load the list of items corresponding to the user under each behavior
        :return:
        """
        self.train_behavior_dict = {}
        for behavior in self.behaviors:
            print(behavior)
            with open(os.path.join(self.path, behavior + '_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                self.train_behavior_dict[behavior] = b_dict
        with open(os.path.join(self.path, 'all_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.train_behavior_dict['all'] = b_dict


    def __get_test_dict(self):
        """
        load the list of items that the user has interacted with in the test set
        :return:
        """
        self.test_split_interacts = self._load_split_dict_bundle('test')
        self.test_interacts = self.test_split_interacts['basic']
        self.test_interacts_seen = self.test_split_interacts['seen']
        self.test_interacts_unseen = self.test_split_interacts['unseen']
                
    def __get_validation_dict(self):
        """
        load the list of items that the user has interacted with in the validation set
        :return:
        """
        self.validation_split_interacts = self._load_split_dict_bundle('validation')
        self.validation_interacts = self.validation_split_interacts['basic']
        self.validation_interacts_seen = self.validation_split_interacts['seen']
        self.validation_interacts_unseen = self.validation_split_interacts['unseen']

    def __get_sparse_interact_dict(self):
        """
        load graphs

        :return:
        """
        self.inter_matrix = []
        self.user_item_inter_set = []
        
        all_row = []
        all_col = []
        
        for behavior in self.behaviors:
            
            with open(os.path.join(self.path, behavior + '.txt'), encoding='utf-8') as f:
                data = f.readlines()
                row = []
                col = []
                for line in data:
                    line = line.strip('\n').strip().split()
                    row.append(int(line[0]))
                    col.append(int(line[1]))

                values = torch.ones(len(row), dtype=torch.float32)
                inter_matrix = sp.coo_matrix((values, (row, col)), [self.user_count + 1, self.item_count + 1])

                self.inter_matrix.append(inter_matrix) # inter matrix for all behaviors

                all_row.extend(row)
                all_col.extend(col)

        all_edge_index = list(set(zip(all_row, all_col)))
        all_row = [sub[0] for sub in all_edge_index]
        all_col = [sub[1] for sub in all_edge_index]
        values = torch.ones(len(all_row), dtype=torch.float32)
        self.all_inter_matrix = sp.coo_matrix((values, (all_row, all_col)), [self.user_count + 1, self.item_count + 1])

        with open(os.path.join(self.path, 'aug_r.txt'), encoding='utf-8') as f:
            data = f.readlines()
            row = []
            col = []
            for line in data:
                line = line.strip('\n').strip().split()
                row.append(int(line[0]))
                col.append(int(line[1]))

            values = torch.ones(len(row), dtype=torch.float32)
            inter_matrix = sp.coo_matrix((values, (row, col)), [self.user_count + 1, self.item_count + 1])
            
            self.aug_inter_matrix = inter_matrix # inter matrix for all behaviors
        
        with open(os.path.join(self.path, 'buy_seen.txt'), encoding='utf-8') as f:
            data = f.readlines()
            row = []
            col = []
            for line in data:
                line = line.strip('\n').strip().split()
                row.append(int(line[0]))
                col.append(int(line[1]))

            values = torch.ones(len(row), dtype=torch.float32)
            inter_matrix = sp.coo_matrix((values, (row, col)), [self.user_count + 1, self.item_count + 1])
            
            self.buy_seen_matrix = inter_matrix # inter matrix for all behaviors
        

    def behavior_dataset(self):
        return BehaviorDate(self.user_count, self.item_count, self.pos_sampling, self.neg_count, self.train_behavior_dict, self.behaviors)
    
    def behavior_dataset_seen(self):
        return BehaviorDate(self.user_count, self.item_count, self.pos_sampling, self.neg_count, self.train_behavior_dict, self.behaviors)
    
    def behavior_dataset_unseen(self):
        return BehaviorDate(self.user_count, self.item_count, self.pos_sampling, self.neg_count, self.train_behavior_dict, self.behaviors)
    
    def validate_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.validation_interacts.keys()))
    
    def validate_dataset_seen(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.validation_interacts_seen.keys()))
    
    def validate_dataset_unseen(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.validation_interacts_unseen.keys()))

    def test_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.test_interacts.keys()))
    
    def test_dataset_seen(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.test_interacts_seen.keys()))
    
    def test_dataset_unseen(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.test_interacts_unseen.keys()))

    def get_eval_bundle(self, split_name='test', setting='basic'):
        cache_key = (split_name, setting)
        if cache_key in self._eval_bundle_cache:
            return self._eval_bundle_cache[cache_key]

        split_name = split_name.lower()
        item_segment, base_setting = self._split_item_popularity_setting(setting)
        user_segment, base_setting = self._split_user_activity_setting(base_setting)
        setting_alias = {
            'basic': 'basic',
            'visited': 'seen',
            'seen': 'seen',
            'unvisited': 'unseen',
            'unseen': 'unseen',
        }
        normalized_setting = setting_alias.get(base_setting, base_setting)

        if split_name == 'test':
            split_interacts = self.test_split_interacts
        elif split_name == 'validation':
            split_interacts = self.validation_split_interacts
        else:
            raise ValueError(f'Unsupported split: {split_name}')

        if normalized_setting not in split_interacts:
            raise ValueError(f'Unsupported evaluation setting: {setting}')

        interacts = split_interacts[normalized_setting]
        if user_segment is not None:
            interacts = self._filter_interacts_by_user_segment(interacts, user_segment)
        if item_segment is not None:
            interacts = self._filter_interacts_by_item_segment(interacts, item_segment)
        gt_length = self._get_gt_length(interacts)
        dataset = TestDate(self.user_count, self.item_count, samples=list(interacts.keys()))
        eval_bundle = (dataset, interacts, gt_length)
        self._eval_bundle_cache[cache_key] = eval_bundle
        return eval_bundle


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--behaviors', type=list, default=['cart', 'click', 'collect', 'buy'], help='')
    parser.add_argument('--data_path', type=str, default='./data/Tmall', help='')
    parser.add_argument('--neg_count', type=int, default=1)
    args = parser.parse_args()
    dataset = DataSet(args)
    loader = DataLoader(dataset=dataset.behavior_dataset(), batch_size=5, shuffle=True)
    for index, item in enumerate(loader):
        print(index, '-----', item)
