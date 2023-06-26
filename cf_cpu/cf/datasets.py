import numpy as np
import random
import mpi4py.MPI as MPI

from cpp_base import CPPBase
import cf_c


class Dataset(CPPBase):
    def __init__(self):
        super().__init__()


class ClickDataset(Dataset):
    def __init__(self, file_path, separator=' ', config=None):
        super().__init__()
        self.c_class = cf_c.modules.datasets.ClickDataset

        self.file_path = file_path
        self.user_items_dic = {}
        self.user_item_ids = []
        self.user_ids_dic = {}
        self.item_ids_dic = {}
        self.num_users = 0
        self.num_items = 0

        self.max_his = config.max_his
        self.his_items = None
        self.masks = None

        with open(file_path, mode='r') as in_file:
            lines = in_file.readlines()
            self.his_items = np.zeros((len(lines), self.max_his), dtype=np.uint64)
            self.masks = np.zeros((len(lines), 1), dtype=np.uint64)

            for line in lines:
                splits = line.strip().split(separator)
                user_id = int(splits[0])
                items = splits[1:]
                items = list(map(int, items))

                if user_id not in self.user_ids_dic:
                    self.user_ids_dic[user_id] = user_id

                self.user_items_dic[user_id] = items

                if len(items) >= self.max_his:
                    user_his = random.sample(items, self.max_his)
                    self.his_items[user_id] = user_his
                    self.masks[user_id] = [self.max_his]
                elif len(items) > 0:
                    user_his = items.copy()
                    user_his.extend([user_his[-1]] * (self.max_his - len(items)))
                    self.his_items[user_id] = user_his
                    self.masks[user_id] = [len(items)]
                else:
                    user_his = []
                    user_his.extend([0] * self.max_his)
                    self.his_items[user_id] = user_his
                    self.masks[user_id] = [0]
                    print(f"Warning {user_id} has 0 items !!! ")

                for item in items:
                    if item not in self.item_ids_dic:
                        self.item_ids_dic[item] = item

                    self.user_item_ids.append([user_id, item])

        self.gen_dataset_info()

        if 'train' in file_path:
            print('update config, init c_instance !!! ')
            config.num_users = self.num_users
            config.num_items = self.num_items
            config.train_size = len(self.user_item_ids)

            self.click_dataset = np.array(self.user_item_ids, dtype=np.uint64)
            self.init_c_instance(click_dataset=self.click_dataset, historical_items=self.his_items, masks=self.masks)
            self.c_instance.max_his = self.max_his

        print('\n')

    def gen_dataset_info(self):
        print(f'gen dataset info of {self.file_path} ')
        self.num_users = len(self.user_ids_dic)
        self.num_items = len(self.item_ids_dic)
        user_ids = list(self.user_ids_dic.keys())
        item_ids = list(self.item_ids_dic.keys())
        # user_ids.sort()
        # item_ids.sort()
        max_user_id = max(user_ids)
        min_user_id = min(user_ids)
        max_item_id = max(item_ids)
        min_item_id = min(item_ids)
        if max_user_id - min_user_id + 1 != self.num_users:
            print('Warning user_id is not continuous! ')

        if max_item_id - min_item_id + 1 != self.num_items:
            print('Warning item_id is not continuous! ')

        print(f'number of users: {self.num_users}; min_user_id: {min_user_id}; max_user_id: {max_user_id}')
        print(f'number of items: {self.num_items}; min_item_id: {min_item_id}; max_item_id: {max_item_id}')
        print(f'total samples: {len(self.user_item_ids)} ')
        # print('\n')

    def get_user_items(self):
        return self.user_items_dic


class SubClickDataset(Dataset):
    def __init__(self, parent_dataset, start, end, rank=None):
        super().__init__()
        self.click_dataset = None

        test = cf_c.modules.test.test_out()
        self.c_class = None

        self.file_path = parent_dataset.file_path
        self.user_items_dic = {}
        for user_id in parent_dataset.get_user_items():
            if start <= user_id < end:
                self.user_items_dic[user_id - start] = parent_dataset.get_user_items()[user_id]

        self.user_item_ids = []
        self.user_ids_dic = {}
        self.item_ids_dic = {}
        self.num_users = 0
        self.num_items = 0
        self.rank = rank

        for user_id in self.user_items_dic:
            items = self.user_items_dic[user_id]
            for item in items:
                if item not in self.item_ids_dic:
                    self.item_ids_dic[item] = item
                self.user_item_ids.append([user_id, item])
            self.user_ids_dic[user_id] = user_id

        self.max_his = parent_dataset.max_his
        self.his_items = np.zeros((len(self.user_ids_dic), self.max_his), dtype=np.uint64)
        self.masks = np.zeros((len(self.user_ids_dic), 1), dtype=np.uint64)

        for user_id in self.user_items_dic:
            items = self.user_items_dic[user_id]
            if len(items) >= self.max_his:
                user_his = random.sample(items, self.max_his)
                self.his_items[user_id] = user_his
                self.masks[user_id] = [self.max_his]
            elif len(items) > 0:
                user_his = items.copy()
                user_his.extend([user_his[-1]] * (self.max_his - len(items)))
                self.his_items[user_id] = user_his
                self.masks[user_id] = [len(items)]
        print(f'number of users: {len(self.user_ids_dic)}; number of items: {len(self.item_ids_dic)}; total samples: {len(self.user_item_ids)} ')
        self.inherit_dataset_info(parent_dataset)

    def inherit_dataset_info(self, parent_dataset):
        self.num_users = len(self.user_ids_dic)
        self.num_items = parent_dataset.num_items
        user_ids = list(self.user_ids_dic.keys())
        item_ids = list(self.item_ids_dic.keys())

        max_user_id = max(user_ids)
        min_user_id = min(user_ids)
        max_item_id = max(item_ids)
        min_item_id = min(item_ids)

        if max_user_id - min_user_id + 1 != self.num_users:
            print('Warning user_id is not continuous! ')

        if max_item_id - min_item_id + 1 != self.num_items:
            print('Warning item_id is not continuous! ')

        print(f'number of users: {self.num_users}; min_user_id: {min_user_id}; max_user_id: {max_user_id}')
        print(f'number of items: {self.num_items}; min_item_id: {min_item_id}; max_item_id: {max_item_id}')
        print(f'total samples: {len(self.user_item_ids)} ')

    def gen_dataset_info(self):
        print(f'gen dataset info in process {self.rank} of {self.file_path}')
        self.num_users = len(self.user_ids_dic)
        self.num_items = len(self.item_ids_dic)
        user_ids = list(self.user_ids_dic.keys())
        item_ids = list(self.item_ids_dic.keys())
        # user_ids.sort()
        # item_ids.sort()
        max_user_id = max(user_ids)
        min_user_id = min(user_ids)
        max_item_id = max(item_ids)
        min_item_id = min(item_ids)
        if max_user_id - min_user_id + 1 != self.num_users:
            print('Warning user_id is not continuous! ')

        if max_item_id - min_item_id + 1 != self.num_items:
            print('Warning item_id is not continuous! ')

        print(f'number of users: {self.num_users}; min_user_id: {min_user_id}; max_user_id: {max_user_id}')
        print(f'number of items: {self.num_items}; min_item_id: {min_item_id}; max_item_id: {max_item_id}')
        print(f'total samples: {len(self.user_item_ids)} ')

    def update_config(self, config=None):
        print('update config, init c_instance !!! ')
        self.c_class = cf_c.modules.datasets.ClickDataset
        config.num_users = self.num_users
        config.num_items = self.num_items
        config.train_size = len(self.user_item_ids)

        test = cf_c.modules.test.test_out()

        self.click_dataset = np.array(self.user_item_ids, dtype=np.uint64)
        test.test(str(self.click_dataset.shape))
        self.init_c_instance(click_dataset=self.click_dataset, historical_items=self.his_items, masks=self.masks)
        self.c_instance.max_his = self.max_his

        self.gen_dataset_info()

        print('\n')


if __name__ == "__main__":
    pass
