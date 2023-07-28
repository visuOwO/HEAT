import os
import numpy as np
import argparse
import time
import torch
import mpi4py

from behavior_aggregators import AggregatorWeights
from datasets import ClickDataset, SubClickDataset
from models import MatrixFactorization
from cf_config import CFConfig
from train import Engine

import metrics
import cf_c

import utils

if __name__ == "__main__":
    print('this is main ...')

    MPI = mpi4py.MPI
    mpi4py.rc.threaded = True
    mpi4py.rc.thread_level = "funneled"
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./benchmarks/AmazonBooks/MF_CCL/configs/config0.yaml',
                        help='The config file for para config.')
    args = parser.parse_args()

    config_path = args.config
    config_dic = utils.load_config(config_path)
    dataset_config = config_dic['dataset_config']
    model_config = config_dic['model_config']
    print(model_config)

    cf_config = CFConfig(emb_dim=model_config['embedding_dim'], num_negs=model_config['num_negs'],
                         max_his=model_config['max_his'], neg_sampler=model_config['neg_sampler'],
                         tile_size=model_config['tile_size'], refresh_interval=model_config['refresh_interval'],
                         l2=model_config['embedding_regularizer'], clip_val=model_config['clip_val'],
                         milestones=model_config['milestones'], l_r=model_config['learning_rate'])

    test = cf_c.modules.test.test_out()
    if rank == 0:
        print('--- Start loading data --- in ' + str(rank))
        train_file = os.path.join(dataset_config['data_dir'], dataset_config['train_data'])
        train_data = ClickDataset(train_file, separator=dataset_config['separator'], config=cf_config)
        k = train_data.num_users // size
        r = train_data.num_users % size
        kk = train_data.num_items // size
        rr = train_data.num_items % size
        print('--- Finished loading data from file ... ---')

        for i in range(1, size):
            start = i * k + min(i, r)
            end = start + k + (i < r)
            col_start = i * kk + min(i, rr)
            col_end = col_start + kk + (i < rr)
            # test.test(str(start) + " is the start index")
            # test.test(str(end) + " is the end index")
            sub_dataset = SubClickDataset(train_data, start, end, col_start, col_end, i)
            MPI.COMM_WORLD.send(sub_dataset, dest=i, tag=11)
            # test.test(str(len(sub_dataset.user_item_ids)) + " is the length of sub user_item_ids of rank " + str(i))
        # test.test("0 is the starting index")
        # test.test(str(k + min(0, r)) + " is the end index")
        new_dataset = SubClickDataset(train_data, 0, k + (0 < r), 0, kk + (0 < rr), rank)
        # print('--- Finished dividing data ...  start sending data ... ---')
        # test.test(str(len(new_dataset.user_item_ids)) + " is the length of sub user_item_ids of rank " + str(0))
        train_data = new_dataset
    else:
        # print('-- Start receiving data -- in ' + str(rank))
        sub_dataset = MPI.COMM_WORLD.recv(source=0, tag=11)
        train_data = sub_dataset

    # print('--- Finished loading data --- in ' + str(rank))
    train_data.update_config(cf_config)

    cf_config.init_c_instance()

    if rank == 0:
        test_file = os.path.join(dataset_config['data_dir'], dataset_config['test_data'])
        test_data = ClickDataset(test_file, separator=dataset_config['separator'], config=cf_config)
        k = test_data.num_users // size
        r = test_data.num_users % size

        kk = test_data.num_items // size
        rr = test_data.num_items % size

        for i in range(1, size):
            start = i * k + min(i, r)
            end = start + k + (i < r)
            col_start = i * kk + min(i, rr)
            col_end = col_start + kk + (i < rr)
            # test.test(str(start) + " is the start index")
            # test.test(str(end) + " is the end index")
            sub_dataset = SubClickDataset(test_data, start, end, col_start, col_end, i)
            MPI.COMM_WORLD.send(sub_dataset, dest=i, tag=11)
        # test.test("0 is the starting index")
        new_dataset = SubClickDataset(test_data, 0, k + (0 < r), 0, kk + (0 < rr), rank)
        # print('--- Finished dividing data ...  start sending data ... ---')
        test_data = new_dataset
    else:
        # print('-- Start receiving data -- in ' + str(rank))
        sub_dataset = MPI.COMM_WORLD.recv(source=0, tag=11)
        test_data = sub_dataset

    aggregator_weights = AggregatorWeights(cf_config)
    model = MatrixFactorization(cf_config)
    model.init_c_instance(cf_config)

    engine = Engine(train_data, aggregator_weights, model, cf_config)

    eval_interval = model_config['eval_interval']
    for epoch in range(model_config['epochs']):
        start_time = time.time()

        test.test("start training epoch")
        epoch_loss = engine.train_one_epoch()

        epoch_time = time.time() - start_time
        print(f'epoch: {epoch}; loss: {epoch_loss}; epoch_time: {epoch_time}')
        test.test(f'epoch: {epoch}; loss: {epoch_loss}; epoch_time: {epoch_time}')
        if epoch > 0 and epoch % eval_interval == 0:
            print('--- Start evaluation ---')
            test.test("start evaluation")
            model.eval()
            with torch.no_grad():
                eva_metrics = ['Recall(k=20)']
                sim_matrix = engine.evaluate0()
                print(f'sim_matrix shape: {np.shape(sim_matrix)} !!! ')

                metrics.evaluate_metrics(train_data, test_data, sim_matrix, eva_metrics)
