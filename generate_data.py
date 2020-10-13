#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 22:21:07 2020

@author: josephinemonica
"""

# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import argparse
import os
from data_util import lmdb_dataflow
import numpy as np
SAVE_DATA_PATH = '/home/josephinemonica/extra_hard_disk/data-PCN/my_data'
from tqdm import tqdm

def get_data(args):

    # Speicfy batch size just equals to 1
    #NOTE: specifically put is_training to false, because if I put
    # is_training=True, it will result to double ids in an epoch for some reason idk
    df_train, num_train = lmdb_dataflow(
        args.lmdb_train, 1, args.num_input_points, args.num_gt_points, is_training=False)
    train_gen = df_train.get_data()
    df_valid, num_valid = lmdb_dataflow(
        args.lmdb_valid, 1, args.num_input_points, args.num_gt_points, is_training=False)
    valid_gen = df_valid.get_data()
    
    print('==================================================================')
    print('We have {} train files and {} val files'.format(num_train, num_valid))
    print('==================================================================')
    ###########################################################################
    # SAVE TRAINING
    id_train_list = []
    for step in tqdm(range(num_train)):
        ids, inputs, npts, gt = next(train_gen)
        
        # Sanity check
        assert inputs.shape[1] == npts[0], 'number of points do not match'
        # Save input
        with open(os.path.join(SAVE_DATA_PATH, 'train_input_{}.npy'.format(step)), 'wb') as f:
            np.save(f, inputs.reshape(-1,3))
        # Save gt
        with open(os.path.join(SAVE_DATA_PATH, 'train_gt_{}.npy'.format(step)), 'wb') as f:
            np.save(f, gt.reshape(-1,3))
            
        #assert ids[0] not in id_train_list, 'Double id at step {}, id {}'.format(step, ids[0])
        if ids[0] in id_train_list:
            print('step {} is the same with step {}'.format(step, id_train_list.index(ids[0])))
        id_train_list.append(ids[0])    
    print('==================================================================')
    print('Finish saving {} train files'.format(num_train))
    print('==================================================================')
    ###########################################################################
    # SAVE VALIDATION
    id_val_list = []
    for step in tqdm(range(num_valid)):
        ids, inputs, npts, gt = next(valid_gen)
        
        # Sanity check
        assert inputs.shape[1] == npts[0], 'number of points do not match'
        # Save input
        with open(os.path.join(SAVE_DATA_PATH, 'val_input_{}.npy'.format(step)), 'wb') as f:
            np.save(f, inputs.reshape(-1,3))
        # Save gt
        with open(os.path.join(SAVE_DATA_PATH, 'val_gt_{}.npy'.format(step)), 'wb') as f:
            np.save(f, gt.reshape(-1,3))
            
        assert ids[0] not in id_val_list, 'Double id at step {}, id {}'.format(step, ids[0])
        id_val_list.append(ids[0])
    print('==================================================================')
    print('Finish saving {} val files'.format(num_valid))
    print('==================================================================')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='data/shapenet/train.lmdb')
    parser.add_argument('--lmdb_valid', default='data/shapenet/valid.lmdb')
    parser.add_argument('--num_input_points', type=int, default=3000)
    parser.add_argument('--num_gt_points', type=int, default=16384)
    args = parser.parse_args()

    get_data(args)
