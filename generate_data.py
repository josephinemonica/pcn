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
SAVE_INPUT_PATH_TRAIN = '/home/josephinemonica/extra_hard_disk/data-PCN/my_data/train/input'
SAVE_GT_PATH_TRAIN = '/home/josephinemonica/extra_hard_disk/data-PCN/my_data/train/gt'
SAVE_INPUT_PATH_VAL = '/home/josephinemonica/extra_hard_disk/data-PCN/my_data/val/input'
SAVE_GT_PATH_VAL = '/home/josephinemonica/extra_hard_disk/data-PCN/my_data/val/gt'

BASE_PATH = '/home/josephinemonica/Documents/gpu_link/joint_pose_and_shape_estimation/data/data-PCN/my_data'
SAVE_GT_PATH_VAL_TRANSFORMED = os.path.join(BASE_PATH, 'transformed_test_data', 'gt')
SAVE_INPUT_PATH_VAL_TRANSFORMED = os.path.join(BASE_PATH, 'transformed_test_data', 'input')
SAVE_POSE_PATH = os.path.join(BASE_PATH, 'transformed_test_data', 'pose')
# my_data
#   train
#       input
#       gt
#   val
#       input
#       gt
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
        with open(os.path.join(SAVE_INPUT_PATH_TRAIN, '{:08d}.npy'.format(step)), 'wb') as f:
            np.save(f, inputs.reshape(-1,3))
        # Save gt
        with open(os.path.join(SAVE_GT_PATH_TRAIN, '{:08d}.npy'.format(step)), 'wb') as f:
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
        with open(os.path.join(SAVE_INPUT_PATH_VAL, '{:08d}.npy'.format(step)), 'wb') as f:
            np.save(f, inputs.reshape(-1,3))
        # Save gt
        with open(os.path.join(SAVE_GT_PATH_VAL, '{:08d}.npy'.format(step)), 'wb') as f:
            np.save(f, gt.reshape(-1,3))
            
        assert ids[0] not in id_val_list, 'Double id at step {}, id {}'.format(step, ids[0])
        id_val_list.append(ids[0])
    print('==================================================================')
    print('Finish saving {} val files'.format(num_valid))
    print('==================================================================')
    
    
def get_transformed_test_data(args, xmax=40., zmax=40.,):
    """Make and save a test data for PCN: 
        apply random y rotation & random xz translation
        Save the point cloud, and also the pose"""
    # Speicfy batch size just equals to 1
    
    df_valid, num_valid = lmdb_dataflow(
        args.lmdb_valid, 1, args.num_input_points, args.num_gt_points, is_training=False)
    valid_gen = df_valid.get_data()
    
    print('==================================================================')
    print('We have {} val files'.format(num_valid))
    print('==================================================================')
    ###########################################################################
    # SAVE VALIDATION
    id_val_list = []
    for step in tqdm(range(num_valid)):
        ids, inputs, npts, gt = next(valid_gen)
        # inputs, gt ---- (1,N_input,3), (1,N_gt,3)
        
        # Sanity check
        assert inputs.shape[1] == npts[0], 'number of points do not match'
        
        # (1,?,3) -> (?,3)
        inputs = inputs.reshape(-1,3)
        gt = inputs.reshape(-1,3)
        
        # Random y-rotation
        ang = np.random.uniform(low=0., high=2.*np.pi)
        R = np.eye(3)
        # First col (x)
        R[0,0] = np.cos(ang)
        R[2,0] = -np.sin(ang)
        # Third col (z)
        R[0,2] = np.sin(ang)
        R[2,2] = np.cos(ang)
        
        # Note: before multiplied by rotation need to be (3,N), then
        # move back to (N,3)
        gt_transformed = (R@gt.transpose()).transpose()
        inputs_transformed = (R@inputs.transpose()).transpose()
        
        # Random x-ztranslation
        xtrans = np.random.uniform(low=-xmax, high=xmax)
        ztrans = np.random.uniform(low=-zmax, high=zmax)
        
        gt_transformed[:,0] = gt_transformed[:,0] + xtrans
        gt_transformed[:,2] = gt_transformed[:,2] + ztrans
        inputs_transformed[:,0] = inputs_transformed[:,0] + xtrans
        inputs_transformed[:,2] = inputs_transformed[:,2] + ztrans
        
        # The pose
        pose = np.array([xtrans, 0, ztrans, ang])
        
        # Save input
        with open(os.path.join(SAVE_INPUT_PATH_VAL_TRANSFORMED, '{:08d}.npy'.format(step)), 'wb') as f:
            np.save(f, inputs_transformed)
        # Save gt
        with open(os.path.join(SAVE_GT_PATH_VAL_TRANSFORMED, '{:08d}.npy'.format(step)), 'wb') as f:
            np.save(f, gt_transformed)
        # Save pose
        with open(os.path.join(SAVE_POSE_PATH, '{:08d}.npy'.format(step)), 'wb') as f:
            np.save(f, pose)
            
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

    #get_data(args)
    get_transformed_test_data(args)