#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:19:29 2020

@author: josephinemonica
"""
import os
import numpy as np

np.random.seed(900)
name = 'pose1'
name = 'pose_trans'
name = 'pose_rot'
save_pose_path = '/home/josephinemonica/Documents/gpu_link/joint_pose_and_shape_estimation/data/data-PCN/my_data'
save_pose_path = os.path.join(save_pose_path, name)

if __name__=='__main__':
    if not os.path.isdir(save_pose_path):
        os.makedirs(save_pose_path)
        
    N = 100
    xmax = 40.
    zmax= 40.
    
    """
    # translation only
    angs = np.zeros(N)
    xtrans = np.random.uniform(low=-xmax, high=xmax, size=N)
    ztrans = np.random.uniform(low=-zmax, high=zmax, size=N)
    """
    
    
    # Rotation only
    angs = np.random.uniform(low=0., high=2*np.pi, size=N)
    xtrans = np.zeros(N)
    ztrans = np.zeros(N)
    for i in range(N):
        pose_gt = np.array([xtrans[i], 0, ztrans[i], angs[i]])
        with open(os.path.join(save_pose_path, '{:08d}.npy'.format(i)), 'wb') as f:
            np.save(f, pose_gt)