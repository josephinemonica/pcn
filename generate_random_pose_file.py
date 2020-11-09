#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 15:18:34 2020

@author: josephinemonica
"""

import os
import numpy as np

save_pose_path = '/home/josephinemonica/Documents/gpu_link/joint_pose_and_shape_estimation/data/data-PCN/my_data'

# 1: xz translation only, 2: y rotation only, 3: xz translation and y rotation
MODE = 3
N = 100
xmax = 40.
zmax= 40.

if MODE==1:
    # 1: translation only
    name = 'pose_xz_translation'
    
    angs = np.zeros((N,1))
    xtrans = np.random.uniform(low=-xmax, high=xmax, size=(N,1))
    ztrans = np.random.uniform(low=-zmax, high=zmax, size=(N,1))
    
elif MODE==2:
    # 2: rotation only (x-z)
    name = 'pose_y_rotation'

    angs = np.random.uniform(low=0., high=2*np.pi, size=(N,1))
    xtrans = np.zeros((N,1))
    ztrans = np.zeros((N,1))

elif MODE==3:
    # 3: translation and rotation
    name = 'pose_xz_translation_y_rotation'
    
    angs = np.random.uniform(low=0., high=2*np.pi, size=(N,1))
    xtrans = np.random.uniform(low=-xmax, high=xmax, size=(N,1))
    ztrans = np.random.uniform(low=-zmax, high=zmax, size=(N,1))
else:
    print('Invalid MODE')

ytrans = np.zeros((N,1))
pose_gt = np.hstack((xtrans, ytrans, ztrans, angs))
with open(os.path.join(save_pose_path, name), 'wb') as f:
    np.save(f, pose_gt)
