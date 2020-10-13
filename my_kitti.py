#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:15:13 2020

@author: josephinemonica
"""
import argparse
import importlib
import models
import numpy as np
import os
import tensorflow as tf
import time
from io_util import read_pcd, save_pcd
from visu_util import plot_pcd_three_views
import sys
sys.path.append('/home/josephinemonica/Documents/estimation_tutorial/PseudoLidar_evaluator')
import directory as dr
from my_kitti_util import Analyzer 

def test(args):
    inputs = tf.placeholder(tf.float32, (1, None, 3))
    npts = tf.placeholder(tf.int32, (1,))
    gt = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs, npts, gt, tf.constant(1.0))
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    
    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)
        
    an = Analyzer(dr.get_PLIDAR_predicted_path)
    argument_list = an.get_datatype_trackno_carno(data_types=["train"])
    
    for i in range(len(argument_list)):
        data_type, track_no, car_no = argument_list[i]
        # Input - partial point cloud .pcd
        pcd_file = \
            dr.get_pcn_lidar_reference_partial_path(data_type, track_no, 
                                                    car_no, extension='pcd')
        # Input - bbox 8 corners .txt
        bbox_file = dr.get_pcn_bbox_lidar_path(data_type, track_no, car_no)
        
        # Result - complete point cloud .pcd
        result_path = dr.get_pcn_lidar_reference_complete_path(data_type,
                                                               track_no, car_no)
        # Result - plot of input and output of pointcloud
        plot_path = dr.get_pcn_plot_lidar_path(data_type, track_no, car_no)

        partial = read_pcd(pcd_file)
        bbox = np.loadtxt(bbox_file)

        # Calculate center, rotation and scale
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale

        partial = np.dot(partial - center, rotation) / scale
        partial = np.dot(partial, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        completion = sess.run(model.outputs, feed_dict={inputs: [partial], npts: [partial.shape[0]]})
        completion = completion[0]

        completion_w = np.dot(completion, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        completion_w = np.dot(completion_w * scale, rotation.T) + center
        
        save_pcd(result_path, completion_w)
    
        plot_pcd_three_views(plot_path, [partial, completion], 
                             ['input', 'output'], 
                             '%d input points' % partial.shape[0], [5, 0.5])
        
        print('Finish {}/{}'.format(i+1, len(argument_list)))
        
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='pcn_emd')
    parser.add_argument('--checkpoint', default='data/trained_models/pcn_emd_car')
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--plot_freq', type=int, default=1)
    parser.add_argument('--save_pcd', action='store_true')

    args = parser.parse_args()
    # NOTE if you want to change num_gt_points: in pcn_emd.py look at
    #self.grid_size = 4
    #self.grid_scale = 0.05
    #self.num_fine = self.grid_size ** 2 * self.num_coarse
    test(args)
