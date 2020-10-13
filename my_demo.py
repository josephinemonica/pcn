#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 15:33:14 2020

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


def test(args):
    inputs = tf.placeholder(tf.float32, (1, None, 3))
    npts = tf.placeholder(tf.int32, (1,))
    print('AAAAAAAAAAAAAAAA')
    print(args.num_gt_points)
    print('AAAAAAAAAAAAAAAAAAAAAAAAAa')
    gt = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs, npts, gt, tf.constant(1.0))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    t0 = time.time()

    partial = read_pcd(args.pcd_file)
    bbox = np.loadtxt(args.bbox_file)

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

    start = time.time()
    completion = sess.run(model.outputs, feed_dict={inputs: [partial], npts: [partial.shape[0]]})
    
    completion = completion[0]

    completion_w = np.dot(completion, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    completion_w = np.dot(completion_w * scale, rotation.T) + center
    
    dummy = args.pcd_file.split('/')[-1]
    result_path = os.path.join(args.results_dir, dummy)
    save_pcd(result_path, completion_w)
    
    plot_path = os.path.join(args.results_dir, 'plots.png')
    plot_pcd_three_views(plot_path, [partial, completion], ['input', 'output'],
                         '%d input points' % partial.shape[0], [5, 0.5])
    sess.close()
    tf = time.time()
    print('Total time: {}'.format(tf-t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='pcn_emd')
    parser.add_argument('--checkpoint', default='data/trained_models/pcn_emd_car')
    #parser.add_argument('--pcd_dir', default='data/kitti/cars')
    #parser.add_argument('--bbox_dir', default='data/kitti/bboxes')
    #parser.add_argument('--results_dir', default='results/kitti_pcn_emd')
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--plot_freq', type=int, default=1)
    parser.add_argument('--save_pcd', action='store_true')
    
    parser.add_argument('--pcd_file', default='data/my_kitti/cars/lidar_accum.pcd')
    parser.add_argument('--bbox_file', default='data/my_kitti/bboxes/lidar_accum_bbox.txt')
    parser.add_argument('--results_dir', default='results/my_kitti')
    args = parser.parse_args()
    # NOTE if you want to change num_gt_points: in pcn_emd.py look at
    #self.grid_size = 4
    #self.grid_scale = 0.05
    #self.num_fine = self.grid_size ** 2 * self.num_coarse
    test(args)
