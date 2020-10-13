#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:33:08 2020

@author: josephinemonica
"""
import os
import numpy as np

class Analyzer():
    '''
    Analyze for error of all track all car and all time
    '''
    def __init__(self, eval_path_fun):
        # To use self.get_eval_path(data_type,track_no,car_no)
        # lambda fuunction
        self.get_eval_path = eval_path_fun
        
        self.poses = np.empty((0,4))
        self.errors = np.empty((0,3))
        
    def get_datatype_trackno_carno(self,data_types = ["train","val"]):
        '''
        argument_list   list of tuple ("data_type", track_no, car_no)
        '''
        # Get data_type, track_no, car_no
        argument_list = []
        
        for data_type in data_types:
            path_ = self.get_eval_path(data_type)
            
            # Get all tracks
            track_number_string_list = os.listdir(path_)
            track_number_list = list(map(int,track_number_string_list))
            track_number_list.sort()
            
            for track_no in track_number_list:
                
                # Get all cars inside track
                track_path = self.get_eval_path(data_type,track_no)
                
                # All car numbers(in string) contains -001 which is invalid
                car_number_string_list = os.listdir(track_path)
                car_number_list = list(map(int,car_number_string_list))
                car_number_list.sort()
                
                car_number_list = [car_no for car_no in car_number_list if car_no >=0]
                
                for car_no in car_number_list:
                    argument_list.append((data_type,track_no,car_no))
                    
        return argument_list 