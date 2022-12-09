#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# ORT analysis - script training data - developed by Victor Ibañez
# 03.04.2021
# -------------------------------------------------------------------------------------


# -----------------------------------------------------
# import libraries
# -----------------------------------------------------  

import os
import glob
import random
import math
import shutil

# -----------------------------------------------------
# create training data
# -----------------------------------------------------

def create_training_data(source_dir,target_dir,objects):

    # iterate through raw data
    def collect_candidates(folder_path):
        
        candidates = []
        
        for filename in glob.iglob(folder_path + '/**/*.jpg', recursive=True):
            candidates.append(filename)
        
        return candidates

    # redesign names and image size
    def create_images(candidates, imgsave_path, file_prefix):
        
        cnt = 0
        for filename in candidates:
            cnt += 1
            new_filename = os.path.join(imgsave_path, '{0}_{1}.jpg'.format(file_prefix, cnt))
            shutil.copy(filename, new_filename)
     
    # create data randomly and splitting into training, test and validation     
    def process(source_dir, target_dir, dist_training, dist_validation):
        
        objects.append('no')

        distributions = {'training': dist_training, 'validation': dist_validation}
            
        for cls in objects:
            candidates = collect_candidates(os.path.join(source_dir, cls))
            random.shuffle(candidates)
            
            offset = 0
            for key, percentage in distributions.items(): 
                
                share = math.floor(len(candidates) / 100 * percentage)
                
                class_path = os.path.join(target_dir, key, cls)
                if not os.path.isdir(class_path): 
                    os.makedirs(class_path)
                
                create_images(candidates[offset:offset+share], class_path, cls)
                offset += share

    process(source_dir,target_dir,80,20)
