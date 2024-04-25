import os
import sys
import numpy as np
import cv2
import json
from pathlib import Path
import argparse   
import shutil


WORKDIR = os.getcwd()
print(f"working dir: {WORKDIR}")


def target_function(args):
    base_dir = args.dataPath
    # target_dir = args.outPath
    batch_name = os.listdir(base_dir)
    for batch in batch_name:
        batch_dir = os.path.join(base_dir, batch)
        for root, dirs, files in os.walk(batch_dir, topdown=False):
            print(root)
            for name in files:
                if name.endswith('.tif'):
                    new_path = os.path.join(batch_dir, name)
                    old_path = os.path.join(root, name)
            
                    shutil.move(old_path, new_path)
                    print(f'Moved {old_path} to {new_path}')

        for root, dirs, files in os.walk(batch_dir, topdown=False):
            if len(os.listdir(root)) < 3: # remove the xml, csv files and directory
                shutil.rmtree(root)
                print(f'Removed empty directory: {root}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", default=os.path.join(WORKDIR, 'cytotox/data/'), type=str)
    parser.add_argument("--outPath", default=os.path.join(WORKDIR, 'cytotox/pre1_data/'), type=str)
    
    args = parser.parse_args()
    
    target_function(args)