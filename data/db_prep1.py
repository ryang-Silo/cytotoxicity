import os
import sys
import numpy as np
from pathlib import Path
import argparse   
import shutil


WORKDIR = os.getcwd()
print(f"working dir: {WORKDIR}")


def target_function(args):

    base_dir = args.dataPath
    target_dir = args.outPath
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    existinglist = ['240219GA24B0', '240219GA24A0', '230907GA24A0', '230907GA24B0']
    # filtered_files = [file for file in os.listdir(base_dir) if not file.startswith('._')]

    batch_name = [name for name in os.listdir(base_dir) if (name not in existinglist) and (not name.startswith('._'))]
    print(batch_name)
    print(len(batch_name))
    for batch in batch_name:
        img_cnt = 0
        batch_dir = os.path.join(base_dir, batch)
        target_batch_dir = os.path.join(target_dir, batch)
        if not os.path.exists(target_batch_dir):
            os.makedirs(target_batch_dir, exist_ok=True)
        # else:
        #     # go to next batch if it is already processed
        #     if len(os.listdir(root)) == len(os.listdir(target_batch_dir)):
        #         print(f"already processed for {batch_dir}")
        #         continue
        
        for root, dirs, files in os.walk(batch_dir, topdown=False):
            print(root)
            for name in files:
                if name.endswith('.tif') and (not name.startswith('._')):
                    new_path = os.path.join(target_batch_dir, name)
                    old_path = os.path.join(root, name)
            
                    shutil.copy(old_path, new_path)
                    # print(f'Copied {old_path} to {new_path}')
                    img_cnt += 1

        # for root, dirs, files in os.walk(batch_dir, topdown=False):
        #     if len(os.listdir(root)) < 3: # remove the xml, csv files and directory
        #         shutil.rmtree(root)
        #         print(f'Removed empty directory: {root}')
        
        print(f"For batch {batch}, in total processed {img_cnt} images!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", default=os.path.join(WORKDIR, 'cytotox/data/'), type=str)
    parser.add_argument("--outPath", default=os.path.join(WORKDIR, 'cytotox/pre1_data/'), type=str)
    
    args = parser.parse_args()
    
    target_function(args)