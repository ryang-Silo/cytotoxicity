import os
import sys
import shutil
import numpy as np
import argparse
WORKDIR = os.getcwd()
print(WORKDIR)

def splitRawData(args):
    train_split_path = os.path.join(args.outPath, "train")
    test_split_path = os.path.join(args.outPath, "test")
    image_list = os.listdir(args.dataPath)
    print(len(image_list))
    train_cnt = 0
    test_cnt = 0
    
    for image_name in image_list:
        src_path = os.path.join(args.dataPath, image_name)
        print(image_name)
        if  "-1" in image_name: # training images
            dsn_path = os.path.join(train_split_path, image_name)
            print(f"train: {dsn_path}")
            shutil.copyfile(src_path, dsn_path)
            train_cnt += 1
        elif "-2" in image_name: # test images 
            dsn_path = os.path.join(test_split_path, image_name)
            print(f"test: {dsn_path}")
            shutil.copyfile(src_path, dsn_path)       
            test_cnt += 1

    print(f"training images count: {train_cnt}, testing count: {test_cnt}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", default=os.path.join(WORKDIR, 'eda_phase/data/Cytotox_HCA_065_20230907_B0_TIFFs'), type=str)
    parser.add_argument("--outPath", default=WORKDIR+'/eda_phase/lbp_experiments/data', type=str)
    
    args = parser.parse_args()
    
    splitRawData(args)    