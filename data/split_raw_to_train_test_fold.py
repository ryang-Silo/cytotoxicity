import os
import sys
import shutil
import numpy as np
import argparse
import pandas as pd

WORKDIR = os.getcwd()
print(WORKDIR)

def splitRawData(args):
    train_split_path = os.path.join(args.outPath, "train")
    test_split_path = os.path.join(args.outPath, "test")
    image_list = os.listdir(args.dataPath)
    print(len(image_list))
    train_cnt = 0
    test_cnt = 0
    train_info = pd.read_csv(os.path.join(args.fold_info_path, "train_fold_" + args.fold + "_info.txt"), delimiter=',', header=None)
    test_info = pd.read_csv(os.path.join(args.fold_info_path, "test_fold_" + args.fold + "_info.txt"), delimiter=',', header=None)

    # split_list = [train_info, test_info]
    # for split in split_list:
    for i in range(len(train_info)):
        compound = train_info.iloc[0]
        label = train_info.iloc[1]

        if label == 1:
            dst_path = os.path.join(train_split_path, 'cytotoxic')
        else:
            dst_path = os.path.join(train_split_path, "non-toxic")
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        compound_src_path = os.path.join(args.dataPath, compound)
        shutil.copytree(compound_src_path, dst_path)
        print(f"copying compound directory of {len(compound_src_path)} to destination of {os.path.join(dst_path, compound)}")
        train_cnt += len(os.path.join(dst_path, compound))

    for i in range(len(test_info)):
        compound = test_info.iloc[0]
        label = test_info.iloc[1]

        if label == 1:
            dst_path = os.path.join(train_split_path, 'cytotoxic')
        else:
            dst_path = os.path.join(train_split_path, "non-toxic")
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        compound_src_path = os.path.join(args.dataPath, compound)
        shutil.copytree(compound_src_path, dst_path)
        print(f"copying testing compound directory of {len(compound_src_path)} to destination of {os.path.join(dst_path, compound)}")
        test_cnt += len(os.path.join(dst_path, compound))

    print(f"training images count: {train_cnt}, testing count: {test_cnt}")




if __name__ == "__main__":
    # mount dataset
    from azureml.core import Dataset
    from azureml.core import Workspace
    ws = Workspace.from_config()
    print(ws.name, ws.location, ws.resource_group, sep='\t')

    # data_folder = './data/cytoData_v2'
    ws.set_default_datastore('workspaceblobcompound')
    ds = ws.get_default_datastore()
    print(ds.name, ds.datastore_type, ds.account_name, ds.container_name)
    ds_paths = [(ds, 'reference_dataset_compounds_v2/')]
    dataset = Dataset.File.from_files(path = ds_paths)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", default=dataset.as_mount(), type=str)
    parser.add_argument("--fold", default='1', type=str)
    parser.add_argument("--split_info_path", default='/home/azureuser/cloudfiles/code/workspace/cytotoxicity/data/train_test_folds', type=str)
    parser.add_argument("--outPath", default=WORKDIR+'/eda_phase/lbp_experiments/data', type=str)
    
    args = parser.parse_args()
    
    splitRawData(args)    