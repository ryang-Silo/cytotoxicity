import os
import sys
import shutil
import numpy as np
import argparse
import pandas as pd

WORKDIR = os.getcwd()
print(WORKDIR)

reference_concentrations = [200, 150.0, 112.6, 84.4, 63.3, 47.5, 35.6, 26.7, 20.1, 10.0, 5.0, 2.5, 1.3, 0.6, 0.3, 0.2]

def find_toxic_row(compound_name, thres):
    if thres > 0:
        row = find_closest_indices([thres], reference_concentrations)
    else:
        print(f'Threshold should be greater than 0.0. This {compound_name} with {thres} should be in non-toxic!')
    return row 

def find_closest_indices(result_thres, concentrations):
    concentrations_array = np.asarray(concentrations)
    indices = []
    if len(result_thres) == 1:
        differences = np.abs(concentrations_array - result_thres)
        closest_index = np.argmin(differences)
        return closest_index
    else:
        for value in result_thres:
            differences = np.abs(concentrations_array - value)
            closest_index = np.argmin(differences)
            indices.append(closest_index)
        return indices


def copy_images_with_prefix(src_folder, dst_folder, img_prefix, compound):
    # Ensure the destination folder exists
    os.makedirs(dst_folder, exist_ok=True)
    
    # Iterate through files in the source folder
    for imgname in os.listdir(src_folder):
        if imgname.startswith(img_prefix):
            src_path = os.path.join(src_folder, imgname)
            new_image_name = f"{compound}_{imgname}"
            dst_path = os.path.join(dst_folder, new_image_name)
            # dst_path = os.path.join(dst_folder, filename)
            shutil.copy(src_path, dst_path)
            print(f"Copied {imgname} to {dst_folder}")


def splitRawData_with_imglabel(args, process):
    os.makedirs(args.outPath, exist_ok=True)
    output_split_path = os.path.join(args.outPath, process)
    info_df = pd.read_csv(os.path.join(args.split_info_path, process + "_fold_" + args.fold + "_info_imglevel.txt"), delimiter=',', header=None)

    for i in range(len(info_df)):
        compound = info_df.iloc[i, 0]
        label = info_df.iloc[i, 2]
        concentration_thres = info_df.iloc[i, 3]

        compound_src_path = os.path.join(args.dataPath, compound)
        # for non-toxic class
        if label == 0 and concentration_thres == 0.0:
            dst_path = os.path.join(output_split_path, "non-toxic")
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            if os.path.isdir(compound_src_path):
                for image_name in os.listdir(compound_src_path):
                    if image_name.endswith(".tif"):
                        image_path = os.path.join(compound_src_path, image_name)
                        new_image_name = f"{compound}_{image_name}"
                        destination_image_path = os.path.join(dst_path, new_image_name)
                        shutil.copy(image_path, destination_image_path)
                        print(f"copying {image_path} to destination {destination_image_path}")

        # for cytotoxic and moderate affect cell healthy classes, need concentration info to determine the label
        elif label == 1 or label == 2:
            toxic_row_num = find_toxic_row(compound, concentration_thres)
            if toxic_row_num:
                # check cyto, moderate case with 200 thres
                if toxic_row_num == 0: # concentration = 200
                    # ? move to non-toxic?
                    dst_path = os.path.join(output_split_path, "non-toxic")
                else:
                    # split images to non-toxic and cytotoxic classes, according to threshold value (row index)
                    for row_ind in range(16):
                        img_prefix = "%03d" % (row_ind + 1)
                        print(f"compound:{compound}, image prefix: {img_prefix}")
                        
                        # those concentration below LEC, should be non-toxic; row_ind greater, lower concentration
                        if row_ind <= toxic_row_num:
                            print(f"result thres: {concentration_thres}, reference concentration: {reference_concentrations[toxic_row_num]}")
                            dst_path = os.path.join(output_split_path, "cytotoxic")
                        else:
                            # non-toxic concentrations
                            dst_path = os.path.join(output_split_path, "non-toxic")
                        
                        # if not os.path.exists(dst_path):
                        #     os.makedirs(dst_path, exist_ok=True)

                        # Copy images with the current prefix to the destination path
                        copy_images_with_prefix(compound_src_path, dst_path, img_prefix, compound)
        else:
            print(f"Unexpected classes: {label} with compound: {compound}")
      
    # print(f"{process} images count: {image_cnt}")



if __name__ == "__main__":
    # # mount dataset
    # from azureml.core import Dataset
    # from azureml.core import Workspace
    # from azure.storage.blob import BlobServiceClient
    # ws = Workspace.from_config()
    # print(ws.name, ws.location, ws.resource_group, sep='\t')

    # ws.set_default_datastore('workspaceblobcompound')
    # ds = ws.get_default_datastore()
    # print(ds.name, ds.datastore_type, ds.account_name, ds.container_name)
    # ds_paths = [(ds, 'reference_dataset_compounds_v2/')]
    # dataset = Dataset.File.from_files(path = ds_paths)
    
    # mount data
    mounted_path = os.path.join(WORKDIR, 'workspace/cytotoxicity/data/mounted_cytoData2')
    # mount_context = dataset.mount(mounted_path)
    # mount_context.start()
    # print(f"Data mounted to: {mounted_path}")
    # # print(os.listdir(mounted_path))

    # if os.path.exists(mounted_path):
    #     print(f"Mount point {mounted_path} exists.")
    # else:
    #     print(f"Mount point {mounted_path} does not exist.")
    #     mount_context.stop()
    #     raise Exception("Mount point not found. Please check your dataset and try again.")
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", default=mounted_path, type=str)
    parser.add_argument("--fold", default='4', type=str)
    parser.add_argument("--split_info_path", default=os.path.join(WORKDIR, 'workspace/cytotoxicity/data/train_test_folds'), type=str)
    parser.add_argument("--outPath", default=os.path.join(WORKDIR, 'workspace/cytotoxicity/data/processed_traintest_data_fold1_v2'), type=str)
    
    args = parser.parse_args()
    
    
    process_list = ["test", "train"]
    for process in process_list:
        splitRawData_with_imglabel(args, process)