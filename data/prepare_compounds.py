"""
To extract all images according to the same compounds
"""
import os
import sys
import numpy as np
import cv2
import json
from pathlib import Path
import argparse
import pandas as pd 
import shutil

WORKDIR = "/Volumes/KINGSTON/Orion"#os.getcwd()
print(f"working dir: {WORKDIR}")

def split_images_by_compound(args):
    plate_data = args.dataPath
    plate_maps = args.platemapPath
    # result_path = args.resultPath
    output_path = args.outPath
    
    plates = os.listdir(plate_data)
    print(f"There are in total {len(plates)} plates")
    plate_compound = get_plate_compound_map(plate_maps)
    
    compound_num = 0
    # for one plate data
    for plate in plates:
        print(f"plate number is {plate}")
        compound_list = plate_compound[plate]
        plate_dir = os.path.join(plate_data, plate)
        image_list = os.listdir(plate_dir)
        
        # loop over all the compounds in one plate
        for cp_ind, compound in enumerate(compound_list):
            # if compound == 'DMSO':
            #     print(f"plate: {plate} has compound: {compound}")
            #     break
            output_compound = os.path.join(output_path, compound)
            if not os.path.exists(output_compound):
                os.makedirs(output_compound, exist_ok=True)
                compound_num += 1
            
            # for each concentration level
            for row_ind in range(16):
                img_prefix = "%03d" % (row_ind + 1) + "%03d" % (cp_ind + 1)
                print(f"compound:{compound}, image prefix: {img_prefix}")
                for image_name in image_list:
                    if image_name[:6] == img_prefix:
                        # copying to compound folder
                        src_path = os.path.join(plate_data, plate, image_name)
                        # print(src_path)
                        dsn_path = os.path.join(output_compound, image_name)
                        # print(dsn_path)
                        shutil.copyfile(src_path, dsn_path)

    print(f"In total created {compound_num} compounds!")
            
            
    

def get_plate_compound_map(csv_path):
    plate = {}
    for plate_map in os.listdir(csv_path):
        plate_map_name = plate_map[:-4]
        plate_df = pd.read_csv(os.path.join(csv_path, plate_map), skiprows=21, nrows=16, header=None)
        # get the compound list
        # TODO: get unique name for DMSO, POS (3 types)
        compound_list = plate_df.iloc[0][1:21].tolist()
        # compound_list = plate_df.iloc[0][20:21].tolist()
        compound_list_renamed = [cpname + '_' + plate_map_name for cpname in compound_list]
        negative = [neg + '_' + plate_map_name for neg in plate_df.iloc[0][21:23].tolist()]
        positive = ['POS' + '_' + plate_map_name for pos in plate_df.iloc[0][23:].tolist()]
        compound_list_renamed.extend(negative)
        compound_list_renamed.extend(positive)
        
        # store compound list to plate
        plate[plate_map_name] = compound_list_renamed
        print(plate)
    return plate
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", default=os.path.join(WORKDIR, 'reference_dataset_imaging_TIFFs_processed'), type=str)
    parser.add_argument("--platemapPath", default=os.path.join(WORKDIR, 'reference_dataset_plate_maps_results'), type=str)
    # parser.add_argument("--resultPath", default=os.path.join(WORKDIR, 'reference_dataset_plate_maps_results'), type=str)
    parser.add_argument("--outPath", default=os.path.join(WORKDIR,'reference_dataset_compounds_v1'), type=str)
    
    args = parser.parse_args()
    
    split_images_by_compound(args)