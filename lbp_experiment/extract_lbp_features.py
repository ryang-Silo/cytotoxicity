import os
import sys
import numpy as np
import cv2
import json
from pathlib import Path
import argparse
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA


WORKDIR = os.getcwd()
print(f"working dir: {WORKDIR}")

def extract_lbp_features(image_path, radius=1, n_points=8, method='uniform'):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    lbp = local_binary_pattern(image, n_points, radius, method)
    # calculate histogram of lbp
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    # normalize
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  
    return lbp_hist

def merge_features_by_filename_prefix(args):
    features_dict = {}
    #TODO: extract full 
    for file_name in os.listdir(args.dataPath):
        if file_name.endswith('.tif') and file_name[-5] in '12345': 
            channel_index = file_name[-5] 
            prefix = file_name[:-5] 
            feature = extract_lbp_features(os.path.join(args.dataPath, file_name))
            if prefix not in features_dict:
                features_dict[prefix] = {}
            features_dict[prefix][channel_index] = feature

    # merge feature of channel 1-5
    if features_dict:
        example_feature = next(iter(next(iter(features_dict.values())).values()))
        for key in features_dict:
            features_dict[key] = np.concatenate([features_dict[key].get(str(i), np.zeros_like(example_feature)) for i in range(1, 6)])

    return features_dict

def default_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def merge_compound_features(features_dict, args):
    compound_features = {}

    for key, features in features_dict.items():
        compound_key = key[4:6]  # get compound index name
        print(f"compound_key: {compound_key}")
        concentration_key = key[1:3]  # get concentration level
        print(f"concentration_key: {concentration_key}")
                
        if compound_key not in compound_features:
            compound_features[compound_key] = {}
        

        if concentration_key in compound_features[compound_key]:
            # 如果已经有了这个浓度级别的特征，直接合并（这种情况理论上不会发生，除非有重复键）
            compound_features[compound_key][concentration_key] = np.concatenate([compound_features[compound_key][concentration_key], features])
        else:
            compound_features[compound_key][concentration_key] = features
    
    # merge and save
    merged_compound_features = {}
    for compound_key, concentrations in compound_features.items():
        print(merge_compound_features)
        print(concentrations)
        merged_features = np.concatenate(list(concentrations.values()))
        merged_compound_features[compound_key] = merged_features
        #save as npy, each compound one npy file
        save_path = os.path.join(args.outPath, "compound_features")
        os.makedirs(save_path, exist_ok=True)
        save_name = save_path + f'/compound_{compound_key}_features.npy'
        np.save(save_name, merged_features)
    
    return merged_compound_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", default=os.path.join(WORKDIR, 'eda_phase/lbp_experiments/data/test'), type=str)
    parser.add_argument("--outPath", default=WORKDIR+'/eda_phase/lbp_experiments/test_features', type=str)
    
    args = parser.parse_args()
    
    features = merge_features_by_filename_prefix(args)
    # print(features)
    
    save_feature_name = 'train_features_concatenated.json'
    with open(os.path.join(args.outPath, save_feature_name), 'w') as f:
        json.dump(features, f, default=default_serializer, indent=4)
    
    merged_compound_features = merge_compound_features(features, args)
    # print(merged_compound_features)
    
    save_json_name = os.path.join(args.outPath, "merged_compound_features.json")
    with open(save_json_name, "w") as f:
        json.dump(merged_compound_features, f, default=default_serializer, indent=4)
    