import os
import re
import glob
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
# os.system('pip install opencv-python')
import cv2
import random
from pathlib import Path
import matplotlib.image as mpimg
# from azureml.fsspec import AzureMachineLearningFileSystem
import glob

channel_dict = {
                1: [179, 65535],
                2: [345, 65535],
                3: [0, 38383],
                4: [144, 10705],
                5: [3621, 65535]}

# TD: change the norm value
class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        video, label = sample['video'], sample['label']
        new_video_x = (video - 127.5)/128
        return {'video': new_video_x, 'label': label}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(384, 384)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        video, pain_label = sample['video'], sample['label']

        frames, h, w, channel = video.shape[0], video.shape[1], video.shape[2], video.shape[3]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        new_video_x = np.zeros((frames, new_h, new_w, channel))
        for i in range(frames):
            image = video[i, :, :, :]
            #img = cv2.resize(image, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
            img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            img_expand = np.expand_dims(img, axis=2)
            # print(img_expand.shape)
            new_video_x[i, :, :, :] = img_expand

        return {'video': new_video_x, 'label': pain_label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(256, 256)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        video, pain_label = sample['video'], sample['label']
        # print(f"transform: {video.shape}") #(80, 384, 384, 1)
        
        frames, h, w, channel = video.shape[0], video.shape[1], video.shape[2], video.shape[3]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_video_x = np.zeros((frames, new_h, new_w, channel))
    
        for i in range(frames):
            image = video[i, :, :, :]
            image = image[top: top + new_h, left: left + new_w]
            new_video_x[i, :, :, :] = image

        return {'video': new_video_x, 'label': pain_label}


class CenterCrop(object):
    def __init__(self, output_size=(256, 256)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        video, label = sample['video'], sample['label']

        frames, h, w, channel = video.shape[0], video.shape[1], video.shape[2], video.shape[3]
        new_h, new_w = self.output_size

        top = int(round((h - new_h) /2.))
        left = int(round((w - new_w) / 2.))

        new_video_x = np.zeros((frames, new_h, new_w, channel))
        
        for i in range(frames):
            image = video[i, :, :, :]
            image = image[top: top + new_h, left: left + new_w]
            new_video_x[i, :, :, :] = image

        return {'video': new_video_x, 'label': label}
    
    

class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        video, label = sample['video'], sample['label']

        frames, h, w, channel = video.shape[0], video.shape[1], video.shape[2], video.shape[3]
        new_video_x = np.zeros((frames, h, w, channel))

        p = random.random()
        if p < 0.5:
            #print('Flip')
            for i in range(frames):
                # video 
                image = video[i, :, :, :]
                image = cv2.flip(image, 1)
                img_expand = np.expand_dims(image, axis=2)
                new_video_x[i, :, :, :] = img_expand  
                
            return {'video': new_video_x, 'label': label}
        else:
            #print('no Flip')
            return {'video': video, 'label': label}



class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        video, label = sample['video'], sample['label']

        # swap color axis because
        # numpy image: (batch_size) x depth x H x W x C
        # torch image: (batch_size) x C x depth X H X W
        video = video.transpose((3, 0, 1, 2))
        video = np.array(video)
        
                        
        label_np = np.array([0],dtype=np.int64)
        label_np[0] = label
        
        

        return {'video': torch.from_numpy(video.astype(np.float64)).float(), 'label': torch.from_numpy(label_np.astype(np.int64)).long()}



def get_label_for_compound(compound_name, labels_df):
    label = labels_df.loc[labels_df[0] == compound_name, 1].values[0] #.iloc[0]
    return label

class cytoDataset(Dataset):
    def __init__(self, cv_info, root_dir, transform=None):
        # set fore azure filesystem
        # self.fs =filesystem
        # f = filesystem.open(cv_info)
        self.compoundList = pd.read_csv(cv_info, delimiter=',', header=None)
        self.data_path = root_dir
        self.cp_samples_info, self.samples_labels = self._generate_separate_compound(self.data_path, self.compoundList)
        self.transform = transform
    
    def __len__(self):
        return len(self.samples_labels)
    
    def __getitem__(self, idx):
        one_compound_info = self.cp_samples_info[idx]
            
        sample_compound = self._get_video_data(one_compound_info)
        sample_label = self.samples_labels[idx]
        
        sample = {'video': sample_compound, 'label': sample_label}
        if self.transform:
            sample = self.transform(sample)
        return sample 
    
    def _generate_separate_compound(self, data_dir, cp_df):
        samples = []
        labels = []
        exclude_cp = ['ORM-0293612_230817GA24B0', 'ORM-0293667_230817GA24B0']
        # loop over the train/test compounds
        for i, compound_name in enumerate(cp_df[0].tolist()):
            if compound_name in exclude_cp:
                print(f"...skipping compound{compound_name}")
                continue
            compound_label = get_label_for_compound(compound_name, cp_df)
            compoundPath = os.path.join(data_dir, compound_name)
            paths = glob.glob(compoundPath + '/*.tif')
            # generate groups/videos for one compound, should have 4 videos
            compound_groups, compound_groups_labels = generate_compound_groups(paths, compoundPath, compound_label)
            
            #TD: add conditions
            # if compound_groups:
            samples.extend(compound_groups)
            labels.extend(compound_groups_labels)
        
        print(f"In total there are {len(samples)} compound samples/videos! ")
        return samples, labels

    def _get_video_data(self, sample_cp_list):
        
        # TD: first combine 5 channels together 
        # normalize
        normalized_arrays = []
        # img = mpimg.imread(sample_cp_list[0])
        
        # nx1080x1080x1
        video = np.zeros((len(sample_cp_list), 1080, 1080, 1))
        
        for i, image in enumerate(sample_cp_list):
            ch = Path(image).stem[-1]
            if int(ch) in channel_dict.keys():
                minmax_list = channel_dict[int(ch)]
                # fsimage = self.fs.to_absolute_path(image)
                # print(fsimage)
                # print(image)
                im_arr = np.asarray(Image.open(image)) #mpimg.imread(fsimage) #np.array(Image.open(os.path.join(sample_cp_list, image)))
                # print(im_arr.shape)
                norm_arr = normalize_channel(im_arr, minmax_list[0], minmax_list[1])
                norm_arr_unit8 = np.uint8(norm_arr * 255)
                # normalized_arrays.append(norm_arr_unit8)
                norm_arr_unit8_expand = np.expand_dims(norm_arr_unit8, axis=2)
                # print(norm_arr_unit8_expand.shape)
                video[i, :, :, :] = norm_arr_unit8_expand
            else:
                continue
        
        # then combine all frames to a video
        # print(f"one video is of {video.shape}")
        return video 

def normalize_channel(image_array, min_value, max_value):
    normalized_array = (image_array - min_value) / (max_value - min_value)
    return normalized_array
   

#TD: split one compount date into 4 videos/sub
def generate_compound_groups(image_path_list, compoundPath, cp_label):
    print(f"preparing {len(image_path_list)} images in compound {compoundPath}")
    # image_list = os.listdir(compoundPath)
    min_column, max_column = find_comp_column_range(image_path_list)
    columns = [min_column, max_column]
    views = [1, 2]
    if min_column is not None and max_column is not None:
        # print(f"The range of columns in the files is from {min_column} to {max_column}.")
        groups = group_images(compoundPath, columns, views)
        groups_labels = [cp_label] * len(groups)
        
        return groups, groups_labels
    else:
        print("No valid column data found in filenames.")
        return None, None

 

def find_comp_column_range(directory_paths):
    pattern = re.compile(r'001(\d{3})-\d-00100100\d\.tif')

    # 存储所有解析出的column值
    columns = []

    # 遍历目录中的所有文件
    for filepath in directory_paths:
        filename = filepath.rsplit('/', 1)[1]
        match = pattern.match(filename)
        if match:
            # 将匹配到的column部分添加到列表中
            column = int(match.group(1))
            # print(column)
            columns.append(column)

    # 如果columns列表不为空，计算最小和最大值
    if columns:
        min_column = min(columns)
        max_column = max(columns)
        return min_column, max_column
    else:
        return None, None
        
def group_images(directory, columns, views):
    # 初始化一个字典来存储分组
    groups = []

    # 构建正则表达式模式来匹配文件名
    pattern = re.compile(r'(\d{3})(\d{3})-(\d)-001001(\d{3}).tif')
    
    # 遍历指定的column和view组合
    for column in columns:
        for view in views:
            # 使用glob来查找匹配特定column和view的所有文件
            path_pattern = f"{directory}/*{column}-{view}-*.tif"
            files = glob.glob(path_pattern)
            
            # 添加到字典中，键为(column, view)
            if files:  # 确保列表不为空
                groups.append(files)

    return groups

if __name__ == '__main__':
    
    curdir = os.path.join(os.getcwd(), 'workspace/cytotoxicity')
    print(curdir) #/mnt/batch/tasks/shared/LS_root/mounts/clusters/cytox/code #/home/azureuser/cloudfiles/code/workspace/cytotoxicity/data/train_test_folds/test_fold_1_info.txt
    root_list = os.path.join(curdir, 'data/cytoData_v2') #'/Volumes/KINGSTON/Orion/reference_dataset_compounds_v1'
    trainval_list = os.path.join(curdir, 'data/train_test_folds/test_fold_1_info.txt')
    # define the azure storage URI
    # 'azureml://subscriptions/25130c3f-778b-4637-bfb8-3b1b885b45e7/resourcegroups/rg-silo-dev-003/workspaces/aml-silo-dev-003/datastores/workspaceblobcompound/paths/reference_dataset_compounds_v2/'
    # 'reference_dataset_compounds_v2/'
    # uri = 'azureml://subscriptions/25130c3f-778b-4637-bfb8-3b1b885b45e7/resourcegroups/rg-silo-dev-003/workspaces/aml-silo-dev-003/datastores/workspaceblobcompound/paths'
    # create the filesystem
    # fs = AzureMachineLearningFileSystem(uri)
    
    cyto_train = cytoDataset(cv_info=trainval_list, root_dir=root_list, transform=transforms.Compose([Normaliztion(), Rescale((384,384)), RandomCrop((256,256)), RandomHorizontalFlip(), ToTensor()]))
    dataloader = DataLoader(cyto_train, batch_size=4, shuffle=True, num_workers=4)
    
    # print first batch for evaluation
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['video'].shape, sample_batched['label'].shape)
        # print(i_batch, sample_batched['video'], sample_batched['label'])
 
        break
