from imghdr import tests
from typing import Callable, Optional, Union

import torch
import numpy as np
import pickle
import os
from sklearn import metrics
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from collections import Counter
from sklearn.model_selection import train_test_split

from torch.nn import BCEWithLogitsLoss


IGNORE_FLAG = -9999

field_type = {'10595_2': 'GC',
              '10595_7': 'GC',
              '9442_1': 'GC',
              '9442_3': 'GC',
              '9442_5': 'GC',
              '10760_2': 'GAL',
              '10760_4': 'GAL',
              '10631_3': 'EX',
              '10631_1': 'EX',
              '10631_4': 'EX',
              '12103_a3': 'EX',
              '13498_b1': 'EX',
              '13737_2': 'GAL',
              '13737_3': 'GAL',
              '9490_a3': 'GAL',
              '10349_30': 'GC',
              '10005_10': 'GC',
              '10120_3': 'GC',
              '12513_2': 'GAL',
              '12513_3': 'GAL',
              '14164_9': 'EX',
              '13718_6': 'EX',
              '10524_7': 'GC',
              '10182_pb': 'GAL',
              '10182_pd': 'GAL',
              '9425_2': 'EX',
              '9425_4': 'EX',
              '9583_99': 'EX',
              '10584_13': 'GAL',
              '9978_5e': 'EX',
              '15100_2': 'EX',
              '15647_13': 'EX',
              '11340_11': 'GC',
              '13389_10': 'EX',
              '9694_6': 'EX',
              '10342_3': 'GAL',

              '14343_1': 'GAL',
              '10536_13': 'EX',
              '13057_1': 'GAL',
              '10260_7': 'GAL',
              '10260_5': 'GAL',
              '10407_3': 'GAL',
              '13375_4': 'EX',
              '13375_7': 'EX',
              '13364_95': 'GAL',
              '10190_28': 'GAL',
              '10190_13': 'GAL',
              '10146_4': 'GC',
              '10146_3': 'GC',
              '10775_ab': 'GC',
              '11586_5': 'GC',
              '12438_1': 'EX',
              '13671_35': 'EX',
              '14164_1': 'GC',

              '9490_a2': 'GAL',
              '9405_6d': 'EX',
              '9405_4b': 'EX',
              '9450_14': 'EX',
              '10092_1': 'EX',
              '13691_11': 'GAL',
              '12058_12': 'GAL',
              '12058_16': 'GAL',
              '12058_1': 'GAL',
              '9450_16': 'EX',
              '10775_52': 'GC',
              '12602_1': 'GC',
              '12602_2': 'GC',
              '10775_29': 'GC',
              '10775_ad': 'GC',
              '12058_6': 'GAL',  # NEW
              '14704_1': 'GAL',  # NEW
              '13804_6': 'GAL'  # NEW
              }

def get_dirs(base_dir, data_base):
    train_dirs = []
    test_dirs = []

    test_base = os.path.join(data_base,'npy_test')
    train_base = os.path.join(data_base,'npy_train')

    print('------------------------------------------------------------')
    print('Fetching directories for the test set')
    print('------------------------------------------------------------')
    for _filter in os.listdir(test_base):
        filter_dir = os.path.join(test_base,_filter)
        if os.path.isdir(filter_dir) and _filter == 'f435w':
            for prop_id in os.listdir(filter_dir):
                prop_id_dir = os.path.join(filter_dir,prop_id)
                if os.path.isdir(prop_id_dir):
                    for vis_num in os.listdir(prop_id_dir):
                        vis_num_dir = os.path.join(prop_id_dir,vis_num)
                        if os.path.isdir(vis_num_dir):
                            for f in os.listdir(vis_num_dir):
                                if '.npy' in f and f != 'sky.npy':
                                    key = f'{prop_id}_{vis_num}'
                                    if field_type[key] == 'GAL':
                                        test_dirs.append(os.path.join(vis_num_dir,f))

    print('------------------------------------------------------------')
    print('Fetching directories for the training set')
    print('------------------------------------------------------------')
    for _filter in os.listdir(train_base):
        filter_dir = os.path.join(train_base,_filter)
        if os.path.isdir(filter_dir) and _filter == 'f435w':
            for prop_id in os.listdir(filter_dir):
                prop_id_dir = os.path.join(filter_dir,prop_id)
                if os.path.isdir(prop_id_dir):
                    for vis_num in os.listdir(prop_id_dir):
                        vis_num_dir = os.path.join(prop_id_dir,vis_num)
                        if os.path.isdir(vis_num_dir):
                            for f in os.listdir(vis_num_dir):
                                if '.npy' in f and f != 'sky.npy':
                                    key = f'{prop_id}_{vis_num}'
                                    if field_type[key] == 'GAL':
                                        train_dirs.append(os.path.join(vis_num_dir,f))


#     print(train_dirs)
    np.save(os.path.join(base_dir,'test_dirs.npy'), test_dirs)
    np.save(os.path.join(base_dir,'train_dirs.npy'), train_dirs)

    return None

class PairedDatasetImagePath(Dataset):
    def __init__(self, paths, skyaug_min=0, skyaug_max=0, part=None, f_val=0.1, seed=1):
        """ custom pytorch dataset class to load deepCR-mask training data
        :param paths: (list) list of file paths to (3, W, H) images: image, cr, ignore.
        :param skyaug_min: [float, float]. If sky is provided, use random sky background in the range
          [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers to allow the trained model to adapt to a
          wider range of sky background or equivalently exposure time. Remedy the fact that exposure time in the
          training set is discrete and limited.
        :param skyaug_min: float. subtract maximum amount of abs(skyaug_min) * sky_level as data augmentation
        :param skyaug_max: float. add maximum amount of skyaug_max * sky_level as data augmentation
        :param part: either 'train' or 'val'.
        :param f_val: percentage of dataset reserved as validation set.
        :param seed: fix numpy random seed to seed, for reproducibility.
        """
        assert 0 <= f_val < 1
        np.random.seed(seed)
        n_total = len(paths)
        n_train = int(n_total * (1 - f_val)) #int(len * (1 - f_val)) JK
        f_test = f_val
        n_search = int(n_total * (1 - f_val - f_test))

        if part == 'train':
            s = np.s_[:max(1, n_train)]
        elif part == 'test':
            s = np.s_[min(n_total - 1, n_train):]
        else:
            s = np.s_[0:]

        self.paths = paths[s]
        self.skyaug_min = skyaug_min
        self.skyaug_max = skyaug_max

    def __len__(self):
        return len(self.paths)

    def get_skyaug(self, i):
        """
        Return the amount of background flux to be added to image
        The original sky background should be saved in sky.npy in each sub-directory
        Otherwise always return 0
        :param i: index of file
        :return: amount of flux to add to image
        """
        path = os.path.split(self.paths[i])[0]
        sky_path = os.path.join(path, 'sky.npy') #JK
        if os.path.isfile(sky_path):
            f_img = self.paths[i].split('/')[-1]
            sky_idx = int(f_img.split('_')[0])
            sky = np.load(sky_path)[sky_idx-1]
            return sky * np.random.uniform(self.skyaug_min, self.skyaug_max)
        else:
            return 0

    def __getitem__(self, i):
        data = np.load(self.paths[i])
        image = data[0]
        mask = data[1]
        if data.shape[0] == 3:
            ignore = data[2]
        else:
            ignore = np.zeros_like(data[0])
        # try:#JK
        skyaug = self.get_skyaug(i)

        image, mask, ignore = get_fixed_crop([image, mask, ignore], 128, 128)
        mask[ignore == 1] = mask[ignore == 1] + IGNORE_FLAG

        image, mask = set_cosmic_input(image + skyaug, mask, shape=128)
        return image.to(dtype=torch.float32), mask

def get_random_crop(images, crop_height, crop_width):

    max_x = images[0].shape[1] - crop_width
    max_y = images[0].shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crops = []
    for image in images:
        crop = image[y: y + crop_height, x: x + crop_width]
        crops.append(crop)

    return crops


def get_fixed_crop(images, crop_height, crop_width):

    x = 64
    y = 64

    crops = []
    for image in images:
        crop = image[y: y + crop_height, x: x + crop_width]
        crops.append(crop)

    return crops
    


def build_nasbench360_cosmic_dataset(cfg):
    path = cfg["cosmic_root"]
    train_dirs = np.load(os.path.join(path, "train_dirs.npy"), allow_pickle=True)
    test_dirs = np.load(os.path.join(path, "test_dirs.npy"), allow_pickle=True)

    # Hacky
    if path == "datasets/cosmic":
        # Exclude ../
        train_dirs = [td[3:] for td in train_dirs]
        test_dirs = [td[3:] for td in test_dirs]

    aug_sky = (-0.9, 3)

    # only train f435 and GAL flag for now
    trainvalset_full = PairedDatasetImagePath(train_dirs[::], aug_sky[0], aug_sky[1], part="train", f_val=0.0)

    testset = PairedDatasetImagePath(test_dirs[::], aug_sky[0], aug_sky[1], part="None")

    train_val_size = len(trainvalset_full)
    val_size = 483

    trainset = Subset(trainvalset_full, np.arange(train_val_size)[:-val_size])
    valset = Subset(trainvalset_full, np.arange(train_val_size)[-val_size:])

    return trainset, valset, testset


def get_dirs(base_dir, data_base):
    train_dirs = []
    test_dirs = []

    test_base = os.path.join(data_base,'npy_test')
    train_base = os.path.join(data_base,'npy_train')

    print('------------------------------------------------------------')
    print('Fetching directories for the test set')
    print('------------------------------------------------------------')
    for _filter in os.listdir(test_base):
        filter_dir = os.path.join(test_base,_filter)
        if os.path.isdir(filter_dir) and _filter == 'f435w':
            for prop_id in os.listdir(filter_dir):
                prop_id_dir = os.path.join(filter_dir,prop_id)
                if os.path.isdir(prop_id_dir):
                    for vis_num in os.listdir(prop_id_dir):
                        vis_num_dir = os.path.join(prop_id_dir,vis_num)
                        if os.path.isdir(vis_num_dir):
                            for f in os.listdir(vis_num_dir):
                                if '.npy' in f and f != 'sky.npy':
                                    key = f'{prop_id}_{vis_num}'
                                    if field_type[key] == 'GAL':
                                        test_dirs.append(os.path.join(vis_num_dir,f))

    print('------------------------------------------------------------')
    print('Fetching directories for the training set')
    print('------------------------------------------------------------')
    for _filter in os.listdir(train_base):
        filter_dir = os.path.join(train_base,_filter)
        if os.path.isdir(filter_dir) and _filter == 'f435w':
            for prop_id in os.listdir(filter_dir):
                prop_id_dir = os.path.join(filter_dir,prop_id)
                if os.path.isdir(prop_id_dir):
                    for vis_num in os.listdir(prop_id_dir):
                        vis_num_dir = os.path.join(prop_id_dir,vis_num)
                        if os.path.isdir(vis_num_dir):
                            for f in os.listdir(vis_num_dir):
                                if '.npy' in f and f != 'sky.npy':
                                    key = f'{prop_id}_{vis_num}'
                                    if field_type[key] == 'GAL':
                                        train_dirs.append(os.path.join(vis_num_dir,f))

    np.save(os.path.join(base_dir,'test_dirs.npy'), test_dirs)
    np.save(os.path.join(base_dir,'train_dirs.npy'), train_dirs)

    return None


class CosmicMetricFunction:
    def __init__(self, use_ignore):
        self.use_ignore = use_ignore
    
    def __call__(self, pred, gt):
        pred = np.concatenate([np.array(x).reshape(-1) for x in pred])
        gt = np.concatenate([np.array(x).reshape(-1) for x in gt])

        if self.use_ignore:
            pred = pred[gt >= 0]
            gt = gt[gt >= 0]
        else:
            gt[gt < 0] = gt[gt < 0] + IGNORE_FLAG
        
        pred = pred.astype(np.int32)
        gt = gt.astype(np.int32)
        auroc = metrics.roc_auc_score(gt, pred)
        return auroc


def set_cosmic_input(img, mask, shape=128):
    img = torch.tensor(img).view((-1, shape, shape))
    mask = torch.tensor(mask).view((-1, shape, shape))
    return img, mask



class CosmicBCEWithLogitsLoss(BCEWithLogitsLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, input, target):
        input_valid = input[target >= 0]
        target_valid = target[target >= 0]
        if input_valid.numel() == 0:
            return input.sum() * 0.0 
        else:
            return super().forward(input_valid, target_valid)


if __name__ == "__main__":
    path = "/localdisk/home/lericsso/code/einspace/data/cosmic/new"
    trainset, valset, testset = build_nasbench360_cosmic_dataset({"cosmic_root": path})
    print("Traning set size:", len(trainset))
    print("Validation set size:", len(valset))
    print("Test set size:", len(testset))

    print(trainset[0][0].shape, trainset[0][1].shape)
    print(valset[0][0].shape, valset[0][1].shape)
    print(testset[0][0].shape, testset[0][1].shape)
