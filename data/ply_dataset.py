from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
from numpy.random import RandomState
import h5py
import random
from tqdm import tqdm
import pickle
import h5py
import glob
from utils.io import read_ply_xyz, read_ply_from_file_list
from utils.pc_transform import swap_axis
from data.real_dataset import RealWorldPointsDataset
from plyfile import PlyData


def get_stems_from_pickle(test_split_pickle_path):
    """
    get the stem list from a split, given a pickle file
    """
    with open(test_split_pickle_path, 'rb') as f:
        test_list = pickle.load(f)
    stem_ls = []
    for itm in test_list:
        stem, ext = os.path.splitext(itm)
        stem_ls.append(stem)
    return stem_ls

class PlyDataset(data.Dataset):
    """
    datasets that with Ply format
    without GT: MatterPort, ScanNet, KITTI
        Datasets provided by pcl2pcl
    with GT: PartNet, each subdir under args.dataset_path contains 
        the partial shape raw.ply and complete shape ply-2048.txt.
        Dataset provided by MPC

    """
    def __init__(self, args):
        self.dataset = args.dataset
        self.dataset_path = args.dataset_path

        if self.dataset in ['MatterPort', 'ScanNet', 'KITTI']:
            input_pathnames = sorted(glob.glob(self.dataset_path+'/*'))
            input_ls = read_ply_from_file_list(input_pathnames)
            # swap axis as pcl2pcl and ShapeInversion have different canonical pose
            input_ls_swapped = [swap_axis(itm, swap_mode='n210') for itm in input_ls]
            self.input_ls = input_ls_swapped
            self.stems = range(len(self.input_ls))
        elif self.dataset in ['PartNet']:
            pathnames = sorted(glob.glob(self.dataset_path+'/*'))
            basenames = [os.path.basename(itm) for itm in pathnames]

            self.stems = [int(itm) for itm in basenames]

            input_ls = [read_ply_xyz(os.path.join(itm,'raw.ply')) for itm in pathnames]
            gt_ls = [np.loadtxt(os.path.join(itm,'ply-2048.txt'),delimiter=';').astype(np.float32) for itm in pathnames]
 
            # swap axis as multimodal and ShapeInversion have different canonical pose
            self.input_ls = [swap_axis(itm, swap_mode='210') for itm in input_ls]
            self.gt_ls = [swap_axis(itm, swap_mode='210') for itm in gt_ls]
        else:
            raise NotImplementedError
    
    def __getitem__(self, index):
        if self.dataset in ['MatterPort','ScanNet','KITTI']:
            stem = self.stems[index]
            input_pcd = self.input_ls[index]
            return (input_pcd, stem)
        elif self.dataset  in ['PartNet']:
            stem = self.stems[index]
            input_pcd = self.input_ls[index]
            gt_pcd = self.gt_ls[index]
            return (gt_pcd, input_pcd, stem)
    
    def __len__(self):
        return len(self.input_ls)  

class RealDataset(data.Dataset):
    """
    datasets that with Ply format
    without GT: MatterPort, ScanNet, KITTI
        Datasets provided by pcl2pcl
    with GT: PartNet, each subdir under args.dataset_path contains 
        the partial shape raw.ply and complete shape ply-2048.txt.
        Dataset provided by MPC

    """
    def __init__(self, args):
        #self.dataset = args.dataset
        #self.dataset_path = args.dataset_path
        self.dataset = args.realdataset#'ScanNet'
        self.random_seed = 0
        self.rand_gen = RandomState(self.random_seed)

        if self.dataset in ['MatterPort', 'ScanNet', 'KITTI']:
            if self.dataset == 'ScanNet':
                REALDATASET = RealWorldPointsDataset('./datasets/data/scannet_v2_'+args.class_choice+'s_aligned/point_cloud', batch_size=6, npoint=2048,  shuffle=False, split=args.split, random_seed=0)
            elif self.dataset == 'MatterPort':
                if args.split in ['train', 'trainval']:
                    REALDATASET = RealWorldPointsDataset('./datasets/data/scannet_v2_'+args.class_choice+'s_aligned/point_cloud', batch_size=6, npoint=2048,  shuffle=False, split=args.split, random_seed=0)
                else:
                    REALDATASET = RealWorldPointsDataset('./datasets/data/MatterPort_v1_'+args.class_choice+'_Yup_aligned/point_cloud', batch_size=6, npoint=2048,  shuffle=False, split=args.split, random_seed=0)
            elif self.dataset == 'KITTI':
                if args.split in ['train']:
                    REALDATASET = KITTIDataset('./datasets/data/KITTI_frustum_data_for_pcl2pcl/point_cloud_train/')
                elif args.split in ['test', 'val']:
                    REALDATASET = KITTIDataset('./datasets/data/KITTI_frustum_data_for_pcl2pcl/point_cloud_val/')
            input_ls = REALDATASET.point_clouds 
            # swap axis as pcl2pcl and ShapeInversion have different canonical pose
            input_ls_swapped = [np.float32(swap_axis(itm, swap_mode='n210')) for itm in input_ls]
            self.input_ls = input_ls_swapped
            self.stems = range(len(self.input_ls))
        elif self.dataset in ['PartNet']:
            pathnames = sorted(glob.glob(self.dataset_path+'/*'))
            basenames = [os.path.basename(itm) for itm in pathnames]

            self.stems = [int(itm) for itm in basenames]

            input_ls = [read_ply_xyz(os.path.join(itm,'raw.ply')) for itm in pathnames]
            gt_ls = [np.loadtxt(os.path.join(itm,'ply-2048.txt'),delimiter=';').astype(np.float32) for itm in pathnames]
 
            # swap axis as multimodal and ShapeInversion have different canonical pose
            self.input_ls = [swap_axis(itm, swap_mode='210') for itm in input_ls]
            self.gt_ls = [swap_axis(itm, swap_mode='210') for itm in gt_ls]
        else:
            raise NotImplementedError
    
    def __getitem__(self, index):
        if self.dataset in ['MatterPort','ScanNet','KITTI']:
            stem = self.stems[index]
            choice = self.rand_gen.choice(self.input_ls[index].shape[0], 2048, replace=True)
            input_pcd = self.input_ls[index][choice,:]
            return (input_pcd, stem)
        elif self.dataset  in ['PartNet']:
            stem = self.stems[index]
            input_pcd = self.input_ls[index]
            gt_pcd = self.gt_ls[index]
            return (gt_pcd, input_pcd, stem)
    
    def __len__(self):
        return len(self.input_ls)  

class KITTIDataset():
    def __init__(self, load_path):
        self.point_clouds = []
        file_list = glob.glob(load_path + '*.ply')
        total_num = len(file_list)
        for i in range(total_num):
            file_name = load_path + str(i) + '.ply'
            ply_file = PlyData.read(file_name)
            pc = np.array([ply_file['vertex']['x'], ply_file['vertex']['y'], ply_file['vertex']['z']])
            pc = np.transpose(pc,(1,0))
            self.point_clouds.append(pc)
        return

class GeneratedDataset(data.Dataset):
    """
    datasets that with Ply format
    without GT: MatterPort, ScanNet, KITTI
        Datasets provided by pcl2pcl
    with GT: PartNet, each subdir under args.dataset_path contains 
        the partial shape raw.ply and complete shape ply-2048.txt.
        Dataset provided by MPC

    """
    def __init__(self, args):
        #self.dataset = args.dataset
        #self.dataset_path = args.dataset_path
        self.dataset = args.dataset#'ModelNet'
        self.category = args.save_inversion_path.split('/')[-3].split('_')[-1]
        if self.dataset == 'ModelNet':
            self.dataset_path = './datasets/ModelNet40_Completion/' + self.category + '/' + args.split
        elif self.dataset == '3D_FUTURE':
            self.dataset_path = './datasets/3D_FUTURE_Completion/' + self.category + '/' + args.split
        self.num_view = 5
        self.random_seed = 0
        self.rand_gen = RandomState(self.random_seed)

        if self.dataset in ['ModelNet', '3D_FUTURE']:
            complete_pathnames = sorted(glob.glob(self.dataset_path+'/*complete.npy'))
            partial_pathnames = sorted(glob.glob(self.dataset_path+'/*partial.npy'))
            self.input_ls = [np.load(itm).astype(np.float32) for itm in partial_pathnames]
            self.gt_ls = [np.load(itm).astype(np.float32) for itm in complete_pathnames]
            self.stems = range(len(self.input_ls))
        else:
            raise NotImplementedError
    
    def __getitem__(self, index):
        if self.dataset in ['MatterPort','ScanNet','KITTI']:
            stem = self.stems[index]
            choice = self.rand_gen.choice(self.input_ls[index].shape[0], 2048, replace=True)
            input_pcd = self.input_ls[index][choice,:]
            return (input_pcd, stem)
        elif self.dataset  in ['PartNet', 'ModelNet', '3D_FUTURE']:
            stem = self.stems[index]
            input_pcd = self.input_ls[index]
            gt_pcd = self.gt_ls[index]
            return (gt_pcd, input_pcd, stem)
    
    def __len__(self):
        return len(self.input_ls)  
if __name__ == '__main__':
    REALDATASET = KITTIDataset('./datasets/data/KITTI_frustum_data_for_pcl2pcl/point_cloud_train/')
    print(REALDATASET.point_clouds)
