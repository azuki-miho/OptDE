import os
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from data.CRN_dataset import CRNShapeNet
from data.ply_dataset import PlyDataset, RealDataset, GeneratedDataset


from arguments import Arguments

from loss import *

from optde import OptDE

from model.network import Generator, Discriminator
from external.ChamferDistancePytorch.chamfer_python import distChamfer, distChamfer_raw

import random

import numpy as np

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()

time_stamp = time.strftime('Log_%Y-%m-%d_%H-%M-%S/', time.gmtime())

class Trainer(object):

    def __init__(self, args):
        self.args = args
        
        save_inversion_dirname = args.save_inversion_path.split('/')
        log_pathname = './'+args.log_dir+'/'+ save_inversion_dirname[-3] + '/' + save_inversion_dirname[-2] + '/log.txt'
        args.log_pathname = log_pathname

        self.model = OptDE(self.args)
        
        ###Load Virtual Train Data
        self.virtual_data_name = self.args.virtualdataset
        self.args.dataset = self.virtual_data_name
        if self.virtual_data_name in ['ScanNet', 'MatterPort']:
            self.args.split = 'trainval'
        elif self.virtual_data_name in ['ModelNet', '3D_FUTURE', 'KITTI', 'CRN']:
            self.args.split = 'train'
        if self.virtual_data_name in ['MatterPort','ScanNet','KITTI','PartNet']:
            train_dataset = PlyDataset(self.args)
        elif self.virtual_data_name in ['ModelNet', '3D_FUTURE']:
            train_dataset = GeneratedDataset(self.args)
        else: 
            train_dataset = CRNShapeNet(self.args)
        
        p2c_batch_size = 10#20
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=p2c_batch_size,
            shuffle=False,
            pin_memory=True)
        ###Load Virtual Test Data
        self.args.split = 'test'

        if self.virtual_data_name in ['MatterPort','ScanNet','KITTI','PartNet']:
            test_dataset = PlyDataset(self.args)
        elif self.virtual_data_name in ['ModelNet', '3D_FUTURE']:
            test_dataset = GeneratedDataset(self.args)
        else: 
            test_dataset = CRNShapeNet(self.args)
        
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True)
        ###Load Real Train Data
        self.real_data_name = self.args.realdataset
        self.args.dataset = self.real_data_name
        if self.real_data_name in ['ScanNet', 'MatterPort']:
            self.args.split = 'trainval'
        elif self.real_data_name in ['ModelNet', '3D_FUTURE', 'KITTI', 'CRN']:
            self.args.split = 'train'
        if self.real_data_name in ['MatterPort','ScanNet','KITTI','PartNet']:
            real_train_dataset = RealDataset(self.args)
        elif self.real_data_name in ['ModelNet', '3D_FUTURE']:
            real_train_dataset = GeneratedDataset(self.args)
        elif self.real_data_name in ['CRN']: 
            real_train_dataset = CRNShapeNet(self.args)
        self.real_train_dataloader = DataLoader(
                real_train_dataset,
                batch_size=p2c_batch_size,
                shuffle=False,
                pin_memory=True)
        ###Load Real Test Data
        self.args.split = 'test'
        if self.real_data_name in ['MatterPort','ScanNet','KITTI','PartNet']:
            real_test_dataset = RealDataset(self.args)
        elif self.real_data_name in ['ModelNet', '3D_FUTURE']:
            real_test_dataset = GeneratedDataset(self.args)
        elif self.real_data_name in ['CRN']:
            real_test_dataset = CRNShapeNet(self.args)
        self.real_test_dataloader = DataLoader(
                real_test_dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=True)
        
    def train(self):
        load_path_name = self.args.finetune_ckpt_load
        print(load_path_name)
        test_real_ucd_loss_list = []
        test_real_uhd_loss_list = []
        test_real_cd_loss_list = []
        for i, data in enumerate(self.real_test_dataloader):
            tic = time.time()
            # with gt
            if self.real_data_name in ['ScanNet', 'MatterPort', 'KITTI']:
                partial, index = data
            elif self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                gt, partial, index = data
                gt = gt.squeeze(0).float().cuda()
            partial = partial.squeeze(0).float().cuda()

            # reset G for each new input
            #self.model.reset_G(pcd_id=index.item())
            self.model.reset_G_tmp()
            self.model.pcd_id = index[0].item()

            # set target and complete shape 
            # for ['reconstruction', 'jittering', 'morphing'], GT is used for reconstruction
            # else, GT is not involved for training
            if self.real_data_name in ['ScanNet', 'MatterPort', 'KITTI']:
                self.model.set_target(partial=partial)
            elif self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                self.model.set_target(gt=gt, partial=partial)
            
            # inversion
            self.model.reset_whole_network(load_path_name)
            if self.real_data_name in ['ScanNet', 'MatterPort', 'KITTI']:
                test_real_ucd_loss, test_real_uhd_loss = self.model.finetune()
            elif self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                test_real_ucd_loss, test_real_uhd_loss, test_real_cd_loss = self.model.finetune(bool_gt=True)
                test_real_cd_loss_list.append(test_real_cd_loss)
            test_real_ucd_loss_list.append(test_real_ucd_loss)
            test_real_uhd_loss_list.append(test_real_uhd_loss)
            toc = time.time()
            print(i, 'out of', len(self.real_test_dataloader),'done in ',int(toc-tic),'s')
        if self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
            np.save(self.args.save_inversion_path+'/cd_list.npy', np.array(test_real_cd_loss_list))
            test_real_cd_loss_mean = np.mean(np.array(test_real_cd_loss_list))
        test_real_ucd_loss_mean = np.mean(np.array(test_real_ucd_loss_list))
        test_real_uhd_loss_mean = np.mean(np.array(test_real_uhd_loss_list))
        if self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
            print("Mean CD on Real Test Set:", test_real_cd_loss_mean)
        print("Mean UCD on Real Test Set:", test_real_ucd_loss_mean)
        print("Mean UHD on Real Test Set:", test_real_uhd_loss_mean)
        with open(self.args.log_pathname, "a") as file_object:
            if self.real_data_name in ['ModelNet', '3D_FUTURE', 'CRN']:
                msg =  "Mean CD on Real Test Set:" + "%.8f"%test_real_cd_loss_mean
                file_object.write(msg+'\n')
            msg =  "Mean UCD on Real Test Set:" + "%.8f"%test_real_ucd_loss_mean
            file_object.write(msg+'\n')
            msg =  "Mean UHD on Real Test Set:" + "%.8f"%test_real_uhd_loss_mean
            file_object.write(msg+'\n')
        os.system("mv " + './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-3] + '/' + time_stamp[:-1] + '/saved_results/*' + ' ./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-3] + '/' + time_stamp[:-1] + '/best_results/')


if __name__ == "__main__":
    args = Arguments(stage='inversion').parser().parse_args()
    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)
    
    if not os.path.isdir('./'+args.log_dir+'/'):
        os.mkdir('./'+args.log_dir+'/')
    if not os.path.isdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1]):
        os.mkdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1])
    if not os.path.isdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1]):
        os.mkdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1])
    if not os.path.isdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/best_results'):
        os.mkdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1]+'/best_results')
    if not os.path.isdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code'):
        os.mkdir('./'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code')
        os.system('cp %s %s'% ('run_optimizer.sh', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] +
 '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('trainer.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('trainer_optimizer.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-
1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('optde.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('model/network.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('data/ply_dataset.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('data/real_dataset.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
        os.system('cp %s %s'% ('trainer_optimizer.py', './'+args.log_dir+'/' + args.save_inversion_path.split('/')[-1] + '/' + time_stamp[:-1] + '/code/'))
    

    args.save_inversion_path += '/' + time_stamp[:-1]
    args.ckpt_path_name = args.save_inversion_path + '/' + args.class_choice + '.pt'
    args.save_inversion_path += '/' + 'saved_results'
    trainer = Trainer(args)
    trainer.train()
    
    
