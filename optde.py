import os
import os.path as osp
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.autograd import Function
from model.network import Discriminator, PCN, Encoder, Decoder, Disentangler, Z_Mapper, Classifier, Generator, ViewPredictor

from utils.common_utils import *
from loss import *
from evaluation.pointnet import *
import time
from external.ChamferDistancePytorch.chamfer_python import distChamfer, distChamfer_raw

class RotMatDecoder(nn.Module):
    def __init__(self):
        super(RotMatDecoder, self).__init__()
    def forward(self, x):
        reshaped_x = x.view(-1, 3, 2)
        b1 = F.normalize(reshaped_x[:,:,0], dim=1)
        dot_prod = torch.sum( b1 * reshaped_x[:,:,1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_x[:,:,1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class OptDE(object):

    def __init__(self, args):
        args.vp_mode = 'angle'
        self.args = args

        if self.args.dist:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank, self.world_size = 0, 1
        
        # init seed for static masks: ball_hole, knn_hole, voxel_mask 
        self.to_reset_mask = True 
        self.mask_type = self.args.mask_type
        self.update_G_stages = self.args.update_G_stages
        self.iterations = self.args.iterations
        self.args.G_lrs = [2e-4]
        self.G_lrs = self.args.G_lrs
        self.z_lrs = self.args.z_lrs
        self.select_num = self.args.select_num

        self.loss_log = []
        
        # create model
        self.Encoder = Encoder().cuda()
        #args.DEGREE = [1, 2, 4, 4, 64]
        #args.G_FEAT = [1024, 96, 128, 128, 64, 3]
        #args.G_FEAT[0] = 1024
        args.G_FEAT[0] = 288
        self.Decoder = Generator(features=args.G_FEAT, degrees=args.DEGREE, support=args.support,args=self.args).cuda()#Decoder().cuda()
        self.DI_Disentangler = Disentangler(f_dims=96).cuda()
        self.MS_Disentangler = Disentangler(f_dims=96).cuda()
        self.DS_Disentangler = Disentangler(f_dims=96).cuda()
        #self.Z_Mapper = Z_Mapper(f_dims=84).cuda()
        self.DI_Classifier = Classifier(f_dims=96).cuda()
        self.DS_Classifier = Classifier(f_dims=96).cuda()
        if self.args.vp_mode == 'matrix':
            self.V_Predictor = ViewPredictor(f_dims=96, out_dims=6).cuda()
            self.rotmatdecoder = RotMatDecoder()
        elif self.args.vp_mode == 'angle':
            self.V_Predictor = ViewPredictor(f_dims=96).cuda()
        else:
            raise NotImplementedError
        self.D = Discriminator(features=args.D_FEAT).cuda() 

        #self.models = {"Encoder": self.Encoder, "Decoder": self.Decoder, "DI": self.DI_Disentangler, "DS": self.DS_Disentangler, "Mapper": self.Z_Mapper, "DIC": self.DI_Classifier, "DSC": self.DS_Classifier, "D": self.D}
        self.models = {"Encoder": self.Encoder, "Decoder": self.Decoder, "DI": self.DI_Disentangler, "DS": self.DS_Disentangler, "MS": self.MS_Disentangler, "DIC": self.DI_Classifier, "DSC": self.DS_Classifier, "VP":self.V_Predictor, "D": self.D}
       
        ###Obtain trainable parameters
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS", "DIC", "DSC", "VP"]:
            trainable_params_tmp = self.models[model_name].parameters()
            self.models[model_name].optim = torch.optim.Adam(
                trainable_params_tmp,
                lr=self.G_lrs[0], 
                betas=(0,0.99),
                weight_decay=0,
                eps=1e-8)

        self.z = torch.zeros((1, 288)).normal_().cuda()
        self.z = Variable(self.z, requires_grad=True)
        self.z_optim = torch.optim.Adam([self.z], lr=self.args.z_lrs[0], betas=(0,0.99))

        # load weights
        checkpoint = torch.load(args.ckpt_load, map_location=self.args.device) 
        self.D.load_state_dict(checkpoint['D_state_dict'])

        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS", "DIC", "DSC", "VP"]:
            self.models[model_name].eval()
        if self.D is not None:
            self.D.eval()
    
        # prepare latent variable and optimizer
        self.schedulers = dict()
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS", "DIC", "DSC", "VP"]:
            self.schedulers[model_name] = LRScheduler(self.models[model_name].optim, self.args.warm_up)
        self.z_scheduler = LRScheduler(self.z_optim, self.args.warm_up)

        # loss functions
        self.ftr_net = self.D
        self.criterion = DiscriminatorLoss()
        self.di_criterion = nn.CrossEntropyLoss()
        self.ds_criterion = nn.CrossEntropyLoss()
        self.vp_criterion = nn.MSELoss()
        self.consistency_criterion = nn.MSELoss()

        self.directed_hausdorff = DirectedHausdorff()

        # for visualization
        self.checkpoint_pcd = [] # to save the staged checkpoints
        self.checkpoint_flags = [] # plot subtitle

        
        if len(args.w_D_loss) == 1:
            self.w_D_loss = args.w_D_loss * len(args.G_lrs)
        else:
            self.w_D_loss = args.w_D_loss

    def finetune(self, bool_gt=False, save_curve=False, ith=-1):
        # forward
        if bool_gt:
            self.args.G_lrs = [2e-7, 1e-6, 1e-6, 2e-7]
            self.args.z_lrs = [9e-3, 2e-3, 1e-3, 1e-6]
            self.iterations = [0, 0, 2, 1]
            #self.iterations = [24, 8, 2, 4]
            #self.iterations = [12, 2, 1, 1]
            self.k_mask_k = [1, 1, 1, 1]
        else:
            self.args.G_lrs = [2e-7, 1e-6, 1e-6, 2e-7]
            self.args.z_lrs = [9e-3, 2e-3, 1e-3, 1e-6]
            self.iterations = [1, 4, 4, 1]
            self.k_mask_k = [1, 1, 1, 1]
        tree = [self.partial]
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.models[model_name].eval()
        with torch.no_grad():
            hidden_z = self.Encoder(tree)
            f_di = self.DI_Disentangler(hidden_z)
            #f_di_c = self.Z_Mapper(f_di)
            f_ms = self.MS_Disentangler(hidden_z)
            f_ds = self.DS_Disentangler(hidden_z)
            f_combine_c = torch.cat([f_di, f_ms*0., f_ds], 1)
            self.z.copy_(f_combine_c)
        loss_dict = {}
        curr_step = 0
        count = 0
        if save_curve:
            cd_curve_list = []
        for stage, iteration in enumerate(self.iterations):

            for i in range(iteration):
                curr_step += 1
                # setup learning rate
                self.schedulers['Decoder'].update(curr_step, self.args.G_lrs[stage])
                self.z_scheduler.update(curr_step, self.args.z_lrs[stage])

                # forward
                self.z_optim.zero_grad()
                
                if self.update_G_stages[stage]:
                    self.Decoder.optim.zero_grad()
                             
                tree = self.z
                x = self.Decoder(tree)
                
                # masking
                x_map = self.pre_process(x,stage=stage)

                ### compute losses
                ftr_loss = self.criterion(self.ftr_net, x_map, self.partial)

                dist1, dist2 , _, _ = distChamfer(self.partial, x)
                ucd_loss = dist1.mean()
                dist1, dist2 , _, _ = distChamfer(x_map, self.partial)
                cd_loss = dist1.mean() + dist2.mean()
                if self.gt is not None:
                    dist1, dist2 , _, _ = distChamfer(x, self.gt)
                    gt_cd_loss = dist1.mean() + dist2.mean()
                # optional early stopping
                if self.args.early_stopping:
                    if cd_loss.item() < self.args.stop_cd:
                        break

                # nll corresponds to a negative log-likelihood loss
                nll = self.z**2 / 2
                nll = nll.mean()
                
                ### loss
                loss = ftr_loss * self.w_D_loss[0] + nll * self.args.w_nll \
                        + cd_loss * 1
                
                # optional to use directed_hausdorff
                directed_hausdorff_loss = self.directed_hausdorff(self.partial.permute([0,2,1]), x.permute([0,2,1]))
                if self.args.directed_hausdorff:
                    print("Using Hausdorff")
                    loss += directed_hausdorff_loss*self.args.w_directed_hausdorff_loss
                
                # backward
                loss.backward()
                self.z_optim.step()
                if self.update_G_stages[stage]:
                    self.Decoder.optim.step()
                if save_curve:
                    if self.gt is not None:
                        dist1, dist2 , _, _ = distChamfer(x,self.gt)
                        test_cd = dist1.mean() + dist2.mean()
                    cd_curve_list.append(test_cd.item())
                    self.x = x
                    if not osp.isdir(self.args.save_inversion_path):
                        os.mkdir(self.args.save_inversion_path)
                    x_np = x[0].detach().cpu().numpy()
                    if ith == -1:
                        basename = str(self.pcd_id)
                    else:
                        basename = str(self.pcd_id)+'_'+str(ith)
                    np.savetxt(osp.join(self.args.save_inversion_path,basename+"_%.4d"%curr_step+'_x.txt'), x_np, fmt = "%f;%f;%f")  

            # save checkpoint for each stage
            #self.checkpoint_flags.append('s_'+str(stage)+' x')
            #self.checkpoint_pcd.append(x)
            #self.checkpoint_flags.append('s_'+str(stage)+' x_map')
            #self.checkpoint_pcd.append(x_map)
                
        ### save point clouds
        self.x = x
        if not osp.isdir(self.args.save_inversion_path):
            os.mkdir(self.args.save_inversion_path)
        x_np = x[0].detach().cpu().numpy()
        #x_map_np = x_map[0].detach().cpu().numpy()
        partial_np = self.partial[0].detach().cpu().numpy()
        if ith == -1:
            basename = str(self.pcd_id)
        else:
            basename = str(self.pcd_id)+'_'+str(ith)
        if self.gt is not None:
            gt_np = self.gt[0].detach().cpu().numpy()
            np.savetxt(osp.join(self.args.save_inversion_path,basename+'_gt.txt'), gt_np, fmt = "%f;%f;%f")  
        np.savetxt(osp.join(self.args.save_inversion_path,basename+'_x.txt'), x_np, fmt = "%f;%f;%f")  
        #np.savetxt(osp.join(self.args.save_inversion_path,basename+'_xmap.txt'), x_map_np, fmt = "%f;%f;%f")  
        np.savetxt(osp.join(self.args.save_inversion_path,basename+'_partial.txt'), partial_np, fmt = "%f;%f;%f")  
        if save_curve:
            cd_curve_arr = np.array(cd_curve_list)
            np.save(osp.join(self.args.save_inversion_path,basename+'_cd_curve.npy'),cd_curve_arr)

        if bool_gt:
            return ucd_loss.item(), directed_hausdorff_loss.item(), gt_cd_loss.item()
        else:
            return ucd_loss.item(), directed_hausdorff_loss.item()
    
    def reset_G_tmp(self): 
        """
        to save the pc for visualizaton 
        """
        self.checkpoint_pcd = [] # to save the staged checkpoints
        self.checkpoint_flags = []
        if self.mask_type == 'voxel_mask':
            self.to_reset_mask = True # reset hole center for each shape 
    
    def reset_whole_network(self, load_path_name):
        checkpoint = torch.load(load_path_name, map_location=self.args.device)
        self.Encoder.load_state_dict(checkpoint['Encoder_state_dict'])
        self.Decoder.load_state_dict(checkpoint['Decoder_state_dict'])
        self.DI_Disentangler.load_state_dict(checkpoint['DI'])
        self.DS_Disentangler.load_state_dict(checkpoint['DS'])
        self.MS_Disentangler.load_state_dict(checkpoint['MS'])
        #self.Z_Mapper.load_state_dict(checkpoint['Mapper'])
        self.DI_Classifier.load_state_dict(checkpoint['DIC'])
        self.DS_Classifier.load_state_dict(checkpoint['DSC'])
        return

    def set_virtual_real(self, virtual_partial=None, real_partial=None, rotmat=None, azel=None):
        if virtual_partial is not None:
            self.virtual_partial = virtual_partial
        if real_partial is not None:
            self.real_partial = real_partial
        if azel is not None:
            self.azel = azel
        if rotmat is not None:
            self.rotmat = rotmat

    def set_target(self, gt=None, partial=None):
        '''
        set partial and gt 
        '''
        if gt is not None:
            if len(gt.shape) == 2:
                self.gt = gt.unsqueeze(0)
            else:
                self.gt = gt
            # for visualization
            self.checkpoint_flags.append('GT')
            self.checkpoint_pcd.append(self.gt)
        else:
            self.gt = None
        
        if partial is not None:
            if self.args.target_downsample_method.lower() == 'fps':
                partial_size = self.args.target_downsample_size
                if len(partial.shape)==2:
                    self.partial = self.downsample(partial.unsqueeze(0), partial_size)
                else:
                    self.partial = self.downsample(partial, partial_size)
            else:
                if len(partial.shape)==2:
                    self.partial = partial.unsqueeze(0)
                else:
                    self.partial = partial
        else:
            self.partial = self.pre_process(self.gt, stage=-1)
        # for visualization
        self.checkpoint_flags.append('partial') 
        self.checkpoint_pcd.append(self.partial)
    
    def run(self, ith=-1):
        self.train_one_batch(ith)
        return

    def test_one_batch(self, use_ema=False, ith=-1):
        loss_dict = {}
        count = 0
        stage = 0

        # forward
        tree = [self.partial]
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.models[model_name].eval()
        hidden_z = self.Encoder(tree)
        f_di = self.DI_Disentangler(hidden_z)
        #f_di_c = self.Z_Mapper(f_di)
        f_ms = self.MS_Disentangler(hidden_z)
        f_ds = self.DS_Disentangler(hidden_z)
        f_combine_c = torch.cat([f_di, f_ms*0., f_ds], 1)
        x = self.Decoder(f_combine_c)
            
        ### compute losses
        #ftr_loss = self.criterion(self.ftr_net, x_map, self.target)
        ftr_loss = self.criterion(self.ftr_net, x, self.gt)

        #dist1, dist2 , _, _ = distChamfer(x_map, self.target)
        dist1, dist2 , _, _ = distChamfer(x, self.gt)
        cd_loss = dist1.mean() + dist2.mean()

        # nll corresponds to a negative log-likelihood loss
        #nll = self.z**2 / 2
        
        ### loss
        loss = ftr_loss * self.w_D_loss[0] \
                + cd_loss * 1
            
        # save checkpoint for each stage
        self.checkpoint_flags.append('s_'+str(stage)+' x')
        self.checkpoint_pcd.append(x)
        #self.checkpoint_flags.append('s_'+str(stage)+' x_map')
        #self.checkpoint_pcd.append(x_map)

        # test only for each stage
        if self.gt is not None:
            dist1, dist2 , _, _ = distChamfer(x,self.gt)
            test_cd = dist1.mean() + dist2.mean()
        
        if self.gt is not None:
            loss_dict = {
                'ftr_loss': np.asscalar(ftr_loss.detach().cpu().numpy()),
                #'nll': np.asscalar(nll.detach().cpu().numpy()),
                'cd': np.asscalar(test_cd.detach().cpu().numpy()),
            }
            self.loss_log.append(loss_dict)
                
        ### save point clouds
        self.x = x
        if not osp.isdir(self.args.save_inversion_path):
            os.mkdir(self.args.save_inversion_path)
        x_np = x[0].detach().cpu().numpy()
        #x_map_np = x_map[0].detach().cpu().numpy()
        partial_np = self.partial[0].detach().cpu().numpy()
        if ith == -1:
            basename = str(self.pcd_id)
        else:
            basename = str(self.pcd_id)+'_'+str(ith)
        if self.gt is not None:
            gt_np = self.gt[0].detach().cpu().numpy()
            #np.savetxt(osp.join(self.args.save_inversion_path,basename+'_gt.txt'), gt_np, fmt = "%f;%f;%f")  
        #np.savetxt(osp.join(self.args.save_inversion_path,basename+'_x.txt'), x_np, fmt = "%f;%f;%f")  
        #np.savetxt(osp.join(self.args.save_inversion_path,basename+'_xmap.txt'), x_map_np, fmt = "%f;%f;%f")  
        #np.savetxt(osp.join(self.args.save_inversion_path,basename+'_partial.txt'), partial_np, fmt = "%f;%f;%f")  
        return test_cd.item()

    def test_real_one_batch(self, bool_gt=False, use_ema=False, ith=-1):
        loss_dict = {}
        count = 0
        stage = 0

        # forward
        tree = [self.partial]
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.models[model_name].eval()
        hidden_z = self.Encoder(tree)
        f_di = self.DI_Disentangler(hidden_z)
        #f_di_c = self.Z_Mapper(f_di)
        f_ms = self.MS_Disentangler(hidden_z)
        f_ds = self.DS_Disentangler(hidden_z)
        f_combine_c = torch.cat([f_di, f_ms*0., f_ds], 1)
        x = self.Decoder(f_combine_c)
            
        ### compute losses
        #ftr_loss = self.criterion(self.ftr_net, x_map, self.target)
        #ftr_loss = self.criterion(self.ftr_net, x, self.gt)

        #dist1, dist2 , _, _ = distChamfer(x_map, self.target)
        dist1, dist2 , _, _ = distChamfer(self.partial, x)
        directed_cd_loss = dist1.mean()
        if bool_gt:
            dist1, dist2 , _, _ = distChamfer(x, self.gt)
            cd_loss = dist1.mean() + dist2.mean()

        # nll corresponds to a negative log-likelihood loss
        #nll = self.z**2 / 2
        #nll = hidden_z**2 / 2
        #nll = nll.mean()
        
        ### loss
        #loss = ftr_loss * self.w_D_loss[0] + nll * self.args.w_nll \
        #        + cd_loss * 1
        ucd_loss = directed_cd_loss 

        uhd_loss = self.directed_hausdorff(self.partial.permute([0,2,1]), x.permute([0,2,1]))
        #uhd_loss = dist1.max()
        #loss += directed_hausdorff_loss*self.args.w_directed_hausdorff_loss * 0.001
            
        # save checkpoint for each stage
        self.checkpoint_flags.append('s_'+str(stage)+' x')
        self.checkpoint_pcd.append(x)
        #self.checkpoint_flags.append('s_'+str(stage)+' x_map')
        #self.checkpoint_pcd.append(x_map)

        # test only for each stage
        
        #if self.gt is not None:
        #    loss_dict = {
        #        'ftr_loss': np.asscalar(ftr_loss.detach().cpu().numpy()),
        #        'nll': np.asscalar(nll.detach().cpu().numpy()),
        #        'cd': np.asscalar(test_cd.detach().cpu().numpy()),
        #    }
        #    self.loss_log.append(loss_dict)
                
        ### save point clouds
        self.x = x
        if not osp.isdir(self.args.save_inversion_path):
            os.mkdir(self.args.save_inversion_path)
        x_np = x[0].detach().cpu().numpy()
        #x_map_np = x_map[0].detach().cpu().numpy()
        partial_np = self.partial[0].detach().cpu().numpy()
        if ith == -1:
            basename = str(self.pcd_id)
        else:
            basename = str(self.pcd_id)+'_'+str(ith)
        if self.gt is not None:
            gt_np = self.gt[0].detach().cpu().numpy()
            np.savetxt(osp.join(self.args.save_inversion_path,basename+'_gt.txt'), gt_np, fmt = "%f;%f;%f")  
        np.savetxt(osp.join(self.args.save_inversion_path,basename+'_x.txt'), x_np, fmt = "%f;%f;%f")  
        #np.savetxt(osp.join(self.args.save_inversion_path,basename+'_xmap.txt'), x_map_np, fmt = "%f;%f;%f")  
        np.savetxt(osp.join(self.args.save_inversion_path,basename+'_partial.txt'), partial_np, fmt = "%f;%f;%f")  
        if bool_gt:
            return ucd_loss.item(), uhd_loss.item(), cd_loss.item()
        else:
            return ucd_loss.item(), uhd_loss.item()

    def train_consistency_one_batch(self, curr_step, cons_feature, return_generated=False, ith=-1):
        loss_dict = {}
        count = 0
        stage = 0
        consistency_loss_value = 0
        # setup learning rate
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.schedulers[model_name].update(curr_step, self.args.G_lrs[0], ratio=0.99998)
        # forward
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.models[model_name].eval()
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.models[model_name].optim.zero_grad()
        x = self.Decoder(cons_feature)
        tree = [x]
        hidden_z = self.Encoder(tree)
        f_di = self.DI_Disentangler(hidden_z)
        f_ms = self.MS_Disentangler(hidden_z)
        f_ds = self.DS_Disentangler(hidden_z)
        f_combine = torch.cat([f_di, f_ms, f_ds], 1)
        consistency_loss = self.consistency_criterion(f_combine, cons_feature)
        consistency_loss_value += consistency_loss.item()
        consistency_loss *= 0.06
        consistency_loss.backward()
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.models[model_name].optim.step()
        if return_generated:
            return x
        else:
            return consistency_loss_value
    def train_domain_one_batch(self, curr_step, alpha, switch_idx_default=None, ith=-1):
        loss_dict = {}
        count = 0
        stage = 0
        di_loss_value = 0
        ds_loss_value = 0
        vp_loss_value = 0
        # setup learning rate
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS", "DIC", "DSC", "VP"]:
            self.schedulers[model_name].update(curr_step, self.args.G_lrs[0], ratio=0.99998)

        # forward
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS", "DIC", "DSC", "VP"]:
            self.models[model_name].eval()
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS", "DIC", "DSC", "VP"]:
            self.models[model_name].optim.zero_grad()
                     
        #tree = [self.z]
        virtual_tree = [self.virtual_partial]
        virtual_hidden_z = self.Encoder(virtual_tree)
        virtual_f_di = self.DI_Disentangler(virtual_hidden_z)
        virtual_f_di = ReverseLayerF.apply(virtual_f_di, alpha)
        virtual_f_ds = self.DS_Disentangler(virtual_hidden_z)
        virtual_f_ms = self.MS_Disentangler(virtual_hidden_z)
        virtual_f_di_cl = self.DI_Classifier(virtual_f_di)
        virtual_f_ds_cl = self.DS_Classifier(virtual_f_ds)
        virtual_f_vp = self.V_Predictor(virtual_f_ms)
        di_s_label = torch.zeros(virtual_f_di_cl.shape[0]).long().cuda()
        ds_s_label = torch.zeros(virtual_f_ds_cl.shape[0]).long().cuda()
        virtual_di_loss = self.di_criterion(virtual_f_di_cl, di_s_label)
        virtual_ds_loss = self.ds_criterion(virtual_f_ds_cl, ds_s_label)
        if self.args.vp_mode == 'matrix':
            virtual_f_vp = self.rotmatdecoder(virtual_f_vp)
            virtual_vp_loss = self.vp_criterion(virtual_f_vp, self.rotmat)
        elif self.args.vp_mode == 'angle':
            virtual_vp_loss = self.vp_criterion(virtual_f_vp, self.azel)
        else:
            raise NotImplementedError
        di_loss_value += virtual_di_loss.item()
        ds_loss_value += virtual_ds_loss.item()
        vp_loss_value += virtual_vp_loss.item()
        virtual_loss = (virtual_di_loss * 0.01 + virtual_ds_loss + virtual_vp_loss) * 0.004
        virtual_loss.backward()

        real_tree = [self.real_partial]
        real_hidden_z = self.Encoder(real_tree)
        real_f_di = self.DI_Disentangler(real_hidden_z)
        real_f_di = ReverseLayerF.apply(real_f_di, alpha)
        real_f_ds = self.DS_Disentangler(real_hidden_z)
        real_f_ms = self.MS_Disentangler(real_hidden_z)
        real_f_di_cl = self.DI_Classifier(real_f_di)
        real_f_ds_cl = self.DS_Classifier(real_f_ds)
        di_t_label = torch.ones(real_f_di_cl.shape[0]).long().cuda()
        ds_t_label = torch.ones(real_f_ds_cl.shape[0]).long().cuda()
        real_di_loss = self.di_criterion(real_f_di_cl, di_t_label)
        real_ds_loss = self.ds_criterion(real_f_ds_cl, ds_t_label)
        di_loss_value += real_di_loss.item()
        ds_loss_value += real_ds_loss.item()
        real_loss = (real_di_loss * 0.01 + real_ds_loss) * 0.004
        real_loss.backward()
        with torch.no_grad():
            cons_feature_di = torch.cat([virtual_f_di, real_f_di], 0)
            cons_feature_ds = torch.cat([virtual_f_ds, real_f_ds], 0)
            cons_feature_ms = torch.cat([virtual_f_ms, real_f_ms], 0)
            if switch_idx_default is None:
                switch_idx = np.random.randint(3)
            else:
                switch_idx = switch_idx_default
            switch_perm = np.random.permutation(cons_feature_di.shape[0])
            batch_size = cons_feature_di.shape[0]
            switch_perm = np.arange(batch_size)
            switch_perm = np.concatenate([switch_perm[batch_size//2:],switch_perm[:batch_size//2]], axis=0)
            if switch_idx == 0:
                pass
                cons_feature_di = cons_feature_di[switch_perm]
            elif switch_idx == 1:
                cons_feature_ms = cons_feature_ms[switch_perm]
            elif switch_idx == 2:
                cons_feature_ds = cons_feature_ds[switch_perm]
            cons_feature = torch.cat([cons_feature_di,cons_feature_ms, cons_feature_ds], 1)

        
        #print(self.G.optim.state_dict()['param_groups'][0]['lr'])
        for model_name in ["Encoder", "DI", "DS", "MS", "DIC", "DSC", "VP"]:
            self.models[model_name].optim.step()

        # save checkpoint for each stage
        #self.checkpoint_flags.append('s_'+str(stage)+' x')
        #self.checkpoint_pcd.append(x)
        #self.checkpoint_flags.append('s_'+str(stage)+' x_map')
        #self.checkpoint_pcd.append(x_map)

        # test only for each stage
        
        return di_loss_value, ds_loss_value, vp_loss_value, cons_feature
    def train_one_batch(self, curr_step, ith=-1, complete_train=False):
        loss_dict = {}
        count = 0
        stage = 0
        # setup learning rate
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.schedulers[model_name].update(curr_step, self.args.G_lrs[0], ratio=0.99998)

        # forward
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.models[model_name].eval()
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.models[model_name].optim.zero_grad()
                     
        #tree = [self.z]
        tree = [self.partial]
        hidden_z = self.Encoder(tree)
        f_di = self.DI_Disentangler(hidden_z)
        f_ms = self.MS_Disentangler(hidden_z)
        #f_di_c = self.Z_Mapper(f_di)
        f_ds = self.DS_Disentangler(hidden_z)
        f_combine = torch.cat([f_di, f_ms, f_ds], 1)
        f_combine_c = torch.cat([f_di, f_ms*0., f_ds], 1)
        x_rec = self.Decoder(f_combine)
        x = self.Decoder(f_combine_c)
        
        # masking
        #x_map = self.pre_process(x,stage=stage)

        ### compute losses
        #ftr_loss = self.criterion(self.ftr_net, x_map, self.target)
        ftr_loss = self.criterion(self.ftr_net, x, self.gt)

        #dist1, dist2 , _, _ = distChamfer(x_map, self.target)
        dist1, dist2 , _, _ = distChamfer(x_rec, self.partial)
        rec_cd_loss = dist1.mean() + dist2.mean()
        rec_cd_loss *= 0.5

        dist1, dist2 , _, _ = distChamfer(x, self.gt)
        cd_loss = dist1.mean() + dist2.mean()

        # nll corresponds to a negative log-likelihood loss
        #nll = self.z**2 / 2
        nll_ms = f_ms**2 / 2
        nll_ms = nll_ms.mean()
        nll = f_combine**2 / 2
        nll = nll.mean()
        
        ### loss
        if complete_train:
            loss = ftr_loss * self.w_D_loss[0] + nll_ms * self.args.w_nll \
                    + cd_loss * 1 + rec_cd_loss
        else:
            loss = ftr_loss * self.w_D_loss[0] + nll * self.args.w_nll \
                    + cd_loss * 1 + rec_cd_loss
        
        # optional to use directed_hausdorff

        #if self.args.directed_hausdorff:
        #    directed_hausdorff_loss = self.directed_hausdorff(self.target, x)
        #    loss += directed_hausdorff_loss*self.args.w_directed_hausdorff_loss
        
        # backward
        loss.backward()
        #print(self.G.optim.state_dict()['param_groups'][0]['lr'])
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.models[model_name].optim.step()

        # save checkpoint for each stage
        self.checkpoint_flags.append('s_'+str(stage)+' x')
        self.checkpoint_pcd.append(x)
        #self.checkpoint_flags.append('s_'+str(stage)+' x_map')
        #self.checkpoint_pcd.append(x_map)

        # test only for each stage
        if self.gt is not None:
            dist1, dist2 , _, _ = distChamfer(x,self.gt)
            test_cd = dist1.mean() + dist2.mean()
        
        if self.gt is not None:
            loss_dict = {
                'ftr_loss': np.asscalar(ftr_loss.detach().cpu().numpy()),
                'nll': np.asscalar(nll.detach().cpu().numpy()),
                'cd': np.asscalar(test_cd.detach().cpu().numpy()),
            }
            self.loss_log.append(loss_dict)
        return test_cd.item()

    def train_real_one_batch(self, curr_step, epoch, ith=-1):
        loss_dict = {}
        count = 0
        stage = 0
        # setup learning rate
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.schedulers[model_name].update(curr_step, self.args.G_lrs[0], ratio=0.99998)

        # forward
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.models[model_name].eval()
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.models[model_name].optim.zero_grad()
                     
        #tree = [self.z]
        tree = [self.partial]
        hidden_z = self.Encoder(tree)
        f_di = self.DI_Disentangler(hidden_z)
        #f_di_c = self.Z_Mapper(f_di)
        f_ms = self.MS_Disentangler(hidden_z)
        f_ds = self.DS_Disentangler(hidden_z)
        f_combine = torch.cat([f_di, f_ms, f_ds], 1)
        f_combine_c = torch.cat([f_di, f_ms*0., f_ds], 1)
        x_rec = self.Decoder(f_combine)
        x = self.Decoder(f_combine_c)
        
        # masking
        self.args.masking_option = "indexing"
        x_map = self.pre_process(x,stage=stage)

        ### compute losses
        #ftr_loss = self.criterion(self.ftr_net, x_map, self.target)
        #ftr_loss = self.criterion(self.ftr_net, x, self.gt)
        dist1, dist2 , _, _ = distChamfer(self.partial, x_rec)
        rec_cd_loss = dist1.mean() + dist2.mean()

        dist1, dist2 , _, _ = distChamfer(self.partial, x)
        #cd_loss = dist1.mean() + dist2.mean()
        directed_cd_loss = dist1.mean()
        #
        dist1, dist2, _, _ = distChamfer(x_map, self.partial)
        mask_cd_loss = dist1.mean()

        # nll corresponds to a negative log-likelihood loss
        #nll = self.z**2 / 2
        nll = f_combine**2 / 2
        nll = nll.mean()
        
        ### loss
        #loss = directed_cd_loss + rec_cd_loss * 0.15 + mask_cd_loss * 0.01
        if epoch < 160:
            loss = rec_cd_loss * 0.15 + nll * self.args.w_nll
        else:
            loss = directed_cd_loss + rec_cd_loss * 0.15 + nll * self.args.w_nll
        
        # optional to use directed_hausdorff

        #if self.args.directed_hausdorff:
        #directed_hausdorff_loss = self.directed_hausdorff(self.partial, x)
        #loss += directed_hausdorff_loss*self.args.w_directed_hausdorff_loss * 0.001
        
        # backward
        loss.backward()
        #print(self.G.optim.state_dict()['param_groups'][0]['lr'])
        for model_name in ["Encoder", "Decoder", "DI", "DS", "MS"]:
            self.models[model_name].optim.step()

        # save checkpoint for each stage
        self.checkpoint_flags.append('s_'+str(stage)+' x')
        self.checkpoint_pcd.append(x)
        #self.checkpoint_flags.append('s_'+str(stage)+' x_map')
        #self.checkpoint_pcd.append(x_map)

        # test only for each stage
        #directed_hausdorff_loss = self.directed_hausdorff(self.partial, x)
        return loss.item()
    
    def select_z(self, select_y=False):
        tic = time.time()
        with torch.no_grad():
            self.select_num = 0
            if self.select_num == 0:
                self.z.zero_()
                return

    def save_checkpoint(self, ckpt_path_name):
        torch.save({
            'Encoder_state_dict': self.Encoder.state_dict(),
            'Decoder_state_dict': self.Decoder.state_dict(),
            'DI': self.DI_Disentangler.state_dict(),
            'DS': self.DS_Disentangler.state_dict(),
            #'Mapper': self.Z_Mapper.state_dict(),
            'MS': self.MS_Disentangler.state_dict(),
            'DIC': self.DI_Classifier.state_dict(),
            'DSC': self.DS_Classifier.state_dict()
            }, ckpt_path_name)
        return
    
    def pre_process(self,pcd,stage=-1):
        """
        transfer a pcd in the observation space:
        with the following mask_type:
            none: for ['reconstruction', 'jittering', 'morphing']
            ball_hole, knn_hole: randomly create the holes from complete pcd, similar to PF-Net
            voxel_mask: baseline in ShapeInversion
            tau_mask: baseline in ShapeInversion
            k_mask: proposed component by ShapeInversion
        """
        
        if self.mask_type == 'none':
            return pcd
        elif self.mask_type in ['ball_hole', 'knn_hole']:
            ### set static mask for each new partial pcd
            if self.to_reset_mask:
                # either ball hole or knn_hole, hence there might be unused configs
                self.hole_k = self.args.hole_k
                self.hole_radius = self.args.hole_radius
                self.hole_n = self.args.hole_n
                seeds = farthest_point_sample(pcd, self.hole_n) # shape (B,hole_n)
                self.hole_centers = torch.stack([img[seed] for img, seed in zip(pcd,seeds)]) # (B, hole_n, 3)
                # turn off mask after set mask, until next partial pcd
                self.to_reset_mask = False
            
            ### preprocess
            flag_map = torch.ones(1,2048,1).cuda()
            pcd_new = pcd.unsqueeze(2).repeat(1,1,self.hole_n,1)
            seeds_new = self.hole_centers.unsqueeze(1).repeat(1,2048,1,1)
            delta = pcd_new.add(-seeds_new) # (B, 2048, hole_n, 3)
            dist_mat = torch.norm(delta,dim=3)
            dist_new = dist_mat.transpose(1,2) # (B, hole_n, 2048)

            if self.mask_type == 'knn_hole':
                # idx (B, hole_n, hole_k), dist (B, hole_n, hole_k)
                dist, idx = torch.topk(dist_new,self.hole_k,largest=False)                
            
            for i in range(self.hole_n):
                dist_per_hole = dist_new[:,i,:].unsqueeze(2)
                if self.mask_type == 'knn_hole':
                    threshold_dist = dist[:,i, -1]
                if self.mask_type == 'ball_hole': 
                    threshold_dist = self.hole_radius
                flag_map[dist_per_hole <= threshold_dist] = 0
            
            target = torch.mul(pcd, flag_map)
            return target    
        elif self.mask_type == 'voxel_mask':
            """
            voxels in the partial and optionally surroundings are 1, the rest are 0.
            """  
            ### set static mask for each new partial pcd
            if self.to_reset_mask:
                mask_partial = self.voxelize(self.target, n_bins=self.args.voxel_bins, pcd_limit=0.5, threshold=0)
                # optional to add surrounding to the mask partial
                surrounding = self.args.surrounding 
                self.mask_dict = {}
                for key_gt in mask_partial:
                    x,y,z = key_gt
                    surrounding_ls = []
                    surrounding_ls.append((x,y,z))
                    for x_s in range(x-surrounding+1, x+surrounding):
                        for y_s in range(y-surrounding+1, y+surrounding):
                            for z_s in range(z-surrounding+1, z+surrounding):
                                surrounding_ls.append((x_s,y_s,z_s))
                    for xyz in surrounding_ls:
                        self.mask_dict[xyz] = 1
                # turn off mask after set mask, until next partial pcd
                self.to_reset_mask = False 
            
            ### preprocess
            n_bins = self.args.voxel_bins
            mask_tensor = torch.zeros(2048,1)
            pcd_new = pcd*n_bins + n_bins * 0.5
            pcd_new = pcd_new.type(torch.int64)
            ls_voxels = pcd_new.squeeze(0).tolist() # 2028 of sublists
            tuple_voxels = [tuple(itm) for itm in ls_voxels]
            for i in range(2048):
                tuple_voxel = tuple_voxels[i] 
                if tuple_voxel in self.mask_dict:
                    mask_tensor[i] = 1
            
            mask_tensor = mask_tensor.unsqueeze(0).cuda()
            pcd_map = torch.mul(pcd, mask_tensor)
            return pcd_map    
        elif self.mask_type == 'k_mask':
            pcd_map = self.k_mask(self.partial, pcd,stage)
            return pcd_map
        elif self.mask_type == 'tau_mask':
            pcd_map = self.tau_mask(self.target, pcd,stage)
            return pcd_map
        else:
            raise NotImplementedError

    def voxelize(self, pcd, n_bins=32, pcd_limit=0.5, threshold=0):
        """
        given a partial/GT pcd
        return {0,1} masks with resolution n_bins^3
        voxel_limit in case the pcd is very small, but still assume it is symmetric
        threshold is needed, in case we would need to handle noise
        the form of output is a dict, key (x,y,z) , value: count
        """
        pcd_new = pcd * n_bins + n_bins * 0.5
        pcd_new = pcd_new.type(torch.int64)
        ls_voxels = pcd_new.squeeze(0).tolist() # 2028 of sublists
        tuple_voxels = [tuple(itm) for itm in ls_voxels]
        mask_dict = {}
        for tuple_voxel in tuple_voxels:
            if tuple_voxel not in mask_dict:
                mask_dict[tuple_voxel] = 1
            else:
                mask_dict[tuple_voxel] += 1
        for voxel, cnt in mask_dict.items():
            if cnt <= threshold:
                del mask_dict[voxel]
        return mask_dict   
    
    def tau_mask(self, target, x, stage=-1):
        """
        tau mask
        """
        # dist_mat shape (B, N_target, N_output), where B = 1
        stage = max(0, stage)
        dist_tau = self.args.tau_mask_dist[stage]
        dist_mat = distChamfer_raw(target, x)
        idx0, idx1, idx2 = torch.where(dist_mat<dist_tau) 
        idx = torch.unique(idx2).type(torch.long)
        x_map = x[:, idx]
        return x_map
        
    def k_mask(self, target, x, stage=-1):
        """
        masking based on CD.
        target: (1, N, 3), partial, can be < 2048, 2048, > 2048
        x: (1, 2048, 3)
        x_map: (1, N', 3), N' < 2048
        x_map: v1: 2048, 0 masked points
        """
        stage = max(0, stage)
        knn = self.args.k_mask_k[stage]
        if knn == 1:
            cd1, cd2, argmin1, argmin2 = distChamfer(target, x)
            idx = torch.unique(argmin1).type(torch.long)
        elif knn > 1:
            # dist_mat shape (B, 2048, 2048), where B = 1
            dist_mat = distChamfer_raw(target, x)
            # indices (B, 2048, k)
            val, indices = torch.topk(dist_mat, k=knn, dim=2,largest=False)
            # union of all the indices
            idx = torch.unique(indices).type(torch.long)

        if self.args.masking_option == 'element_product':   
            mask_tensor = torch.zeros(2048,1)
            mask_tensor[idx] = 1
            mask_tensor = mask_tensor.cuda().unsqueeze(0)
            x_map = torch.mul(x, mask_tensor) 
        elif self.args.masking_option == 'indexing':  
            x_map = x[:, idx]

        return x_map

    def downsample(self, dense_pcd, n=2048):
        """
        input pcd cpu tensor
        return downsampled cpu tensor
        """
        idx = farthest_point_sample(dense_pcd,n)
        sparse_pcd = dense_pcd[0,idx]
        return sparse_pcd



