import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
from os.path import join
from tqdm import tqdm
import pdb
import argparse
import pickle
import yaml
import numpy as np
import pymeshlab
import cv2
import trimesh
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop, SGD
from pytorch3d.ops.knn import knn_points
from kaolin.ops.mesh import index_vertices_by_faces, face_normals
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.metrics.pointcloud import chamfer_distance, sided_distance
from datasets.xhumans import XHumans, custom_collate_fn
from models.smplx import SMPLX, batch_rodrigues
from models.mlp import ImplicitNetwork, ColorAE
from models.deformer import hierarchical_softmax
from models.emd_module import emdFunction
from models.utils import get_face_normals, gen_sample, optimize_points, psr, mysample, myestimate_pointcloud_normals


def calc_loss(ae, points, neural_texture, sample_points, sample_color, nums_k=4):
    num_s = torch.ones(sample_points.shape[0],dtype=torch.long,device=points.device) * sample_points.shape[1]
    num_p = torch.ones(points.shape[0],dtype=torch.long,device=points.device) * points.shape[1]
    knn = knn_points(sample_points,points,lengths1=num_s,lengths2=num_p,K=nums_k,return_nn=True)

    distance = ((sample_points.unsqueeze(2) - knn.knn)**2).sum(3)
    # exp_distance = torch.exp(-distance)
    # weight = exp_distance / exp_distance.sum(2,keepdim=True)
    rec_distance = 1 / torch.sqrt(distance)
    weight = rec_distance / rec_distance.sum(2,keepdim=True)
    idx_expanded = knn.idx.unsqueeze(-1).expand(-1,-1,-1,neural_texture.shape[-1]) # b n k f

    texture = (neural_texture.unsqueeze(2).expand(-1,-1,idx_expanded.shape[2],-1).gather(1,idx_expanded) * weight.unsqueeze(-1)).sum(2)

    pred_color = ae.decoder(texture)
    pred_color = torch.sigmoid(pred_color)

    return F.mse_loss(pred_color, sample_color)


def main(config, subject):
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_type = config['data_type']
    dataset_path = config['dataset_path']
    batch_size = config['batch_size']
    soft_blend = config['soft_blend']
    smpl_seg = np.load(config['smpl_seg'], allow_pickle=True).item()
    face_index = smpl_seg['face_index']
    random_lengths = smpl_seg['random_lengths']
    faces_part = smpl_seg['faces_part']
    num_sample = len(face_index)
    use_smplx = config['use_smplx']
    hierarchical = config['hierarchical_softmax']
    estimate_normals = config['estimate_normals']
    neighborhood_size = config['neighborhood_size']
    if estimate_normals:
        delta_out = 3
    else:
        delta_out = 6
    model_type = 'smplx' if use_smplx else 'smpl'
    if use_smplx:
        dout = 59 if hierarchical else 55
    else:
        dout = 25 if hierarchical else 24
    depth = config['depth']
    width = config['width']
    cond_dim = config['cond_dim']
    dim_cond_embed = config['dim_cond_embed']
    multires = config['multires']
    lbsres = config['lbsres']
    skip_layer = config['skip_layer']
    cond_layer = config['cond_layer']
    input_scale = config['input_scale']
    emd = emdFunction.apply
    ms = pymeshlab.MeshSet(verbose=False)

    latent_feature = config['latent_feature']
    color_width = config['color_width']
    color_depth = config['color_depth']
    if_vae = config['if_vae']
    nums_k = config['nums_k']
    ae = ColorAE(latent_feature, color_width, color_depth, if_vae).cuda()
    ae.load_state_dict(torch.load('data/color_ae.pth'))
    for p in ae.parameters():
        p.requires_grad_(False)

    if data_type == 'xhumans':
        dataset = XHumans(dataset_path, subject, use_smplx, num_sample)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn, num_workers=8)
        if use_smplx:
            smpl = SMPLX('data/SMPLX_%s.pkl'%dataset.gender.upper(),is_smplx=True,
                         use_pca=False,hand_mean=True,betas_dim=10,expr_dim=10).cuda()
            betas = torch.from_numpy(dataset.smplx_mean_shape).unsqueeze(0).expand(batch_size,-1).cuda()

        v_template = None

        verts, faces, _, poses, expression, sample_points, sample_normals, colors = dataset.__getitem__(0)
        expression = torch.from_numpy(expression.astype(np.float32)).unsqueeze(0).cuda()
        poses = torch.from_numpy(poses.astype(np.float32)).unsqueeze(0).cuda()
        shape = torch.cat([betas[0:1],expression],dim=1)
        gt_category = np.load(dataset.category_list[0]).astype(np.int64)
        gt_category = torch.from_numpy(gt_category).cuda()
        gt_mesh = trimesh.load(dataset.obj_list[0], process=False, maintain_order=False)
        param = pickle.load(open(dataset.smplx_list[0], 'rb'), encoding='latin1')
        gt_mesh.vertices -= param['transl']
    
    lbs_net = ImplicitNetwork(3,dout,128,5,False,multires=lbsres).cuda()
    delta_net = ImplicitNetwork(3,delta_out,width,depth, geometric_init=False, skip_layer=skip_layer, 
                cond_layer=cond_layer, cond_dim=cond_dim, dim_cond_embed=dim_cond_embed, multires=multires).cuda()
    
    if estimate_normals:
        subject = '%d_estimate'%subject
    else:
        subject = '%d'%subject

    sap = torch.load('trained_models/%s/sap.pth'%subject)
    points = sap['points']
    normals = sap['normals']
    poses_A = sap['poses_A']
    if 'inv_poses_A' in sap.keys():
        inv_poses_A = sap['inv_poses_A']
    else:
        inv_poses_A = poses_A.inverse()
    if 'color' in sap.keys():
        neural_texture = ae.encoder(sap['color'])
    else:
        neural_texture = torch.zeros(1,points.shape[1],latent_feature,device=points.device)

    neural_texture.requires_grad_(True)
    optimizer = Adam([neural_texture], lr=0.001)
    
    lbs_net.load_state_dict(torch.load('trained_models/%s/lbs.pth'%subject))
    delta_net.load_state_dict(torch.load('trained_models/%s/delta.pth'%subject))
    lbs_net.eval()
    delta_net.eval()
    for p in lbs_net.parameters():
        p.requires_grad_(False)
    for p in delta_net.parameters():
        p.requires_grad_(False)

    pbar = tqdm(range(21))
    for i in pbar:
        for idx, data in enumerate(loader):
            if data_type == 'xhumans':
                verts, faces, category, poses, expression, sample_points, sample_normals, colors = data
                poses = poses.cuda()
                expression = expression.cuda()
                sample_points = sample_points.cuda()
                sample_normals = sample_normals.cuda()
                colors = colors.cuda()
                cond = torch.cat([poses[:,3:66]/np.pi, expression], dim=1) # b 69
                shape = torch.cat([betas,expression],dim=1)

            with torch.no_grad():
                cond_delta = delta_net(points.expand(batch_size,-1,-1)*input_scale, cond)
                if estimate_normals:
                    trans_points = points + cond_delta
                else:
                    trans_points = points + cond_delta[...,:3]

                pred_weights = lbs_net(trans_points*input_scale, None) * soft_blend
                if hierarchical:
                    pred_weights = hierarchical_softmax(pred_weights, model_type)
                else:
                    pred_weights = F.softmax(pred_weights,-1)
                _, A = smpl.get_skinning(poses, shape, v_template)

                final_A = torch.einsum('bnij,bnjk->bnik', A, inv_poses_A)
                final_T = torch.einsum('bnd,bdij->bnij', pred_weights, final_A)

                pred_points = torch.einsum('bnij,bnj->bni', final_T[:,:,:3,:3], trans_points) + final_T[:,:,:3,3]

            loss = 50 * calc_loss(ae, pred_points, neural_texture.expand(batch_size,-1,-1), sample_points, colors, nums_k)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            des = 'color:%.4f'%loss.item()
            pbar.set_description(des)

        if i % 10 == 0:
            with torch.no_grad():
                delta = delta_net(points*input_scale, cond[0:1])
                if estimate_normals:
                    trans_points = points + delta
                else:
                    rot_mat = batch_rodrigues(delta[...,3:].reshape(-1,3)).reshape(1,-1,3,3)
                    trans_points = points + delta[...,:3]
                    trans_normals = torch.einsum('bnij,bnj->bni', rot_mat, normals)
                pred_weights = lbs_net(trans_points*input_scale, None) * soft_blend
                if hierarchical:
                    pred_weights = hierarchical_softmax(pred_weights, model_type)
                else:
                    pred_weights = F.softmax(pred_weights,-1)

                final_T = torch.einsum('bnd,bdij->bnij', pred_weights, final_A[0:1])
                pred_points = torch.einsum('bnij,bnj->bni', final_T[:,:,:3,:3], trans_points) + final_T[:,:,:3,3]
                if estimate_normals:
                    pred_normals = myestimate_pointcloud_normals(pred_points, neighborhood_size, disambiguate_directions=False, use_symeig_workaround=False)
                else:
                    pred_normals = torch.einsum('bnij,bnj->bni', final_T[:,:,:3,:3], trans_normals)

                tmp_points = pred_points.squeeze().detach().cpu().numpy()
                tmp_normals = pred_normals.squeeze().detach().cpu().numpy()
                if estimate_normals:
                    tmp_mesh = psr(ms, tmp_points)
                else:
                    tmp_mesh = psr(ms, tmp_points, tmp_normals)

                rec_verts = torch.from_numpy(tmp_mesh.vertices.astype(np.float32)).unsqueeze(0).cuda()
                num_v = torch.ones(rec_verts.shape[0],dtype=torch.long,device=rec_verts.device) * rec_verts.shape[1]
                num_p = torch.ones(pred_points.shape[0],dtype=torch.long,device=pred_points.device) * pred_points.shape[1]
                knn = knn_points(rec_verts,pred_points,lengths1=num_v,lengths2=num_p,K=nums_k,return_nn=True)
                distance = ((rec_verts.unsqueeze(2) - knn.knn)**2).sum(3)
                rec_distance = 1 / torch.sqrt(distance)
                weight = rec_distance / rec_distance.sum(2,keepdim=True)
                idx_expanded = knn.idx.unsqueeze(-1).expand(-1,-1,-1,neural_texture.shape[-1]) # b n k f
                texture = (neural_texture.unsqueeze(2).expand(-1,-1,idx_expanded.shape[2],-1).gather(1,idx_expanded) * weight.unsqueeze(-1)).sum(2)
                pred_color = ae.decoder(texture)
                pred_color = torch.sigmoid(pred_color)
                final_color = (pred_color.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
                color_mesh = trimesh.Trimesh(tmp_mesh.vertices, tmp_mesh.faces, vertex_colors=final_color, process=False, maintain_order=True)
                color_mesh.export('color%d.obj'%i)

            torch.save({'points': points, 'normals': normals, 'poses_A': poses_A, 'inv_poses_A': inv_poses_A, 'neural_texture': neural_texture}, 
                            'trained_models/%s/sap_color.pth'%subject)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/xhumans.yaml')
    parser.add_argument('--subject', type=int, default=27)
    args = parser.parse_args()
    main(args.config, args.subject)