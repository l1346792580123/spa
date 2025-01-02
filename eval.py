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
import kaolin
from kaolin.ops.mesh import index_vertices_by_faces, face_normals
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.metrics.pointcloud import chamfer_distance
from datasets.xhumans import XHumans, custom_collate_fn
from models.smplx import SMPLX, batch_rodrigues
from models.mlp import ImplicitNetwork
from models.deformer import hierarchical_softmax
from models.emd_module import emdFunction
from models.utils import get_face_normals, gen_sample, optimize_points, psr, myestimate_pointcloud_normals

def compute_iou_w_mesh(mesh, gt_mesh):
    mesh_bounds = mesh.bounds
    gt_mesh_bounds = gt_mesh.bounds
    xx1 = np.max([mesh_bounds[0, 0], gt_mesh_bounds[0, 0]])
    yy1 = np.max([mesh_bounds[0, 1], gt_mesh_bounds[0, 1]])
    zz1 = np.max([mesh_bounds[0, 2], gt_mesh_bounds[0, 2]])

    xx2 = np.min([mesh_bounds[1, 0], gt_mesh_bounds[1, 0]])
    yy2 = np.min([mesh_bounds[1, 1], gt_mesh_bounds[1, 1]])
    zz2 = np.min([mesh_bounds[1, 2], gt_mesh_bounds[1, 2]])

    vol1 = (mesh_bounds[1, 0] - mesh_bounds[0, 0]) * (
        mesh_bounds[1, 1] - mesh_bounds[0, 1]) * (mesh_bounds[1, 2] -
                                                  mesh_bounds[0, 2])
    vol2 = (gt_mesh_bounds[1, 0] - gt_mesh_bounds[0, 0]) * (
        gt_mesh_bounds[1, 1] - gt_mesh_bounds[0, 1]) * (gt_mesh_bounds[1, 2] -
                                                        gt_mesh_bounds[0, 2])
    inter_vol = np.max([0, xx2 - xx1]) * np.max([0, yy2 - yy1]) * np.max(
        [0, zz2 - zz1])

    iou = inter_vol / (vol1 + vol2 - inter_vol + 1e-11)
    return iou

def compute_normal_consistency(normals_src, mesh_tgt, src2tgt_idx):
    normals_src = normals_src / np.linalg.norm(
        normals_src, axis=-1, keepdims=True)
    normals_tgt = mesh_tgt.face_normals[src2tgt_idx]
    normals_tgt = normals_tgt / np.linalg.norm(
        normals_tgt, axis=-1, keepdims=True)

    src2tgt_normals_dot_product = (normals_tgt * normals_src).sum(axis=-1)
    src2tgt_normals_dot_product = np.abs(src2tgt_normals_dot_product)
    src2tgt_normals_dot_product[np.isnan(src2tgt_normals_dot_product)] = 1.

    return src2tgt_normals_dot_product


def compute_p2f_distance(points_src, mesh_tgt):
    _, src_tgt_dist, src2tgt_idx = trimesh.proximity.closest_point(
        mesh_tgt, points_src)
    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    return src_tgt_dist, src2tgt_idx


def find_hand_face_points(points_src, normals_src, points_smplx,
                          smplx_label_vector):
    # find points belong to each hand and face in scan mesh with registered SMPLX mesh
    points_src = torch.tensor(points_src).cuda().unsqueeze(0)
    normals_src = torch.tensor(normals_src).cuda().unsqueeze(0)
    points_smplx = torch.tensor(points_smplx).cuda().unsqueeze(0)

    _, close_id_src = kaolin.metrics.pointcloud.sided_distance(
        points_src, points_smplx)
    src_label_vector = smplx_label_vector[close_id_src[0]]
    src_lhand_ids = torch.where(src_label_vector == 1)[0]
    src_rhand_ids = torch.where(src_label_vector == 2)[0]
    src_face_ids = torch.where(src_label_vector == 3)[0]
    
    lhand_points_src = torch.gather(points_src[0], 0,
                                    src_lhand_ids.unsqueeze(-1).expand(-1, 3))
    lhand_normals_src = torch.gather(normals_src[0], 0,
                                     src_lhand_ids.unsqueeze(-1).expand(-1, 3))
    rhand_points_src = torch.gather(points_src[0], 0,
                                    src_rhand_ids.unsqueeze(-1).expand(-1, 3))
    rhand_normals_src = torch.gather(normals_src[0], 0,
                                     src_rhand_ids.unsqueeze(-1).expand(-1, 3))
    face_points_src = torch.gather(points_src[0], 0,
                                   src_face_ids.unsqueeze(-1).expand(-1, 3))
    face_normals_src = torch.gather(normals_src[0], 0,
                                    src_face_ids.unsqueeze(-1).expand(-1, 3))
    
    lhand_points_src = lhand_points_src.detach().cpu().numpy()
    lhand_normals_src = lhand_normals_src.detach().cpu().numpy()
    rhand_points_src = rhand_points_src.detach().cpu().numpy()
    rhand_normals_src = rhand_normals_src.detach().cpu().numpy()
    face_points_src = face_points_src.detach().cpu().numpy()
    face_normals_src = face_normals_src.detach().cpu().numpy()
    
    return lhand_points_src, rhand_points_src, lhand_normals_src, rhand_normals_src, \
        face_points_src, face_normals_src

def compute_iou_w_points(points, gt_points):
    mesh_bounds = np.concatenate([
        np.min(points, axis=0, keepdims=True),
        np.max(points, axis=0, keepdims=True)
    ],
                                 axis=0)
    gt_mesh_bounds = np.concatenate([
        np.min(gt_points, axis=0, keepdims=True),
        np.max(gt_points, axis=0, keepdims=True)
    ],
                                    axis=0)
    xx1 = np.max([mesh_bounds[0, 0], gt_mesh_bounds[0, 0]])
    yy1 = np.max([mesh_bounds[0, 1], gt_mesh_bounds[0, 1]])
    zz1 = np.max([mesh_bounds[0, 2], gt_mesh_bounds[0, 2]])

    xx2 = np.min([mesh_bounds[1, 0], gt_mesh_bounds[1, 0]])
    yy2 = np.min([mesh_bounds[1, 1], gt_mesh_bounds[1, 1]])
    zz2 = np.min([mesh_bounds[1, 2], gt_mesh_bounds[1, 2]])

    vol1 = (mesh_bounds[1, 0] - mesh_bounds[0, 0]) * (
        mesh_bounds[1, 1] - mesh_bounds[0, 1]) * (mesh_bounds[1, 2] -
                                                  mesh_bounds[0, 2])
    vol2 = (gt_mesh_bounds[1, 0] - gt_mesh_bounds[0, 0]) * (
        gt_mesh_bounds[1, 1] - gt_mesh_bounds[0, 1]) * (gt_mesh_bounds[1, 2] -
                                                        gt_mesh_bounds[0, 2])
    inter_vol = np.max([0, xx2 - xx1]) * np.max([0, yy2 - yy1]) * np.max(
        [0, zz2 - zz1])

    iou = inter_vol / (vol1 + vol2 - inter_vol + 1e-11)
    return iou

def evaluate_per_frame(mesh_src,
                       mesh_tgt,
                       mesh_smplx,
                       lhand_ids,
                       rhand_ids,
                       face_ids,
                       num_samples=10000):
    # load source mesh, target mesh, and smplx mesh
    points_src, faces_src = trimesh.sample.sample_surface(
        mesh_src, num_samples)
    normals_src = mesh_src.face_normals[faces_src]
    points_tgt, faces_tgt = trimesh.sample.sample_surface(
        mesh_tgt, num_samples)
    normals_tgt = mesh_tgt.face_normals[faces_tgt]
    points_smplx = mesh_smplx.vertices

    # compute global chamfer distance (whole body)
    src_tgt_dist, src2tgt_idx = compute_p2f_distance(points_src, mesh_tgt)
    tgt_src_dist, tgt2src_idx = compute_p2f_distance(points_tgt, mesh_src)
    global_CD_mean = (src_tgt_dist.mean() + tgt_src_dist.mean()) / 2
    global_CD_max = (src_tgt_dist.max() + tgt_src_dist.max()) / 2

    # compute global normal consistency (whole body)
    src_tgt_NC = compute_normal_consistency(normals_src, mesh_tgt, src2tgt_idx)
    tgt_src_NC = compute_normal_consistency(normals_tgt, mesh_src, tgt2src_idx)
    global_NC_mean = (src_tgt_NC.mean() + tgt_src_NC.mean()) / 2

    # compute global IOU (whole body)
    global_iou = compute_iou_w_mesh(mesh_src, mesh_tgt)

    # load SMPLX part label vector
    smplx_label_vector = torch.zeros(points_smplx.shape[0],
                                     dtype=torch.int32).cuda()
    smplx_label_vector[lhand_ids] = 1
    smplx_label_vector[rhand_ids] = 2
    smplx_label_vector[face_ids] = 3

    # find hand points in source mesh with registered SMPLX mesh
    lhand_points_src, rhand_points_src, lhand_normals_src, rhand_normals_src, face_points_src, face_normals_src = \
        find_hand_face_points(points_src, normals_src, points_smplx, smplx_label_vector)

    # find hand points in target mesh with registered SMPLX mesh
    lhand_points_tgt, rhand_points_tgt, lhand_normals_tgt, rhand_normals_tgt, face_points_tgt, face_normals_tgt = \
        find_hand_face_points(points_tgt, normals_tgt, points_smplx, smplx_label_vector)

    # concatenate left and right hand points
    hand_points_src = np.concatenate([lhand_points_src, rhand_points_src],
                                     axis=0)
    hand_normals_src = np.concatenate([lhand_normals_src, rhand_normals_src],
                                      axis=0)
    hand_points_tgt = np.concatenate([lhand_points_tgt, rhand_points_tgt],
                                     axis=0)
    hand_normals_tgt = np.concatenate([lhand_normals_tgt, rhand_normals_tgt],
                                      axis=0)

    # compute local chamfer distance (hands)
    hand_src_tgt_dist, hand_src2tgt_idx = compute_p2f_distance(
        hand_points_src, mesh_tgt)
    hand_tgt_src_dist, hand_tgt2src_idx = compute_p2f_distance(
        hand_points_tgt, mesh_src)
    local_hand_CD_mean = (hand_src_tgt_dist.mean() +
                          hand_tgt_src_dist.mean()) / 2
    local_hand_CD_max = (hand_src_tgt_dist.max() + hand_tgt_src_dist.max()) / 2

    # compute local chamfer distance (face)
    face_src_tgt_dist, face_src2tgt_idx = compute_p2f_distance(
        face_points_src, mesh_tgt)
    face_tgt_src_dist, face_tgt2src_idx = compute_p2f_distance(
        face_points_tgt, mesh_src)
    local_face_CD_mean = (face_src_tgt_dist.mean() +
                          face_tgt_src_dist.mean()) / 2
    local_face_CD_max = (face_src_tgt_dist.max() + face_tgt_src_dist.max()) / 2

    # compute local normal consistency (hands)
    hand_src_tgt_NC = compute_normal_consistency(hand_normals_src, mesh_tgt,
                                                 hand_src2tgt_idx)
    hand_tgt_src_NC = compute_normal_consistency(hand_normals_tgt, mesh_src,
                                                 hand_tgt2src_idx)
    local_hand_NC_mean = (hand_src_tgt_NC.mean() + hand_tgt_src_NC.mean()) / 2

    # compute local normal consistency (face)
    face_src_tgt_NC = compute_normal_consistency(face_normals_src, mesh_tgt,
                                                 face_src2tgt_idx)
    face_tgt_src_NC = compute_normal_consistency(face_normals_tgt, mesh_src,
                                                 face_tgt2src_idx)
    local_face_NC_mean = (face_src_tgt_NC.mean() + face_tgt_src_NC.mean()) / 2

    # compute local IOU (hands)
    lhand_iou = compute_iou_w_points(lhand_points_src, lhand_points_tgt)
    rhand_iou = compute_iou_w_points(rhand_points_src, rhand_points_tgt)
    local_hand_iou = (lhand_iou + rhand_iou) / 2

    # compute local IOU (face)
    local_face_iou = compute_iou_w_points(face_points_src, face_points_tgt)

    return global_CD_mean, global_CD_max, global_NC_mean, global_iou, \
        local_hand_CD_mean, local_hand_CD_max, local_hand_NC_mean, local_hand_iou, \
        local_face_CD_mean, local_face_CD_max, local_face_NC_mean, local_face_iou



def main(config, subject):
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_type = config['data_type']
    dataset_path = config['dataset_path']
    soft_blend = config['soft_blend']
    smpl_seg = np.load(config['smpl_seg'], allow_pickle=True).item()
    face_index = smpl_seg['face_index']
    random_lengths = smpl_seg['random_lengths']
    faces_part = smpl_seg['faces_part']
    num_sample = len(face_index)
    use_smplx = config['use_smplx']
    hierarchical = config['hierarchical_softmax']
    model_type = 'smplx' if use_smplx else 'smpl'
    if use_smplx:
        dout = 59 if hierarchical else 55
    else:
        dout = 25 if hierarchical else 24
    estimate_normals = config['estimate_normals']
    neighborhood_size = config['neighborhood_size']
    if estimate_normals:
        delta_out = 3
    else:
        delta_out = 6
    depth = config['depth']
    width = config['width']
    cond_dim = config['cond_dim']
    dim_cond_embed = config['dim_cond_embed']
    multires = config['multires']
    lbsres = config['lbsres']
    skip_layer = config['skip_layer']
    cond_layer = config['cond_layer']
    input_scale = config['input_scale']
    ms = pymeshlab.MeshSet(verbose=False)

    if data_type == 'xhumans':
        dataset = XHumans(dataset_path, subject, use_smplx, num_sample, mode='test')
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
        if use_smplx:
            smpl = SMPLX('data/SMPLX_%s.pkl'%dataset.gender.upper(),is_smplx=True,
                         use_pca=False,hand_mean=True,betas_dim=10,expr_dim=10).cuda()
            betas = torch.from_numpy(dataset.smplx_mean_shape).unsqueeze(0).cuda()
        else:
            smpl = SMPLX('data/SMPL_%s.pkl'%dataset.gender.upper(), is_smplx=False).cuda()
            betas = torch.from_numpy(dataset.smpl_mean_shape).unsqueeze(0).cuda()
        v_template = None

    print(len(dataset))

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
    
    lbs_net.load_state_dict(torch.load('trained_models/%s/lbs.pth'%subject))
    delta_net.load_state_dict(torch.load('trained_models/%s/delta.pth'%subject))
    lbs_net.eval()
    delta_net.eval()


    verts_ids = pickle.load(open('data/non_watertight_%s_vertex_labels.pkl'%dataset.gender, 'rb'), encoding='latin1')
    lhand_ids = torch.tensor(verts_ids['left_hand']).cuda()
    rhand_ids = torch.tensor(verts_ids['right_hand']).cuda()
    face_ids = torch.tensor(verts_ids['face']).cuda()

    global_CD_mean_list, global_CD_max_list, global_NC_mean_list, global_iou_list, \
        local_hand_CD_mean_list, local_hand_CD_max_list, local_hand_NC_mean_list, local_hand_iou_list, \
        local_face_CD_mean_list, local_face_CD_max_list, local_face_NC_mean_list, local_face_iou_list \
            = [], [], [], [], [], [], [], [], [], [], [], []

    os.makedirs('%s_ret'%subject, exist_ok=True)

    with torch.no_grad():
        for idx, data in tqdm(enumerate(loader)):
            if data_type == 'xhumans':
                verts, faces, category, poses, expression, sample_points, sample_normals, colors = data
                # verts = [item.cuda() for item in verts]
                # faces = [item.cuda() for item in faces]
                gt_mesh = trimesh.Trimesh(verts[0].squeeze().numpy(), faces[0].squeeze().numpy(), process=False, maintain_order=True)
                poses = poses.cuda()
                expression = expression.cuda()
                sample_points = sample_points.cuda()
                sample_normals = sample_normals.cuda()
                cond = torch.cat([poses[:,3:66]/np.pi, expression], dim=1) # b 69
                shape = torch.cat([betas,expression],dim=1)
            if os.path.exists('%s_ret/pred%d.obj'%(subject,idx)):
                pred_mesh = trimesh.load('%s_ret/pred%d.obj'%(subject,idx), process=False, maintain_order=True)
            else:
                cond_delta = delta_net(points*input_scale, cond)
                if estimate_normals:
                    trans_points = points + cond_delta
                else:
                    rot_mat = batch_rodrigues(cond_delta[...,3:].reshape(-1,3)).reshape(1,-1,3,3)
                    trans_normals = torch.einsum('bnij,bnj->bni', rot_mat, normals)
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
                if estimate_normals:
                    pred_normals = myestimate_pointcloud_normals(pred_points, neighborhood_size, disambiguate_directions=False, use_symeig_workaround=False)
                else:
                    pred_normals = torch.einsum('bnij,bnj->bni', final_T[:,:,:3,:3], trans_normals)

                tmp_points = pred_points.squeeze().cpu().numpy()
                tmp_normals = pred_normals.squeeze().cpu().numpy()
                if estimate_normals:
                    pred_mesh = psr(ms, tmp_points)
                else:
                    pred_mesh = psr(ms, tmp_points, tmp_normals)
                pred_mesh.export('%s_ret/pred%d.obj'%(subject,idx))
            gt_mesh.export('%s_ret/gt%d.obj'%(subject,idx))

            smpl_verts, _ = smpl(poses,shape,v_template)
            smpl_mesh = trimesh.Trimesh(smpl_verts.squeeze().cpu().numpy(), smpl.faces, process=False, maintain_order=True)
            smpl_mesh.export('%s_ret/smpl%d.obj'%(subject,idx))

            global_CD_mean, global_CD_max, global_NC_mean, global_iou, \
            local_hand_CD_mean, local_hand_CD_max, local_hand_NC_mean, local_hand_iou, \
            local_face_CD_mean, local_face_CD_max, local_face_NC_mean, local_face_iou = evaluate_per_frame(pred_mesh, gt_mesh, smpl_mesh, lhand_ids, rhand_ids, face_ids)

            global_CD_mean_list.append(global_CD_mean)
            global_CD_max_list.append(global_CD_max)
            global_NC_mean_list.append(global_NC_mean)
            global_iou_list.append(global_iou)
            local_hand_CD_mean_list.append(local_hand_CD_mean)
            local_hand_CD_max_list.append(local_hand_CD_max)
            local_hand_NC_mean_list.append(local_hand_NC_mean)
            local_hand_iou_list.append(local_hand_iou)
            local_face_CD_mean_list.append(local_face_CD_mean)
            local_face_CD_max_list.append(local_face_CD_max)
            local_face_NC_mean_list.append(local_face_NC_mean)
            local_face_iou_list.append(local_face_iou)

    global_CD_mean_array = np.array(global_CD_mean_list)
    global_CD_max_array = np.array(global_CD_max_list)
    global_NC_mean_array = np.array(global_NC_mean_list)
    global_iou_array = np.array(global_iou_list)
    local_hand_CD_mean_array = np.array(local_hand_CD_mean_list)
    local_hand_CD_max_array = np.array(local_hand_CD_max_list)
    local_hand_NC_mean_array = np.array(local_hand_NC_mean_list)
    local_hand_iou_array = np.array(local_hand_iou_list)
    local_face_CD_mean_array = np.array(local_face_CD_mean_list)
    local_face_CD_max_array = np.array(local_face_CD_max_list)
    local_face_NC_mean_array = np.array(local_face_NC_mean_list)
    local_face_iou_array = np.array(local_face_iou_list)

    print('global_CD_mean: {:.2f}, global_CD_max: {:.2f}, global_NC_mean: {:.3f}, global_iou: {:.3f}, \n \
        local_hand_CD_mean: {:.2f}, local_hand_CD_max: {:.2f}, local_hand_NC_mean: {:.3f}, local_hand_iou: {:.3f}, \n \
        local_face_CD_mean: {:.2f}, local_face_CD_max: {:.2f}, local_face_NC_mean: {:.3f}, local_face_iou: {:.3f}'.format(\
            global_CD_mean_array.mean()*1e3, global_CD_max_array.mean()*1e3, global_NC_mean_array.mean(), global_iou_array.mean(), \
            local_hand_CD_mean_array.mean()*1e3, local_hand_CD_max_array.mean()*1e3, local_hand_NC_mean_array.mean(), local_hand_iou_array.mean(), \
            local_face_CD_mean_array.mean()*1e3, local_face_CD_max_array.mean()*1e3, local_face_NC_mean_array.mean(), local_face_iou_array.mean()))
    

    np.savetxt('%s_CD.txt'%subject, global_CD_mean_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/xhumans.yaml')
    parser.add_argument('--subject', type=int, default=27)
    args = parser.parse_args()
    main(args.config, args.subject)