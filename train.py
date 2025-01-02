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
from kaolin.ops.mesh import index_vertices_by_faces, face_normals
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.metrics.pointcloud import chamfer_distance, sided_distance
from datasets.xhumans import XHumans, custom_collate_fn
from models.smplx import SMPLX, batch_rodrigues
from models.mlp import ImplicitNetwork
from models.deformer import hierarchical_softmax
from models.emd_module import emdFunction
from models.utils import get_face_normals, gen_sample, optimize_points, psr, myestimate_pointcloud_normals

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

    verts_color = []
    for item in faces_part:
        if item == 'leftup':
            color = [0,0,0]
        elif item == 'lefthand':
            color = [255,0,0]
        elif item == 'leftdown':
            color = [0,255,0]
        elif item == 'rightup':
            color = [0,0,255]
        elif item == 'righthand':
            color = [255,255,0]
        elif item == 'rightdown':
            color = [255,0,255]
        elif item == 'head':
            color = [0,255,255]
        elif item == 'body':
            color = [255,255,255]
        verts_color.append(color)

    verts_color = np.array(verts_color)

    if data_type == 'xhumans':
        dataset = XHumans(dataset_path, subject, use_smplx, num_sample,mode='train')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn, num_workers=8)
        if use_smplx:
            smpl = SMPLX('data/SMPLX_%s.pkl'%dataset.gender.upper(),is_smplx=True,
                         use_pca=False,hand_mean=True,betas_dim=10,expr_dim=10).cuda()
            betas = torch.from_numpy(dataset.smplx_mean_shape).unsqueeze(0).expand(batch_size,-1).cuda()

        v_template = None

        if subject == 18:
            template_idx = 809
        else:
            template_idx = 0

        verts, faces, _, poses, expression, sample_points, sample_normals, colors = dataset.__getitem__(template_idx)
        expression = torch.from_numpy(expression.astype(np.float32)).unsqueeze(0).cuda()
        poses = torch.from_numpy(poses.astype(np.float32)).unsqueeze(0).cuda()
        shape = torch.cat([betas[0:1],expression],dim=1)
        gt_category = np.load(dataset.category_list[template_idx]).astype(np.int64)
        gt_category = torch.from_numpy(gt_category).cuda()
        gt_mesh = trimesh.load(dataset.obj_list[template_idx], process=False, maintain_order=False)
        param = pickle.load(open(dataset.smplx_list[template_idx], 'rb'), encoding='latin1')
        gt_mesh.vertices -= param['transl']
        face_colors = trimesh.visual.color.vertex_to_face_color(gt_mesh.visual.to_color().vertex_colors,gt_mesh.faces)
        face_colors = (face_colors[:,:3]/255).astype(np.float32)

    print(len(dataset))

    smpl_verts,_ = smpl(poses,shape,v_template)
    _, poses_A = smpl.get_skinning(poses, shape, v_template)
    inv_poses_A = poses_A.inverse()
    smpl_weights = smpl.weights.unsqueeze(0)

    gt_mesh.export('gt.obj')
    smpl_mesh = trimesh.Trimesh(smpl_verts.squeeze().detach().cpu().numpy(), smpl.faces, process=False, maintain_order=True)
    smpl_mesh.export('smpl.obj')

    lbs_net = ImplicitNetwork(3,dout,128,5,False,multires=lbsres).cuda()
    delta_net = ImplicitNetwork(3,delta_out,width,depth, geometric_init=False, skip_layer=skip_layer, 
                cond_layer=cond_layer, cond_dim=cond_dim, dim_cond_embed=dim_cond_embed, multires=multires).cuda()

    lbs_optimizer = Adam([{'params': lbs_net.parameters(), 'lr': 0.0001}])
    delta_optimizer = Adam([{'params': delta_net.parameters(), 'lr': 0.0005}])

    lbs_scheduler = MultiStepLR(lbs_optimizer, milestones=[151], gamma=0.1)
    delta_scheduler = MultiStepLR(delta_optimizer, milestones=[151], gamma=0.1)

    # category 0 body 1 face 2 lhand 3 rhand 4 eye_mouth

    chamfer_weight = torch.Tensor([10000, 20000, 20000, 20000, 40000]).cuda()
    emd_weight = torch.Tensor([10000, 20000, 20000, 20000, 40000]).cuda()
    normal_weight = torch.Tensor([2, 4, 4, 4, 8]).cuda()

    # chamfer_weight = torch.Tensor([10000, 20000, 20000, 20000, 40000]).cuda()
    # emd_weight = torch.Tensor([10000, 20000, 20000, 20000, 40000]).cuda()
    # normal_weight = torch.Tensor([0, 0, 0, 0, 0]).cuda()

    # pretrain lbs net
    pre_optimizer = Adam(lbs_net.parameters(), lr=0.0005)
    pbar = tqdm(range(2000))
    for i in pbar:
        pred_weights = lbs_net(smpl_verts*input_scale, None) * soft_blend
        if hierarchical:
            pred_weights = hierarchical_softmax(pred_weights, model_type)
        else:
            pred_weights = F.softmax(pred_weights,-1)

        loss = 100 * F.mse_loss(smpl_weights, pred_weights)
        
        pre_optimizer.zero_grad()
        loss.backward()
        pre_optimizer.step()

        des = 'loss:%.4f'%loss.item()
        pbar.set_description(des)
    
    if estimate_normals:
        subject = '%d_estimate'%subject
    else:
        subject = '%d'%subject

    os.makedirs('trained_models/%s'%subject, exist_ok=True)
    writer = SummaryWriter('logs/%s'%subject)
    Debug = False
    start_add = False
    global_idx = 0
    pbar = tqdm(range(101))
    for i in pbar:
        for idx, data in enumerate(loader):
            if data_type == 'xhumans':
                verts, faces, category, poses, expression, sample_points, sample_normals, colors = data
                # verts = [item.cuda() for item in verts]
                # faces = [item.cuda() for item in faces]
                poses = poses.cuda()
                expression = expression.cuda()
                sample_points = sample_points.cuda()
                sample_normals = sample_normals.cuda()
                cond = torch.cat([poses[:,3:66]/np.pi, expression], dim=1) # b 69
                shape = torch.cat([betas,expression],dim=1)

            if global_idx == 100:
                start_add = True

            points,face_idx = trimesh.sample.sample_surface_even(gt_mesh, num_sample)
            normals = gt_mesh.face_normals[face_idx]
            category2 = gt_category[face_idx].unsqueeze(0)
            points = torch.from_numpy(points.astype(np.float32)).unsqueeze(0).cuda()
            normals = torch.from_numpy(normals.astype(np.float32)).unsqueeze(0).cuda()
            canc_points = points.expand(batch_size,-1,-1)
            canc_normals = normals.expand(batch_size,-1,-1)

            cond_delta = delta_net(canc_points*input_scale, cond)
            if estimate_normals:
                trans_points = canc_points + cond_delta
            else:
                rot_mat = batch_rodrigues(cond_delta[...,3:].reshape(-1,3)).reshape(batch_size,-1,3,3)
                trans_points = canc_points + cond_delta[...,:3]
                trans_normals = torch.einsum('bnij,bnj->bni', rot_mat, canc_normals)

            if start_add:
                pred_weights = lbs_net(trans_points*input_scale, None) * soft_blend
            else:
                pred_weights = lbs_net(canc_points*input_scale, None) * soft_blend
            if hierarchical:
                pred_weights = hierarchical_softmax(pred_weights, model_type)
            else:
                pred_weights = F.softmax(pred_weights,-1)

            _, A = smpl.get_skinning(poses, shape, v_template)
        
            # T = torch.einsum('bnd,bdij->bnij', pred_weights, A)
            # poses_T = torch.einsum('bnd,bdij->bnij', pred_weights, poses_A.expand(batch_size,-1,-1,-1))
            # inv_poseT = poses_T.inverse()
            # final_T = torch.einsum('bnij,bnjk->bnik', T, inv_poseT)

            final_A = torch.einsum('bnij,bnjk->bnik', A, inv_poses_A.expand(batch_size,-1,-1,-1))
            final_T = torch.einsum('bnd,bdij->bnij', pred_weights, final_A)

            if not start_add:
                pred_points = torch.einsum('bnij,bnj->bni', final_T[:,:,:3,:3], canc_points) + final_T[:,:,:3,3]
                if estimate_normals:
                    pred_normals = myestimate_pointcloud_normals(pred_points, neighborhood_size, disambiguate_directions=False, use_symeig_workaround=False)
                else:
                    pred_normals = torch.einsum('bnij,bnj->bni', final_T[:,:,:3,:3], canc_normals)
                final_T = final_T.detach()

            pred_points2 = torch.einsum('bnij,bnj->bni', final_T[:,:,:3,:3], trans_points) + final_T[:,:,:3,3]
            if estimate_normals:
                pred_normals2 = myestimate_pointcloud_normals(pred_points2, neighborhood_size, disambiguate_directions=False, use_symeig_workaround=False)
            else:
                pred_normals2 = torch.einsum('bnij,bnj->bni', final_T[:,:,:3,:3], trans_normals)

            if start_add:
                tmp_points = pred_points2
                tmp_normals = pred_normals2
            else:
                tmp_points = pred_points
                tmp_normals = pred_normals

            if Debug:
                smpl_final = torch.einsum('bnd,bdij->bnij', smpl_weights, final_A)
                pred_smpl = torch.einsum('bnij,bnj->bni', smpl_final[:,:,:3,:3], smpl_verts.expand(batch_size,-1,-1)) + smpl_final[:,:,:3,3]
                tmp_mesh = trimesh.Trimesh(pred_smpl[0].squeeze().detach().cpu().numpy(),smpl.faces,process=False, maintain_order=True)
                tmp_mesh.export('pred.obj')
                np.savetxt('pred.xyz', tmp_points[0].squeeze().detach().cpu().numpy())
                np.savetxt('gt.xyz', sample_points[0].squeeze().detach().cpu().numpy())
                pdb.set_trace()

            emd_dis, assignment = emd(tmp_points, sample_points, 0.002, 50)
            emd_loss = (emd_weight[category2] * emd_dis).mean()
            gt_normals = torch.gather(sample_normals, dim=1, index=assignment.long().unsqueeze(-1).expand(-1,-1,3))
            if estimate_normals:
                normal_loss = (normal_weight[category2] * (1 - F.cosine_similarity(tmp_normals, gt_normals, dim=2).abs())).mean()
            else:
                normal_loss = (normal_weight[category2] * (1 - F.cosine_similarity(tmp_normals, gt_normals, dim=2))).mean()
            dis1, _ = sided_distance(sample_points, tmp_points)
            dis2, _ = sided_distance(tmp_points, sample_points)
            dis1 = (chamfer_weight[category] * dis1).mean()
            dis2 = (chamfer_weight[category2] * dis2).mean()
            chamfer_loss = dis1 + dis2

            if start_add:
                delta_loss = torch.zeros_like(emd_loss)
            else:
                emd_dis2, assignment2 = emd(pred_points2, sample_points, 0.002, 50)
                emd_loss2 = (emd_weight[category2] * emd_dis2).mean()
                gt_normals2 = torch.gather(sample_normals, dim=1, index=assignment2.long().unsqueeze(-1).expand(-1,-1,3))
                if estimate_normals:
                    normal_loss2 = (normal_weight[category2] * (1 - F.cosine_similarity(pred_normals2, gt_normals2, dim=2).abs())).mean()
                else:
                    normal_loss2 = (normal_weight[category2] * (1 - F.cosine_similarity(pred_normals2, gt_normals2, dim=2))).mean()
                sid_dis1, _ = sided_distance(sample_points, pred_points2)
                sid_dis2, _ = sided_distance(pred_points2, sample_points)
                sid_dis1 = (chamfer_weight[category] * sid_dis1).mean()
                sid_dis2 = (chamfer_weight[category2] * sid_dis2).mean()
                chamfer_loss2 = sid_dis1 + sid_dis2
                delta_loss = chamfer_loss2 + emd_loss2 + normal_loss2

            pred_smpl_weights = lbs_net(smpl_verts*input_scale, None) * soft_blend
            if hierarchical:
                pred_smpl_weights = hierarchical_softmax(pred_smpl_weights, model_type)
            else:
                pred_smpl_weights = F.softmax(pred_smpl_weights,-1)

            # 100 10
            smpl_loss = 100 * F.mse_loss(smpl_weights, pred_smpl_weights)
            reg_loss = 10 * F.mse_loss(cond_delta[...,:3], torch.zeros_like(cond_delta[...,:3]))

            loss = chamfer_loss + emd_loss + delta_loss + smpl_loss + normal_loss + reg_loss

            lbs_optimizer.zero_grad()
            delta_optimizer.zero_grad()
            loss.backward()
            lbs_optimizer.step()
            delta_optimizer.step()

            des = 'chamfer:%.4f'%chamfer_loss.item() + ' emd:%.4f'%emd_loss.item() + ' delta:%.4f'%delta_loss.item()
            pbar.set_description(des)

            writer.add_scalar('chamfer_loss', chamfer_loss.item(), global_idx)
            writer.add_scalar('emd_loss', emd_loss.item(), global_idx)
            writer.add_scalar('delta_loss', delta_loss.item(), global_idx)
            writer.add_scalar('reg_loss', reg_loss.item(), global_idx)
            writer.add_scalar('smpl_loss', smpl_loss.item(), global_idx)
            writer.add_scalar('normal_loss', normal_loss.item(), global_idx)
            global_idx += 1

        if i % 10 == 0:
            gt_points,_ = trimesh.sample.sample_surface_even(gt_mesh, num_sample)
            gt_points = torch.from_numpy(gt_points.astype(np.float32)).cuda()
            points, _ = gen_sample(smpl_mesh, face_index, random_lengths)
            points = torch.from_numpy(points.astype(np.float32)).cuda()
            points = optimize_points(points, gt_points, emd)
            _,_,face_idx = trimesh.proximity.closest_point(gt_mesh, points.detach().cpu().numpy())
            normals = torch.from_numpy(gt_mesh.face_normals[face_idx].astype(np.float32)).unsqueeze(0).cuda()
            points_color = torch.from_numpy(face_colors[face_idx]).unsqueeze(0).cuda()
            points = points.unsqueeze(0)
            torch.save(lbs_net.state_dict(), 'trained_models/%s/lbs%d.pth'%(subject,i))
            torch.save(delta_net.state_dict(), 'trained_models/%s/delta%d.pth'%(subject,i))
            torch.save({'points': points, 'normals': normals, 'poses_A': poses_A, 'inv_poses_A': inv_poses_A, 'color': points_color}, 'trained_models/%s/sap%d.pth'%(subject,i))
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
                tmp_mesh.export('pred%d.obj'%i)

                np.savetxt('pred%d.txt'%i, np.concatenate([tmp_points, verts_color], axis=1))

                tmp_gt = verts[0].squeeze().detach().cpu().numpy()
                tmp_face = faces[0].squeeze().detach().cpu().numpy()
                tmp_mesh = trimesh.Trimesh(tmp_gt, tmp_face, process=False, maintain_order=True)
                tmp_mesh.export('gt%d.obj'%i)

        lbs_scheduler.step()
        delta_scheduler.step()

    torch.save(lbs_net.state_dict(), 'trained_models/%s/lbs.pth'%subject)
    torch.save(delta_net.state_dict(), 'trained_models/%s/delta.pth'%subject)
    torch.save({'points': points, 'normals': normals, 'poses_A': poses_A, 'inv_poses_A': inv_poses_A, 'color': points_color}, 'trained_models/%s/sap.pth'%subject)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/xhumans.yaml')
    parser.add_argument('--subject', type=int, default=27)
    args = parser.parse_args()
    main(args.config, args.subject)
