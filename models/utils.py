import json
import math
import pdb
from typing import Optional, Tuple
from tqdm import tqdm
import numpy as np
import cv2
import skimage
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import trimesh
from trimesh.points import remove_close
from trimesh.visual import uv_to_interpolated_color
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop, SGD
from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.metrics.pointcloud import chamfer_distance
from .smplx import batch_rodrigues

def psr(ms, points, normals=None, depth=9):
    if normals is None:
        np.savetxt('tmp.xyz', points)
        ms.load_new_mesh('tmp.xyz')
        ms.compute_normals_for_point_sets()
    else:
        np.savetxt('tmp.xyz', np.concatenate([points,normals],axis=1))
        ms.load_new_mesh('tmp.xyz')
        
    ms.surface_reconstruction_screened_poisson(depth=depth)
    ms.save_current_mesh('tmp.obj')
    ms.clear()
    mesh = trimesh.load('tmp.obj', process=False, maintain_order=True)

    return mesh

def optimize_points(points, gt_points, emd, valid_idx=None):
    ori_points = points.clone()
    if valid_idx is None:
        points.requires_grad_(True)
        optimizer = Adam([points], lr=0.001)
    else:
        grad_points = points[valid_idx]
        nongrad_points = points[~valid_idx]
        grad_points.requires_grad_(True)
        optimizer = Adam([grad_points], lr=0.001)

    pbar = tqdm(range(50))
    for i in pbar:
        if valid_idx is not None:
            points = torch.cat([grad_points, nongrad_points], dim=0)
        chamfer_loss = 2000 * chamfer_distance(points.unsqueeze(0), gt_points.unsqueeze(0)).mean()
        emd_dis, assignment = emd(points.unsqueeze(0), gt_points.unsqueeze(0), 0.002, 50)
        emd_loss = 2000 * emd_dis.mean()

        reg_loss = F.mse_loss(points,ori_points)

        loss = chamfer_loss + emd_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        des = 'chamfer:%.4f'%chamfer_loss.item() + ' emd:%.4f'%emd_loss.item()
        pbar.set_description(des)
    
    if valid_idx is None:
        return points.detach()
    else:
        ret = torch.zeros_like(points)
        ret[valid_idx] = grad_points.detach()
        ret[~valid_idx] = nongrad_points
        return ret

def gen_newmesh(ms, mesh, num_sample=40960,depth=9):
    points, face_idx = trimesh.sample.sample_surface_even(mesh, num_sample)
    normals = mesh.face_normals[face_idx]
    np.savetxt('tmp.xyz', np.concatenate([points,normals],axis=1))
    ms.load_new_mesh('tmp.xyz')
    ms.surface_reconstruction_screened_poisson(depth=depth)
    # ms.simplification_quadric_edge_collapse_decimation(targetfacenum=40000)
    ms.save_current_mesh('tmp.obj')
    ms.clear()
    mesh = trimesh.load('tmp.obj', process=False, maintain_order=True)
    return mesh

def gen_sample(mesh, face_index, random_lengths):
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    samples = sample_vector + tri_origins

    return samples, mesh.face_normals[face_index]

def split_verts(file_name):
    verts_seg = json.load(open(file_name,'r'))

    verts_part = dict()
    if 'eyeballs' in verts_seg.keys():
        verts_part['ignore'] = list(set(verts_seg['leftEye'] + verts_seg['rightEye'] + verts_seg['eyeballs']))
        missed = [8811, 8812,8813,8814,9161,9165] # some points missed in smplx_verts_seg
    else:
        verts_part['ignore'] = []
        missed = []
    verts_part['leftup'] = list(set(verts_seg['leftArm'] + verts_seg['leftShoulder'] + verts_seg['leftForeArm']))
    verts_part['lefthand'] = list(set(verts_seg['leftHand'] + verts_seg['leftHandIndex1']))
    verts_part['leftdown'] = list(set(verts_seg['leftLeg'] + verts_seg['leftToeBase'] + verts_seg['leftFoot'] + verts_seg['leftUpLeg']))
    verts_part['rightup'] = list(set(verts_seg['rightArm'] + verts_seg['rightShoulder'] + verts_seg['rightForeArm']))
    verts_part['righthand'] = list(set(verts_seg['rightHand'] + verts_seg['rightHandIndex1']))
    verts_part['rightdown'] = list(set(verts_seg['rightLeg'] + verts_seg['rightToeBase'] + verts_seg['rightFoot'] + verts_seg['rightUpLeg']))
    verts_part['head'] = list(set(verts_seg['head'] + verts_seg['neck'] + missed))
    verts_part['body'] = list(set(verts_seg['spine1'] + verts_seg['spine2'] + verts_seg['spine'] + verts_seg['hips']))

    part_verts = dict()
    for key in verts_part.keys():
        for item in verts_part[key]:
            part_verts[item] = key

    return verts_part, part_verts

def init_smpl_sample(mesh, json_file, count):

    verts_part, part_verts = split_verts(json_file)

    part_faces = dict()
    for i in range(len(mesh.faces)):
        a,b,c = mesh.faces[i]
        if part_verts[a] == part_verts[b] == part_verts[c]:
            part_faces[i] = part_verts[a]
        elif (part_verts[a] == part_verts[b]):
            part_faces[i] = part_verts[a]
        elif (part_verts[a] == part_verts[c]):
            part_faces[i] = part_verts[a]
        elif (part_verts[b] == part_verts[c]):
            part_faces[i] = part_verts[b]
        else:
            print(part_verts[a])
            print(part_verts[b])
            print(part_verts[c])
            raise NotImplementedError

    radius = np.sqrt(mesh.area / (4 * count))

    face_weight = mesh.area_faces.copy()
    for i in range(len(mesh.faces)):
        face = mesh.faces[i]
        if face[0] in verts_part['ignore'] or face[1] in verts_part['ignore'] or face[2] in verts_part['ignore']:
            face_weight[i] = 0

    weight_cum = np.cumsum(face_weight)
    face_pick = np.random.random(count*4) * weight_cum[-1]
    face_index = np.searchsorted(weight_cum, face_pick)

    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    random_lengths = np.random.random((len(tri_vectors), 2, 1))
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    samples = sample_vector + tri_origins

    points, mask = remove_close(samples, radius)

    assert len(points) >= count

    face_index = face_index[mask][:count]
    random_lengths = random_lengths[mask][:count]

    faces_part = []
    for face in face_index:
        faces_part.append(part_faces[face])

    return face_index, random_lengths, faces_part


def sample_surface_wncolor(mesh, count):
    """
    Sample the surface of a mesh, returning the specified
    number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Parameters
    ---------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    """

    # len(mesh.faces) float, array of the areas
    # of each face of the mesh
    area = mesh.area_faces.copy()
        
    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)
    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # do the same for normal
    normals = mesh.vertex_normals.view(np.ndarray)[mesh.faces]
    nml_origins = normals[:, 0]
    nml_vectors = normals[:, 1:]#.copy()
    nml_vectors -= np.tile(nml_origins, (1, 2)).reshape((-1, 2, 3))

    colors = mesh.visual.vertex_colors[:,:3].astype(np.float32)
    colors = colors / 255.0
    colors = colors.view(np.ndarray)[mesh.faces]
    clr_origins = colors[:, 0]
    clr_vectors = colors[:, 1:]#.copy()
    clr_vectors -= np.tile(clr_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # pull the vectors for the faces we are going to sample from
    nml_origins = nml_origins[face_index]
    nml_vectors = nml_vectors[face_index]

    clr_origins = clr_origins[face_index]
    clr_vectors = clr_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    sample_normal = (nml_vectors * random_lengths).sum(axis=1)
    sample_color = (clr_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    normals = sample_normal + nml_origins

    colors = sample_color + clr_origins

    return samples, face_index, colors


def mysample(mesh, count, sample_color=True, face_weight=None):
    """
    Sample the surface of a mesh, returning the specified
    number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    """

    # len(mesh.faces) float, array of the areas
    # of each face of the mesh
    if face_weight is None:
        face_weight = mesh.area_faces
    # cumulative area (len(mesh.faces))
    weight_cum = np.cumsum(face_weight)
    face_pick = np.random.random(count) * weight_cum[-1]
    face_index = np.searchsorted(weight_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    if sample_color and hasattr(mesh.visual, 'uv'):
        uv_origins = mesh.visual.uv[mesh.faces[:, 0]]
        uv_vectors = mesh.visual.uv[mesh.faces[:, 1:]].copy()
        uv_origins_tile = np.tile(uv_origins, (1, 2)).reshape((-1, 2, 2))
        uv_vectors -= uv_origins_tile
        uv_origins = uv_origins[face_index]
        uv_vectors = uv_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    if sample_color:
        if hasattr(mesh.visual, 'uv'):
            sample_uv_vector = (uv_vectors * random_lengths).sum(axis=1)
            uv_samples = sample_uv_vector + uv_origins
            texture = mesh.visual.material.image
            colors = uv_to_interpolated_color(uv_samples, texture)
        else:
            colors = mesh.visual.face_colors[face_index]

    return samples, face_index, colors


def mysample_even(mesh, count):
    radius = np.sqrt(mesh.area / (3 * count))

    points, index, albedo = mysample(mesh, count * 3)

    # remove the points closer than radius
    points, mask = remove_close(points, radius)

    # we got all the samples we expect
    if len(points) >= count:
        return points[:count], index[mask][:count], albedo[mask][:count]
    
    raise NotImplementedError

def meshcleaning(file_name):

    mesh = trimesh.load(file_name, process=False, maintain_order=True)
    cc = mesh.split(only_watertight=False)    

    out_mesh = cc[0]
    bbox = out_mesh.bounds
    area = (bbox[1,0] - bbox[0,0]) * (bbox[1,1] - bbox[0,1])
    for c in cc:
        bbox = c.bounds
        if area < (bbox[1,0] - bbox[0,0]) * (bbox[1,1] - bbox[0,1]):
            area = (bbox[1,0] - bbox[0,0]) * (bbox[1,1] - bbox[0,1])
            out_mesh = c
    
    out_mesh.export(file_name)


def load_K_Rt_from_P(P):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    # c2w
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    # convert to w2c
    pose = np.linalg.inv(pose)

    return intrinsics, pose

def export_mesh(verts, faces, file_name):
    if isinstance(verts, torch.Tensor):
        verts = verts.squeeze().detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.squeeze().detach().cpu().numpy()

    mesh = trimesh.Trimesh(verts,faces, process=False, maintain_order=True)
    mesh.export(file_name)

def inv_softmax(x, inf=1e10):
    out = torch.log(x)
    isinf = torch.isinf(out)
    out[isinf] = -inf

    return out

def get_normals(vertices, faces):
    '''
    vertices b n 3
    faces f 3
    '''
    verts_normals = torch.zeros_like(vertices)

    vertices_faces = vertices[:, faces] # b f 3 3

    verts_normals.index_add_(
        1,
        faces[:, 1],
        torch.cross(
            vertices_faces[:, :, 2] - vertices_faces[:, :, 1],
            vertices_faces[:, :, 0] - vertices_faces[:, :, 1],
            dim=2,
        ),
    )
    verts_normals.index_add_(
        1,
        faces[:, 2],
        torch.cross(
            vertices_faces[:, :, 0] - vertices_faces[:, :, 2],
            vertices_faces[:, :, 1] - vertices_faces[:, :, 2],
            dim=2,
        ),
    )
    verts_normals.index_add_(
        1,
        faces[:, 0],
        torch.cross(
            vertices_faces[:, :, 1] - vertices_faces[:, :, 0],
            vertices_faces[:, :, 2] - vertices_faces[:, :, 0],
            dim=2,
        ),
    )

    verts_normals = F.normalize(verts_normals, p=2, dim=2)

    return verts_normals

def get_face_normals(vertices, faces):
    '''
    vertices b n 3
    faces f 3
    '''
    vertices_faces = vertices[:, faces] # b f 3 3
    face_normals = torch.cross(vertices_faces[:, :, 1] - vertices_faces[:, :, 0], vertices_faces[:, :, 2] - vertices_faces[:, :, 0], dim=2)
    face_normals = F.normalize(face_normals, p=2, dim=2)

    return face_normals

def laplacian_cot(verts, faces):
    '''
    verts n 3
    faces f 3
    '''
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces] # f 3 3
    v0, v1, v2 = face_verts[:,0], face_verts[:,1], face_verts[:,2]

    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    s = 0.5 * (A + B + C)
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot = cot / 4.0

    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]

    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    L = L + L.t()

    idx = faces.view(-1)
    inv_areas = torch.zeros(V, dtype=torch.float32, device=verts.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    inv_areas.scatter_add_(0, idx, val)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(-1, 1)

    return L, inv_areas

def get_edges(verts, faces):
    V = verts.shape[0]
    f = faces.shape[0]
    device = verts.device
    v0, v1, v2 = faces.chunk(3, dim=1)
    e01 = torch.cat([v0, v1], dim=1)  # (sum(F_n), 2)
    e12 = torch.cat([v1, v2], dim=1)  # (sum(F_n), 2)
    e20 = torch.cat([v2, v0], dim=1)  # (sum(F_n), 2)

    edges = torch.cat([e12, e20, e01], dim=0)  # (sum(F_n)*3, 2)
    edges, _ = edges.sort(dim=1)
    edges_hash = V * edges[:, 0] + edges[:, 1]
    u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
    sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
    unique_mask = torch.ones(edges_hash.shape[0], dtype=torch.bool, device=device)
    unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
    edges = torch.stack([u // V, u % V], dim=1)

    faces_to_edges = inverse_idxs.reshape(3, f).t()

    return edges, faces_to_edges

def compute_laplacian(verts, faces):
    # first compute edges

    V = verts.shape[0]
    device = verts.device

    edges, faces_to_edges = get_edges(verts, faces)

    e0, e1 = edges.unbind(1)
    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L

def laplacian_smoothing(verts, faces, method="uniform"):
    weights = 1.0 / verts.shape[0]

    with torch.no_grad():
        if method == "uniform":
            L = compute_laplacian(verts, faces)
        else:
            L, inv_areas = laplacian_cot(verts, faces)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas

    if method == "uniform":
        loss = L.mm(verts)
    elif method == "cot":
        loss = L.mm(verts) * norm_w - verts
    else:
        loss = (L.mm(verts) - L_sum * verts) * norm_w
    
    loss = loss.norm(dim=1)

    loss = loss * weights
    return loss.sum()


def get_radiance(coeff, normal, degree=3):
    '''
    coeff 9 or n 9
    normal n 3
    '''

    radiance = coeff[...,0]
    if degree > 1:
        radiance = radiance + coeff[...,1] * normal[:,1]
        radiance = radiance + coeff[...,2] * normal[:,2]
        radiance = radiance + coeff[...,3] * normal[:,0]
    if degree > 2:
        radiance = radiance + coeff[...,4] * normal[:,0] * normal[:,1]
        radiance = radiance + coeff[...,5] * normal[:,1] * normal[:,2]
        radiance = radiance + coeff[...,6] * (2*normal[:,2]*normal[:,2]-normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])
        radiance = radiance + coeff[...,7] * normal[:,2] * normal[:,0]
        radiance = radiance + coeff[...,8] * (normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])

    return radiance

def get_matrix(normal, degree=3):
    if isinstance(normal, np.ndarray):
        matrix = np.zeros((normal.shape[0], degree**2))
    elif isinstance(normal, torch.Tensor):
        matrix = torch.zeros(normal.shape[0], degree**2, device=normal.device)

    matrix[:,0] = 1
    if degree > 1:
        matrix[:,1] = normal[:,1]
        matrix[:,2] = normal[:,2]
        matrix[:,3] = normal[:,0]
    if degree > 2:
        matrix[:,4] = normal[:,0] * normal[:,1]
        matrix[:,5] = normal[:,1] * normal[:,2]
        matrix[:,6] = (2*normal[:,2]*normal[:,2]-normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])
        matrix[:,7] = normal[:,2] * normal[:,0]
        matrix[:,8] = (normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])

    return matrix


def lookat2extrinsic(origin, target, up):
    z = normalize((target-origin)[None]).reshape(-1)
    x = normalize(np.cross(up, z)[None]).reshape(-1)
    y = normalize(np.cross(z,x)[None]).reshape(-1)
    minv = np.eye(4)
    tr = np.eye(4)
    minv[0:1,:3] = x
    minv[1:2,:3] = y
    minv[2:3,:3] = z
    tr[:3,3] = -origin

    check = np.eye(4)
    check[:3,0] = x
    check[:3,1] = y
    check[:3,2] = z
    check[:3,3] = origin

    return np.matmul(minv, tr)

def extrinsic2lookat(matrix):
    origin = np.matmul(np.linalg.inv(matrix[:3,:3]), (-matrix[:3,3:])).reshape(-1)
    target = origin + matrix[2,:3]
    up = normalize(np.cross(matrix[2,:3], matrix[0,:3])[None]).reshape(-1)
    return origin, target, up





# code adapted from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/points_normals.py


def _disambiguate_vector_directions(pcl, knns, vecs):
    """
    Disambiguates normal directions according to [1].

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """
    # parse out K from the shape of knns
    K = knns.shape[2]
    # the difference between the mean of each neighborhood and
    # each element of the neighborhood
    df = knns - pcl[:, :, None]
    # projection of the difference on the principal direction
    proj = (vecs[:, :, None] * df).sum(3)
    # check how many projections are positive
    n_pos = (proj > 0).type_as(knns).sum(2, keepdim=True)
    # flip the principal directions where number of positive correlations
    flip = (n_pos < (0.5 * K)).type_as(knns)
    vecs = (1.0 - 2.0 * flip) * vecs
    return vecs



class _SymEig3x3(nn.Module):
    """
    Optimized implementation of eigenvalues and eigenvectors computation for symmetric 3x3
     matrices.

    Please see https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
     and https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    """

    def __init__(self, eps: Optional[float] = None) -> None:
        """
        Args:
            eps: epsilon to specify, if None then use torch.float eps
        """
        super().__init__()

        self.register_buffer("_identity", torch.eye(3))
        self.register_buffer("_rotation_2d", torch.tensor([[0.0, -1.0], [1.0, 0.0]]))
        self.register_buffer(
            "_rotations_3d", self._create_rotation_matrices(self._rotation_2d)
        )

        self._eps = eps or torch.finfo(torch.float).eps

    @staticmethod
    def _create_rotation_matrices(rotation_2d) -> torch.Tensor:
        """
        Compute rotations for later use in U V computation

        Args:
            rotation_2d: a π/2 rotation matrix.

        Returns:
            a (3, 3, 3) tensor containing 3 rotation matrices around each of the coordinate axes
            by π/2
        """

        rotations_3d = torch.zeros((3, 3, 3))
        rotation_axes = set(range(3))
        for rotation_axis in rotation_axes:
            rest = list(rotation_axes - {rotation_axis})
            rotations_3d[rotation_axis][rest[0], rest] = rotation_2d[0]
            rotations_3d[rotation_axis][rest[1], rest] = rotation_2d[1]

        return rotations_3d

    def forward(
        self, inputs: torch.Tensor, eigenvectors: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute eigenvalues and (optionally) eigenvectors

        Args:
            inputs: symmetric matrices with shape of (..., 3, 3)
            eigenvectors: whether should we compute only eigenvalues or eigenvectors as well

        Returns:
            Either a tuple of (eigenvalues, eigenvectors) or eigenvalues only, depending on
             given params. Eigenvalues are of shape (..., 3) and eigenvectors (..., 3, 3)
        """
        if inputs.shape[-2:] != (3, 3):
            raise ValueError("Only inputs of shape (..., 3, 3) are supported.")

        inputs_diag = inputs.diagonal(dim1=-2, dim2=-1)
        inputs_trace = inputs_diag.sum(-1)
        q = inputs_trace / 3.0

        # Calculate squared sum of elements outside the main diagonal / 2
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        p1 = ((inputs**2).sum(dim=(-1, -2)) - (inputs_diag**2).sum(-1)) / 2
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        p2 = ((inputs_diag - q[..., None]) ** 2).sum(dim=-1) + 2.0 * p1.clamp(self._eps)

        p = torch.sqrt(p2 / 6.0)
        B = (inputs - q[..., None, None] * self._identity) / p[..., None, None]

        r = torch.det(B) / 2.0
        # Keep r within (-1.0, 1.0) boundaries with a margin to prevent exploding gradients.
        r = r.clamp(-1.0 + self._eps, 1.0 - self._eps)

        phi = torch.acos(r) / 3.0
        eig1 = q + 2 * p * torch.cos(phi)
        eig2 = q + 2 * p * torch.cos(phi + 2 * math.pi / 3)
        eig3 = 3 * q - eig1 - eig2
        # eigenvals[..., i] is the i-th eigenvalue of the input, α0 ≤ α1 ≤ α2.
        eigenvals = torch.stack((eig2, eig3, eig1), dim=-1)

        # Soft dispatch between the degenerate case (diagonal A) and general.
        # diag_soft_cond -> 1.0 when p1 < 6 * eps and diag_soft_cond -> 0.0 otherwise.
        # We use 6 * eps to take into account the error accumulated during the p1 summation
        diag_soft_cond = torch.exp(-((p1 / (6 * self._eps)) ** 2)).detach()[..., None]

        # Eigenvalues are the ordered elements of main diagonal in the degenerate case
        diag_eigenvals, _ = torch.sort(inputs_diag, dim=-1)
        eigenvals = diag_soft_cond * diag_eigenvals + (1.0 - diag_soft_cond) * eigenvals

        if eigenvectors:
            eigenvecs = self._construct_eigenvecs_set(inputs, eigenvals)
        else:
            eigenvecs = None

        return eigenvals, eigenvecs

    def _construct_eigenvecs_set(
        self, inputs: torch.Tensor, eigenvals: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct orthonormal set of eigenvectors by given inputs and pre-computed eigenvalues

        Args:
            inputs: tensor of symmetric matrices of shape (..., 3, 3)
            eigenvals: tensor of pre-computed eigenvalues of of shape (..., 3, 3)

        Returns:
            Tuple of three eigenvector tensors of shape (..., 3, 3), composing an orthonormal
             set
        """
        eigenvecs_tuple_for_01 = self._construct_eigenvecs(
            inputs, eigenvals[..., 0], eigenvals[..., 1]
        )
        eigenvecs_for_01 = torch.stack(eigenvecs_tuple_for_01, dim=-1)

        eigenvecs_tuple_for_21 = self._construct_eigenvecs(
            inputs, eigenvals[..., 2], eigenvals[..., 1]
        )
        eigenvecs_for_21 = torch.stack(eigenvecs_tuple_for_21[::-1], dim=-1)

        # The result will be smooth here even if both parts of comparison
        # are close, because eigenvecs_01 and eigenvecs_21 would be mostly equal as well
        eigenvecs_cond = (
            eigenvals[..., 1] - eigenvals[..., 0]
            > eigenvals[..., 2] - eigenvals[..., 1]
        ).detach()
        eigenvecs = torch.where(
            eigenvecs_cond[..., None, None], eigenvecs_for_01, eigenvecs_for_21
        )

        return eigenvecs

    def _construct_eigenvecs(
        self, inputs: torch.Tensor, alpha0: torch.Tensor, alpha1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct an orthonormal set of eigenvectors by given pair of eigenvalues.

        Args:
            inputs: tensor of symmetric matrices of shape (..., 3, 3)
            alpha0: first eigenvalues of shape (..., 3)
            alpha1: second eigenvalues of shape (..., 3)

        Returns:
            Tuple of three eigenvector tensors of shape (..., 3, 3), composing an orthonormal
             set
        """

        # Find the eigenvector corresponding to alpha0, its eigenvalue is distinct
        ev0 = self._get_ev0(inputs - alpha0[..., None, None] * self._identity)
        u, v = self._get_uv(ev0)
        ev1 = self._get_ev1(inputs - alpha1[..., None, None] * self._identity, u, v)
        # Third eigenvector is computed as the cross-product of the other two
        ev2 = torch.cross(ev0, ev1, dim=-1)

        return ev0, ev1, ev2

    def _get_ev0(self, char_poly: torch.Tensor) -> torch.Tensor:
        """
        Construct the first normalized eigenvector given a characteristic polynomial

        Args:
            char_poly: a characteristic polynomials of the input matrices of shape (..., 3, 3)

        Returns:
            Tensor of first eigenvectors of shape (..., 3)
        """

        r01 = torch.cross(char_poly[..., 0, :], char_poly[..., 1, :], dim=-1)
        r12 = torch.cross(char_poly[..., 1, :], char_poly[..., 2, :], dim=-1)
        r02 = torch.cross(char_poly[..., 0, :], char_poly[..., 2, :], dim=-1)

        cross_products = torch.stack((r01, r12, r02), dim=-2)
        # Regularize it with + or -eps depending on the sign of the first vector
        cross_products += self._eps * self._sign_without_zero(
            cross_products[..., :1, :]
        )

        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        norms_sq = (cross_products**2).sum(dim=-1)
        max_norms_index = norms_sq.argmax(dim=-1)

        # Pick only the cross-product with highest squared norm for each input
        max_cross_products = self._gather_by_index(
            cross_products, max_norms_index[..., None, None], -2
        )
        # Pick corresponding squared norms for each cross-product
        max_norms_sq = self._gather_by_index(norms_sq, max_norms_index[..., None], -1)

        # Normalize cross-product vectors by thier norms
        return max_cross_products / torch.sqrt(max_norms_sq[..., None])

    def _gather_by_index(
        self, source: torch.Tensor, index: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """
        Selects elements from the given source tensor by provided index tensor.
        Number of dimensions should be the same for source and index tensors.

        Args:
            source: input tensor to gather from
            index: index tensor with indices to gather from source
            dim: dimension to gather across

        Returns:
            Tensor of shape same as the source with exception of specified dimension.
        """

        index_shape = list(source.shape)
        index_shape[dim] = 1

        return source.gather(dim, index.expand(index_shape)).squeeze(dim)

    def _get_uv(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes unit-length vectors U and V such that {U, V, W} is a right-handed
        orthonormal set.

        Args:
            w: eigenvector tensor of shape (..., 3)

        Returns:
            Tuple of U and V unit-length vector tensors of shape (..., 3)
        """

        min_idx = w.abs().argmin(dim=-1)
        rotation_2d = self._rotations_3d[min_idx].to(w)

        u = F.normalize((rotation_2d @ w[..., None])[..., 0], dim=-1)
        v = torch.cross(w, u, dim=-1)
        return u, v

    def _get_ev1(
        self, char_poly: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the second normalized eigenvector given a characteristic polynomial
        and U and V vectors

        Args:
            char_poly: a characteristic polynomials of the input matrices of shape (..., 3, 3)
            u: unit-length vectors from _get_uv method
            v: unit-length vectors from _get_uv method

        Returns:
            desc
        """

        j = torch.stack((u, v), dim=-1)
        m = j.transpose(-1, -2) @ char_poly @ j

        # If angle between those vectors is acute, take their sum = m[..., 0, :] + m[..., 1, :],
        # otherwise take the difference = m[..., 0, :] - m[..., 1, :]
        # m is in theory of rank 1 (or 0), so it snaps only when one of the rows is close to 0
        is_acute_sign = self._sign_without_zero(
            (m[..., 0, :] * m[..., 1, :]).sum(dim=-1)
        ).detach()

        rowspace = m[..., 0, :] + is_acute_sign[..., None] * m[..., 1, :]
        # rowspace will be near zero for second-order eigenvalues
        # this regularization guarantees abs(rowspace[0]) >= eps in a smooth'ish way
        rowspace += self._eps * self._sign_without_zero(rowspace[..., :1])

        return (
            j
            @ F.normalize(rowspace @ self._rotation_2d.to(rowspace), dim=-1)[..., None]
        )[..., 0]

    @staticmethod
    def _sign_without_zero(tensor):
        """
        Args:
            tensor: an arbitrary shaped tensor

        Returns:
            Tensor of the same shape as an input, but with 1.0 if tensor > 0.0 and -1.0
             otherwise
        """
        return 2.0 * (tensor > 0.0).to(tensor.dtype) - 1.0


def symeig3x3(
    inputs: torch.Tensor, eigenvectors: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute eigenvalues and (optionally) eigenvectors

    Args:
        inputs: symmetric matrices with shape of (..., 3, 3)
        eigenvectors: whether should we compute only eigenvalues or eigenvectors as well

    Returns:
        Either a tuple of (eigenvalues, eigenvectors) or eigenvalues only, depending on
         given params. Eigenvalues are of shape (..., 3) and eigenvectors (..., 3, 3)
    """
    return _SymEig3x3().to(inputs.device)(inputs, eigenvectors=eigenvectors)


from pytorch3d.ops.knn import knn_points

def get_point_covariances(
    points_cloud,
    neighborhood_size,
):
    """
    Computes the per-point covariance matrices by of the 3D locations of
    K-nearest neighbors of each point.

    Args:
        **points_padded**: Input point clouds as a padded tensor
            of shape `(minibatch, num_points, dim)`.
        **num_points_per_cloud**: Number of points per cloud
            of shape `(minibatch,)`.
        **neighborhood_size**: Number of nearest neighbors for each point
            used to estimate the covariance matrices.

    Returns:
        **covariances**: A batch of per-point covariance matrices
            of shape `(minibatch, dim, dim)`.
        **k_nearest_neighbors**: A batch of `neighborhood_size` nearest
            neighbors for each of the point cloud points
            of shape `(minibatch, num_points, neighborhood_size, dim)`.
    """
    # get K nearest neighbor idx for each point in the point cloud

    num_points_per_cloud = torch.ones(points_cloud.shape[0],dtype=torch.long,device=points_cloud.device) * points_cloud.shape[1]
    k_nearest_neighbors = knn_points(points_cloud,points_cloud,lengths1=num_points_per_cloud,lengths2=num_points_per_cloud,K=neighborhood_size,return_nn=True,).knn

    # obtain the mean of the neighborhood
    pt_mean = k_nearest_neighbors.mean(2, keepdim=True)
    # compute the diff of the neighborhood and the mean of the neighborhood
    central_diff = k_nearest_neighbors - pt_mean
    # per-nn-point covariances
    per_pt_cov = central_diff.unsqueeze(4) * central_diff.unsqueeze(3)
    # per-point covariances
    covariances = per_pt_cov.mean(2)

    return covariances, k_nearest_neighbors



def myestimate_pointcloud_normals(pointclouds, neighborhood_size=10, disambiguate_directions=True, use_symeig_workaround=True):
    '''
    pointclouds b n 3
    '''
    num_points = pointclouds.shape[1]
    pcl_mean = pointclouds.mean(1)
    points_centered = pointclouds - pcl_mean[:, None, :]

    cov, knns = get_point_covariances(points_centered, neighborhood_size)

    if use_symeig_workaround:
        curvatures, local_coord_frames = symeig3x3(cov, eigenvectors=True)
    else:
        curvatures, local_coord_frames = torch.linalg.eigh(cov)


    if disambiguate_directions:
        # disambiguate normal
        n = _disambiguate_vector_directions(
            points_centered, knns, local_coord_frames[:, :, :, 0]
        )
        # disambiguate the main curvature
        z = _disambiguate_vector_directions(
            points_centered, knns, local_coord_frames[:, :, :, 2]
        )
        # the secondary curvature is just a cross between n and z
        y = torch.cross(n, z, dim=2)
        # cat to form the set of principal directions
        local_coord_frames = torch.stack((n, y, z), dim=3)


    normals = local_coord_frames[:, :, :, 0]

    return normals