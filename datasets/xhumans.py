import os
from os.path import join
from glob import glob
import numpy as np
import trimesh
from trimesh.points import remove_close
from trimesh.visual import uv_to_interpolated_color
import pickle
import torch
from torch.utils.data import Dataset


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



def custom_collate_fn(batch):
    '''
    :param batch B N
    '''
    batch_size = len(batch)
    length = len(batch[0])
    ret = []
    vertices = [torch.from_numpy(item[0]).unsqueeze(0) for item in batch]
    faces = [torch.from_numpy(item[1]) for item in batch]
    ret.append(vertices)
    ret.append(faces)
    for i in range(2,length):
        data = [torch.from_numpy(item[i]) for item in batch]
        ret.append(torch.stack(data,dim=0))

    return ret

# remove maintain_order=True to keep uv right
class XHumans(Dataset):
    def __init__(self, data_path, subject, use_smplx=True, num_sample=40960, mode='train'):
        data_path = join(data_path,'%05d'%subject)
        with open(join(data_path,'gender.txt'), 'r') as f:
            self.gender = f.readline().strip()

        self.use_smplx = use_smplx
        self.num_sample = num_sample
        self.obj_list = sorted(glob(join(data_path,'%s/*/meshes_obj/*.obj'%mode)))
        self.category_list = sorted(glob(join(data_path,'%s/*/meshes_category/*.npy'%mode)))
        self.smplx_list = sorted(glob(join(data_path,'%s/*/SMPLX/*.pkl'%mode)))
        self.smpl_list = sorted(glob(join(data_path, '%s/*/SMPL/*.pkl'%mode)))
        self.smpl_mean_shape = np.load(join(data_path, 'mean_shape_smpl.npy'))
        self.smplx_mean_shape = np.load(join(data_path,'mean_shape_smplx.npy'))

    def __len__(self):
        return len(self.obj_list)
    
    def __getitem__(self, index):
        mesh = trimesh.load(self.obj_list[index], process=False, maintain_order=False)
        category = np.load(self.category_list[index])
        if self.use_smplx:
            param = pickle.load(open(self.smplx_list[index], 'rb'), encoding='latin1')
            expression = param['expression']
            pose = np.concatenate([param['global_orient'],param['body_pose'],param['jaw_pose'],param['leye_pose'],param['reye_pose'],
                               param['left_hand_pose'],param['right_hand_pose']])
        else:
            param = pickle.load(open(self.smpl_list[index], 'rb'), encoding='latin1')
            expression = np.zeros(10).astype(np.float32)
            pose = np.concatenate([param['global_orient'], param['body_pose']])

        mesh.vertices -= param['transl']
        # points, face_idx = trimesh.sample.sample_surface_even(mesh, self.num_sample)
        points, face_idx, colors = mysample(mesh, self.num_sample)
        normals = mesh.face_normals[face_idx]
        category = category[face_idx].astype(np.int64)
        points = points.astype(np.float32)
        normals = normals.astype(np.float32)

        vertices = np.array(mesh.vertices).astype(np.float32)
        faces = np.array(mesh.faces).astype(np.int64)

        return vertices, faces, category, pose, expression, points, normals, np.array(colors[:,:3]/255.).astype(np.float32)



class XColor(Dataset):
    def __init__(self, data_path, subject_list, num_sample=40960, mode='train'):
        self.num_sample = num_sample

        self.obj_list = []
        for subject in subject_list:
            self.obj_list.extend(sorted(glob(join(data_path,'%05d/%s/*/meshes_obj/*.obj'%(subject,mode)))))

    def __len__(self):
        return len(self.obj_list)

    def __getitem__(self, index):
        mesh = trimesh.load(self.obj_list[index], process=False, maintain_order=False)

        samples, face_index, colors = mysample(mesh, self.num_sample)

        return np.array(colors[:,:3]/255.).astype(np.float32)
    




class GRAB(Dataset):
    def __init__(self, data_path, subject, use_smplx=True, num_sample=40960, mode='train'):
        data_path = join(data_path,'%05d'%subject)
        with open(join(data_path,'gender.txt'), 'r') as f:
            self.gender = f.readline().strip()

        self.use_smplx = use_smplx
        self.num_sample = num_sample
        self.obj_list = sorted(glob(join(data_path,'%s/*/meshes_obj/*.obj'%mode)))
        self.category_list = sorted(glob(join(data_path,'%s/*/meshes_category/*.npy'%mode)))
        self.smplx_list = sorted(glob(join(data_path,'%s/*/SMPLX/*.pkl'%mode)))
        self.smplx_mean_shape = np.load(join(data_path,'mean_shape_smplx.npy'))

    def __len__(self):
        return len(self.obj_list)
    
    def __getitem__(self, index):
        mesh = trimesh.load(self.obj_list[index], process=False, maintain_order=False)
        category = np.load(self.category_list[index])
        if self.use_smplx:
            param = pickle.load(open(self.smplx_list[index], 'rb'), encoding='latin1')
            expression = param['expression']
            pose = np.concatenate([param['global_orient'],param['body_pose'],param['jaw_pose'],param['leye_pose'],param['reye_pose'],
                               param['left_hand_pose'],param['right_hand_pose']])
        else:
            param = pickle.load(open(self.smpl_list[index], 'rb'), encoding='latin1')
            expression = np.zeros(10).astype(np.float32)
            pose = np.concatenate([param['global_orient'], param['body_pose']])

        mesh.vertices -= param['transl']
        points, face_idx, colors = mysample(mesh, self.num_sample)
        normals = mesh.face_normals[face_idx]
        category = category[face_idx].astype(np.int64)
        points = points.astype(np.float32)
        normals = normals.astype(np.float32)

        vertices = np.array(mesh.vertices).astype(np.float32)
        faces = np.array(mesh.faces).astype(np.int64)

        return vertices, faces, category, pose, expression, points, normals, np.array(colors[:,:3]/255.).astype(np.float32)