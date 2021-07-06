import urllib
import os
import numpy as np
import bz2
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import time
from scipy.optimize import least_squares

#read data
BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"
URL = BASE_URL + FILE_NAME

if not os.path.isfile(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)

def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d

'''
1.camera_params with shape (n_cameras, 9) contains initial estimates of parameters for all cameras. First 3 components, rotation vector, next 3 components form a translation vector, then a focal distance and two distortion parameters.
2.points_3d with shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
3.camera_ind with shape (n_observations,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
4.point_ind with shape (n_observations,) contatins indices of points (from 0 to n_points - 1) involved in each observation.
5.points_2d with shape (n_observations, 2) contains measured 2-D coordinates of points projected on images in each observations.
'''
#n_observations <=> the number of images
camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)
print('camera_params -> {}'.format(np.shape(camera_params)))
print('points_3d -> {}'.format(np.shape(points_3d)))
print('camera_indices -> {}'.format(np.shape(camera_indices)))
print('point_indices -> {}'.format(np.shape(point_indices)))
print('points_2d -> {}'.format(np.shape(points_2d)))
print('----------------------------------------------')

n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3]) #transform world points to camera frame using camera extrinsics
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis] #project them on image using intrinsics and distortion
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))                 #get camera parameters
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))                      #get 3D point locations
    points_proj = project(points_3d[point_indices], camera_params[camera_indices]) #project 3D point on image
    return (points_proj - points_2d).ravel() #compute the errors

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

#Initial unknowns (camera parameters(extrinsics + intrinsics) + 3D locations of points)
x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

#Error function to minimize
f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

plt.subplot(2, 1, 1)
plt.plot(f0, label = 'Initial error')
plt.legend()

A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
t0 = time.time()
res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
t1 = time.time()
print("Optimization took {0:.0f} seconds".format(t1 - t0))
plt.subplot(2, 1, 2)
plt.plot(res.fun, label='Final error')
plt.legend()
plt.show()
