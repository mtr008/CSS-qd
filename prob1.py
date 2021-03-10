## Default modules imported. Import more if you need to.
### Problem designed by Abby Stylianou

import numpy as np
from scipy.signal import convolve2d as conv2


def get_cluster_centers(im, num_clusters):
    # Implement a method that returns an initial grid of cluster centers. You should first
    # create a grid of evenly spaced centers (hint: np.meshgrid), and then use the method
    # discussed in class to make sure no centers are initialized on a sharp boundary.
    # You can use the get_gradients method from the support code below.

    # cluster_centers = np.zeros((num_clusters, 2), dtype='int')

    """ YOUR CODE GOES HERE """
    H, W = im.shape[0], im.shape[1]
    S = np.sqrt(H * W / num_clusters)
    h = np.arange(S / 2, H + S / 2, S, dtype='int')
    w = np.arange(S / 2, W + S / 2, S, dtype='int')

    # evenly guess
    hh, ww = np.meshgrid(h, w)
    hh_refined = hh.copy()
    ww_refined = ww.copy()
    cluster_centers_init = np.hstack((hh.reshape((-1, 1)), ww.reshape((-1, 1))))

    # refine the guess
    grad_map = get_gradients(im)
    # grad_at_cluster = grad_map[hh, ww]

    [hh_refined, ww_refined] = np.where(grad_map[hh - 1, ww - 1] < grad_map[hh, ww],
                                        [hh - 1, ww - 1], [hh_refined, ww_refined])
    [hh_refined, ww_refined] = np.where(grad_map[hh - 1, ww] < grad_map[hh, ww],
                                        [hh - 1, ww], [hh_refined, ww_refined])
    [hh_refined, ww_refined] = np.where(grad_map[hh - 1, ww + 1] < grad_map[hh, ww],
                                        [hh - 1, ww + 1], [hh_refined, ww_refined])
    [hh_refined, ww_refined] = np.where(grad_map[hh, ww - 1] < grad_map[hh, ww],
                                        [hh, ww - 1], [hh_refined, ww_refined])
    [hh_refined, ww_refined] = np.where(grad_map[hh, ww + 1] < grad_map[hh, ww],
                                        [hh, ww + 1], [hh_refined, ww_refined])
    [hh_refined, ww_refined] = np.where(grad_map[hh + 1, ww - 1] < grad_map[hh, ww],
                                        [hh + 1, ww - 1], [hh_refined, ww_refined])
    [hh_refined, ww_refined] = np.where(grad_map[hh + 1, ww] < grad_map[hh, ww],
                                        [hh + 1, ww], [hh_refined, ww_refined])
    [hh_refined, ww_refined] = np.where(grad_map[hh + 1, ww + 1] < grad_map[hh, ww],
                                        [hh + 1, ww + 1], [hh_refined, ww_refined])
    cluster_centers = np.hstack((hh_refined.reshape((-1, 1)), ww_refined.reshape((-1, 1))))

    return cluster_centers


def slic(im, num_clusters, cluster_centers):
    # Implement the slic function such that all pixels assigned to a label
    # should be close to each other in squared distance of augmented vectors.
    # You can weight the color and spatial components of the augmented vectors
    # differently. To do this, experiment with different values of spatial_weight.
    h, w, c = im.shape
    # clusters = np.zeros((h, w))

    """ YOUR CODE GOES HERE """
    # spatial_weight
    alpha = 2.5

    # initialization
    min_dist = float("inf") * np.ones((h, w))
    L = -1 * np.ones((h, w))
    S = int(np.sqrt(h * w / num_clusters))
    h_axis = alpha * np.repeat(np.arange(h, dtype='int').reshape(-1, 1), w, 1)
    w_axis = alpha * np.repeat(np.arange(w, dtype='int').reshape(1, -1), h, 0)
    aug_im = np.dstack((im, h_axis))
    aug_im = np.dstack((aug_im, w_axis))

    # minimization
    for k in range(num_clusters):
        color_val = im[cluster_centers[k, 0], cluster_centers[k, 1], :]
        mu_k = np.array([color_val[0], color_val[1], color_val[2],
                         alpha * cluster_centers[k, 0], alpha * cluster_centers[k, 1]])
        h_start = np.maximum(0, cluster_centers[k, 0] - S)
        h_end = np.minimum(h - 1, cluster_centers[k, 0] + S)
        w_start = np.maximum(0, cluster_centers[k, 1] - S)
        w_end = np.minimum(w - 1, cluster_centers[k, 1] + S)
        im_patch = aug_im[h_start:h_end, w_start:w_end, :]
        dist2 = np.sum(np.square(im_patch - mu_k), axis=-1)
        dist = np.sqrt(dist2)
        L[h_start:h_end, w_start:w_end] = np.where(dist < min_dist[h_start:h_end, w_start:w_end],
                                                   k, L[h_start:h_end, w_start:w_end])
        min_dist[h_start:h_end, w_start:w_end] = np.where(dist < min_dist[h_start:h_end, w_start:w_end],
                                                          dist, min_dist[h_start:h_end, w_start:w_end])

    return L


########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn  # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings

warnings.filterwarnings('ignore')


# Use get_gradients (code from pset1) to get the gradient of your image when initializing your cluster centers.
def get_gradients(im):
    if len(im.shape) > 2:
        im = np.mean(im, axis=2)
    df = np.float32([[1, 0, -1]])
    sf = np.float32([[1, 2, 1]])
    gx = conv2(im, sf.T, 'same', 'symm')
    gx = conv2(gx, df, 'same', 'symm')
    gy = conv2(im, sf, 'same', 'symm')
    gy = conv2(gy, df.T, 'same', 'symm')
    return np.sqrt(gx * gx + gy * gy)


# normalize_im normalizes our output to be between 0 and 1
def normalize_im(im):
    im += np.abs(np.min(im))
    im /= np.max(im)
    return im


# create an output image of our cluster centers
def create_centers_im(im, centers):
    for center in centers:
        im[center[0] - 2:center[0] + 2, center[1] - 2:center[1] + 2] = [255., 0., 255.]
    return im


im = np.float32(imread(fn('inputs/24063.jpg')))

num_clusters = [25]
for num_clusters in num_clusters:
    cluster_centers = get_cluster_centers(im, num_clusters)
    imsave(fn('outputs/prob1a_' + str(num_clusters) + '_centers.jpg'),
           normalize_im(create_centers_im(im.copy(), cluster_centers)))
    out_im = slic(im, num_clusters, cluster_centers)

    Lr = np.random.permutation(num_clusters)
    out_im = Lr[np.int32(out_im)]
    dimg = cm.jet(np.minimum(1, np.float32(out_im.flatten()) / float(num_clusters)))[:, 0:3]
    dimg = dimg.reshape([out_im.shape[0], out_im.shape[1], 3])
    imsave(fn('outputs/prob1b_' + str(num_clusters) + '.jpg'), normalize_im(dimg))
