import numpy as np
from skimage import io, color
import mock
import networkx as nx
# import graph
# import dijkstra_path
# import dijkstra
from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt
import time


def add_vertices_and_edges(Graph_X, h, w, V):
    Graph_X.add_nodes_from(V)
    for i in range(h):
        for j in range(w):
            # Graph_X.add_vertex(V[i * w + j])
            if i == 0 and j == 0:
                Graph_X.add_edge(V[0], V[1])
                Graph_X.add_edge(V[0], V[w])
                Graph_X.add_edge(V[0], V[w + 1])
            elif i == 0 and j == w - 1:
                Graph_X.add_edge(V[w - 1], V[w - 2])
                Graph_X.add_edge(V[w - 1], V[2 * w - 1])
                Graph_X.add_edge(V[w - 1], V[2 * w - 2])
            elif i == 0 and j != 0 and j != w - 1:
                Graph_X.add_edge(V[j], V[j - 1])
                Graph_X.add_edge(V[j], V[j + 1])
                Graph_X.add_edge(V[j], V[w + j])
                Graph_X.add_edge(V[j], V[w + j - 1])
                Graph_X.add_edge(V[j], V[w + j + 1])
            elif i == h - 1 and j == 0:
                Graph_X.add_edge(V[h * w - w], V[h * w - w + 1])
                Graph_X.add_edge(V[h * w - w], V[h * w - 2 * w])
                Graph_X.add_edge(V[h * w - w], V[h * w - 2 * w + 1])
            elif i != 0 and i != h - 1 and j == 0:
                Graph_X.add_edge(V[i * w], V[i * w + 1])
                Graph_X.add_edge(V[i * w], V[(i - 1) * w])
                Graph_X.add_edge(V[i * w], V[(i - 1) * w + 1])
                Graph_X.add_edge(V[i * w], V[(i + 1) * w])
                Graph_X.add_edge(V[i * w], V[(i + 1) * w + 1])
            elif i == h - 1 and j == w - 1:
                Graph_X.add_edge(V[h * w - 1], V[h * w - 2])
                Graph_X.add_edge(V[h * w - 1], V[h * w - w - 1])
                Graph_X.add_edge(V[h * w - 1], V[h * w - w - 2])
            elif i != 0 and i != h - 1 and j == w - 1:
                Graph_X.add_edge(V[i * w + w - 1], V[i * w + w - 2])
                Graph_X.add_edge(V[i * w + w - 1], V[(i - 1) * w + w - 1])
                Graph_X.add_edge(V[i * w + w - 1], V[(i - 1) * w + w - 2])
                Graph_X.add_edge(V[i * w + w - 1], V[(i + 1) * w + w - 1])
                Graph_X.add_edge(V[i * w + w - 1], V[(i + 1) * w + w - 2])
            elif i == h - 1 and j != 0 and j != w - 1:
                Graph_X.add_edge(V[(h - 1) * w + j], V[(h - 1) * w + j - 1])
                Graph_X.add_edge(V[(h - 1) * w + j], V[(h - 1) * w + j + 1])
                Graph_X.add_edge(V[(h - 1) * w + j], V[(h - 2) * w + j])
                Graph_X.add_edge(V[(h - 1) * w + j], V[(h - 2) * w + j - 1])
                Graph_X.add_edge(V[(h - 1) * w + j], V[(h - 2) * w + j + 1])
            else:
                Graph_X.add_edge(V[i * w + j], V[i * w + j - 1])
                Graph_X.add_edge(V[i * w + j], V[i * w + j + 1])
                Graph_X.add_edge(V[i * w + j], V[(i - 1) * w + j])
                Graph_X.add_edge(V[i * w + j], V[(i - 1) * w + j - 1])
                Graph_X.add_edge(V[i * w + j], V[(i - 1) * w + j + 1])
                Graph_X.add_edge(V[i * w + j], V[(i + 1) * w + j])
                Graph_X.add_edge(V[i * w + j], V[(i + 1) * w + j - 1])
                Graph_X.add_edge(V[i * w + j], V[(i + 1) * w + j + 1])
    return Graph_X


def improved_css_img(X, K, max_iter, lmda_1):
    h, w, c = X.shape
    X_lab = color.rgb2lab(X)
    h_axis = np.repeat(np.arange(h, dtype='int').reshape(-1, 1), w, 1)
    w_axis = np.repeat(np.arange(w, dtype='int').reshape(1, -1), h, 0)
    aug_X = np.dstack((X_lab * lmda_1, h_axis))
    aug_X = np.dstack((aug_X, w_axis))  # 1

    V = list(range(h * w))
    Graph_X = nx.Graph()
    Graph_X = add_vertices_and_edges(Graph_X, h, w, V)

    ############################################################

    c_point_init = V[np.random.random_integers(0, h * w - 1)]
    C_set = []
    C_set.append(c_point_init)  # 3

    i = 1
    while i < K:  # 4
        dist_vol = np.zeros((w * h, len(C_set)))

        for k in range(len(C_set)):
            dist = nx.shortest_path_length(Graph_X, source=C_set[k])
            tt = list(dist.items())
            tt1 = np.array(tt)
            tt2 = tt1[tt1[:, 0].argsort()]
            dist_vol[:, k] = np.array(tt2[:, -1])

        dist_vol = np.hstack((np.array(V).reshape((-1, 1)), dist_vol))  # dimension should be [w * h, len(C_set) + 1]
        dist_vol = np.delete(dist_vol, C_set, axis=0)

        shortest_dist = np.min(dist_vol[:, 1:], axis=1)  # dimension should be [w * h - len(C_set), 1]  # 5
        chosen_prob = shortest_dist / np.sum(shortest_dist)
        c_point = int(np.random.choice(dist_vol[:, 0], p=chosen_prob))  # 6
        C_set.append(c_point)
        i += 1  # 7, 8

    dist_record = np.zeros((w * h, len(C_set)))
    for k in range(len(C_set)):
        dist = nx.shortest_path_length(Graph_X, source=C_set[k])
        tt = list(dist.items())
        tt1 = np.array(tt)
        tt2 = tt1[tt1[:, 0].argsort()]
        dist_record[:, k] = np.array(tt2[:, -1])
    dist_record = np.hstack((np.array(V).reshape((-1, 1)), dist_record))
    dist_record = np.delete(dist_record, C_set, axis=0)
    dist_idx = dist_record[:, 0]
    dist_val = dist_record[:, 1:]

    C_bar = C_set.copy()
    # C_init = C_set.copy()
    delta = -1.0
    iter = 1  # 9
    while delta < 0 and iter <= max_iter:  # 10
        C_set = C_bar  # 11

        for k in range(len(C_set)):
            dist_logic = np.where(np.repeat(dist_val[:, k].reshape((-1, 1)), len(C_set) - 1, axis=1) <
                                  np.delete(dist_val, k, axis=1), 1, 0)
            dist_logic = np.sum(dist_logic, axis=1).reshape((-1, 1))
            mask = (dist_logic[:, 0] == len(C_set) - 1)
            masked_dist_idx = dist_idx[mask]
            ii = np.ndarray.tolist((masked_dist_idx // w).astype(int))
            jj = np.ndarray.tolist((np.mod(masked_dist_idx, w)).astype(int))
            V_set = aug_X[ii, jj, :]  # 12

            c_point_tilde = np.sum(V_set, axis=0) / V_set.shape[0]  # dimension should be 1x5  # 13
            c_point_bar = (np.round(c_point_tilde[3]) * w + np.round(c_point_tilde[4])).astype(int)
            C_bar[k] = c_point_bar  # 14

        ii1 = np.ndarray.tolist((np.array(C_bar) // w).astype(int))
        jj1 = np.ndarray.tolist((np.mod(np.array(C_bar), w)).astype(int))
        ii2 = np.ndarray.tolist((np.array(C_set) // w).astype(int))
        jj2 = np.ndarray.tolist((np.mod(np.array(C_set), w)).astype(int))
        delta = np.mean(aug_X[ii1, jj1, :]) - \
                np.mean(aug_X[ii2, jj2, :])  # 15
        iter += 1  # 16, 17

    C_set_array = np.hstack((
        ((np.array(C_set) // w).astype(int)).reshape((-1, 1)),
        ((np.mod(np.array(C_set), w)).astype(int)).reshape((-1, 1))
    ))

    return C_set, C_set_array  # 18


from skimage.io import imread, imsave
from os.path import normpath as fn  # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings

warnings.filterwarnings('ignore')


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


def unit_vector(vector):
    # return np.divide(vector,np.linalg.norm(vector,axis=-1))
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


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


def manifold_slic(im, num_clusters, cluster_centers):
    h, w, c = im.shape
    lmda_1 = 0.5

    # image augmentation
    X_lab = color.rgb2lab(im)
    h_axis = np.repeat(np.arange(h, dtype='int').reshape(-1, 1), w, 1)
    w_axis = np.repeat(np.arange(w, dtype='int').reshape(1, -1), h, 0)
    aug_im = np.dstack((X_lab * lmda_1, h_axis))
    aug_im = np.dstack((aug_im, w_axis))
    # initialization
    min_dist = float("inf") * np.ones((h, w))
    L = -1 * np.ones((h, w))
    S = int(np.sqrt(h * w / num_clusters))

    # compute area
    xx = np.arange(1, h - 1)
    yy = np.arange(1, w - 1)

    lab_1 = (aug_im[1:h - 1, 1:w - 1, :3] + aug_im[:h - 2, :w - 2, :3] +
             aug_im[:h - 2, 1:w - 1, :3] + aug_im[1:h - 1, :w - 2, :3]) / 4
    lab_2 = (aug_im[1:h - 1, 1:w - 1, :3] + aug_im[1:h - 1, :w - 2, :3] +
             aug_im[2:h, :w - 2, :3] + aug_im[2:h, 1:w - 1, :3]) / 4
    lab_3 = (aug_im[1:h - 1, 1:w - 1, :3] + aug_im[2:h, 1:w - 1, :3] +
             aug_im[1:h - 1, 2:w, :3] + aug_im[2:h, 2:w, :3]) / 4
    lab_4 = (aug_im[1:h - 1, 1:w - 1, :3] + aug_im[:h - 2, 1:w - 1, :3] +
             aug_im[1:h - 1, 2:w, :3] + aug_im[:h - 2, 2:w, :3]) / 4

    euclid_2_1 = np.sqrt(np.sum(1 + (lab_2 - lab_1) ** 2, axis=-1))
    euclid_2_3 = np.sqrt(np.sum(1 + (lab_2 - lab_3) ** 2, axis=-1))
    euclid_4_3 = np.sqrt(np.sum(1 + (lab_4 - lab_3) ** 2, axis=-1))
    euclid_4_1 = np.sqrt(np.sum(1 + (lab_4 - lab_1) ** 2, axis=-1))

    phi_1 = np.dstack((lab_1, np.repeat((xx - 0.5).reshape(-1, 1), w - 2, 1)))
    phi_1 = np.dstack((phi_1, np.repeat((yy - 0.5).reshape(1, -1), h - 2, 0)))
    phi_2 = np.dstack((lab_2, np.repeat((xx + 0.5).reshape(-1, 1), w - 2, 1)))
    phi_2 = np.dstack((phi_2, np.repeat((yy - 0.5).reshape(1, -1), h - 2, 0)))
    phi_3 = np.dstack((lab_3, np.repeat((xx + 0.5).reshape(-1, 1), w - 2, 1)))
    phi_3 = np.dstack((phi_3, np.repeat((yy + 0.5).reshape(1, -1), h - 2, 0)))
    phi_4 = np.dstack((lab_4, np.repeat((xx - 0.5).reshape(-1, 1), w - 2, 1)))
    phi_4 = np.dstack((phi_4, np.repeat((yy + 0.5).reshape(1, -1), h - 2, 0)))

    phi_21 = phi_1 - phi_2
    phi_23 = phi_3 - phi_2
    phi_43 = phi_3 - phi_4
    phi_41 = phi_1 - phi_4

    angle_21_23 = np.zeros((h - 2, w - 2))
    angle_43_41 = np.zeros((h - 2, w - 2))
    for i in range(h - 2):
        for j in range(w - 2):
            angle_21_23[i, j] = angle_between(phi_21[i, j, :], phi_23[i, j, :])
            angle_43_41[i, j] = angle_between(phi_43[i, j, :], phi_41[i, j, :])

    area_123 = 0.5 * np.multiply(np.multiply(euclid_2_1, euclid_2_3), np.sin(angle_21_23))
    area_341 = 0.5 * np.multiply(np.multiply(euclid_4_3, euclid_4_1), np.sin(angle_43_41))
    Area = area_123 + area_341  # dimension should be [h-2,w-2]

    # construct local search range
    THETA = 0
    for k in range(num_clusters):
        THETA = Area[cluster_centers[k, 0] - 1, cluster_centers[k, 1] - 1]
        THETA += THETA
    THETA = 16 * THETA / num_clusters
    # Lambda = 1

    # minimization
    for k in range(num_clusters):
        # compute scaling factor Lambda
        h_start = np.maximum(0, cluster_centers[k, 0] - 1 - S)
        h_end = np.minimum(h - 2, cluster_centers[k, 0] - 1 + S)
        w_start = np.maximum(0, cluster_centers[k, 1] - 1 - S)
        w_end = np.minimum(w - 2, cluster_centers[k, 1] - 1 + S)

        Area_patch = np.sum(Area[h_start:h_end, w_start:w_end])
        Lambda = np.sqrt(THETA / Area_patch)

        # similar minimization in SLIC
        mu_k = aug_im[cluster_centers[k, 0], cluster_centers[k, 1], :]
        h_start = np.maximum(0, np.round(cluster_centers[k, 0] - Lambda * S).astype(int))
        h_end = np.minimum(h - 1, np.round(cluster_centers[k, 0] + Lambda * S).astype(int))
        w_start = np.maximum(0, np.round(cluster_centers[k, 1] - Lambda * S).astype(int))
        w_end = np.minimum(w - 1, np.round(cluster_centers[k, 1] + Lambda * S).astype(int))
        im_patch = aug_im[h_start:h_end, w_start:w_end, :]

        dist2 = np.sum(np.square(im_patch - mu_k), axis=-1)
        dist = np.sqrt(dist2)
        L[h_start:h_end, w_start:w_end] = np.where(dist < min_dist[h_start:h_end, w_start:w_end],
                                                   k, L[h_start:h_end, w_start:w_end])
        min_dist[h_start:h_end, w_start:w_end] = np.where(dist < min_dist[h_start:h_end, w_start:w_end],
                                                          dist, min_dist[h_start:h_end, w_start:w_end])

    return L


def slic(im, num_clusters, cluster_centers):
    h, w, c = im.shape
    lmda_1 = 1

    X_lab = color.rgb2lab(im)
    h_axis = np.repeat(np.arange(h, dtype='int').reshape(-1, 1), w, 1)
    w_axis = np.repeat(np.arange(w, dtype='int').reshape(1, -1), h, 0)
    aug_im = np.dstack((X_lab * lmda_1, h_axis))
    aug_im = np.dstack((aug_im, w_axis))

    # minimization
    min_dist = float("inf") * np.ones((h, w))
    L = -1 * np.ones((h, w))
    S = int(np.sqrt(h * w / num_clusters))
    for k in range(num_clusters):
        mu_k = aug_im[cluster_centers[k, 0], cluster_centers[k, 1], :]
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


def slic_adjS(im, num_clusters, cluster_centers):
    h, w, c = im.shape
    lmda_1 = 2

    X_lab = color.rgb2lab(im)
    h_axis = np.repeat(np.arange(h, dtype='int').reshape(-1, 1), w, 1)
    w_axis = np.repeat(np.arange(w, dtype='int').reshape(1, -1), h, 0)
    aug_im = np.dstack((X_lab * lmda_1, h_axis))
    aug_im = np.dstack((aug_im, w_axis))

    # minimization
    min_dist = float("inf") * np.ones((h, w))
    L = -1 * np.ones((h, w))

    ####################################################
    S = np.zeros((num_clusters, 1))
    for k in range(num_clusters):
        centers_temp = np.delete(cluster_centers, k, axis=0)
        dist = np.sqrt((centers_temp[:, 0] - cluster_centers[k, 0]) ** 2 +
                       (centers_temp[:, 1] - cluster_centers[k, 1]) ** 2)
        S[k] = np.min(dist)
    ####################################################

    for k in range(num_clusters):
        mu_k = aug_im[cluster_centers[k, 0], cluster_centers[k, 1], :]
        h_start = np.maximum(0, int(cluster_centers[k, 0] - S[k]))
        h_end = np.minimum(h - 1, int(cluster_centers[k, 0] + S[k]))
        w_start = np.maximum(0, int(cluster_centers[k, 1] - S[k]))
        w_end = np.minimum(w - 1, int(cluster_centers[k, 1] + S[k]))
        im_patch = aug_im[h_start:h_end, w_start:w_end, :]
        dist2 = np.sum(np.square(im_patch - mu_k), axis=-1)
        dist = np.sqrt(dist2)
        L[h_start:h_end, w_start:w_end] = np.where(dist < min_dist[h_start:h_end, w_start:w_end],
                                                   k, L[h_start:h_end, w_start:w_end])
        min_dist[h_start:h_end, w_start:w_end] = np.where(dist < min_dist[h_start:h_end, w_start:w_end],
                                                          dist, min_dist[h_start:h_end, w_start:w_end])

    return L


im = np.float32(imread(fn('inputs/24063.jpg')))

num_clusters = [25,49,100]
for num_clusters in num_clusters:
    start_time = time.time()
    [src_set, cluster_centers] = improved_css_img(X=im, K=num_clusters, max_iter=10, lmda_1=2)
    end_time = time.time()
    print("time elapse:")
    print(end_time - start_time)
    imsave(fn('outputs/dijkstra/dijkstra_' + str(num_clusters) + '_centers.jpg'),
           normalize_im(create_centers_im(im.copy(), cluster_centers)))
    out_im = slic_adjS(im, num_clusters, cluster_centers)

    border_im = np.ones_like(out_im)
    gg = get_gradients(out_im)
    gg2 = get_gradients(gg)
    # border_im = np.where(gg != 0, 0, 1)
    border_im = np.where(np.logical_and(gg != 0, gg <= 4 * (num_clusters - 1) * np.sqrt(2)), 0, 1)
    imsave(fn('outputs/dijkstra/dijkstra_' + str(num_clusters) + '_border.jpg'), border_im)

    Lr = np.random.permutation(num_clusters)
    out_im = Lr[np.int32(out_im)]
    dimg = cm.jet(np.minimum(1, np.float32(out_im.flatten()) / float(num_clusters)))[:, 0:3]
    dimg = dimg.reshape([out_im.shape[0], out_im.shape[1], 3])
    imsave(fn('outputs/dijkstra/dijkstra_' + str(num_clusters) + '.jpg'), normalize_im(dimg))
