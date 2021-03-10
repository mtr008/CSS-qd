import numpy as np
from skimage import io, color
import mock
# import graph
# import dijkstra_path
# import dijkstra
from scipy.signal import convolve2d as conv2


def construct_graph(X):
    import graph
    h, w, c = X.shape
    G = {}
    V = list(range(h * w))
    # E = []
    Graph_X = graph.Graph(G)
    for i in range(h):
        for j in range(w):
            Graph_X.add_vertex(V[i * w + j])
            if i == 0 and j == 0:
                Graph_X.add_edge((V[0], V[1]))
                Graph_X.add_edge((V[0], V[w]))
                Graph_X.add_edge((V[0], V[w + 1]))
            elif i == 0 and j == w - 1:
                Graph_X.add_edge((V[w - 1], V[w - 2]))
                Graph_X.add_edge((V[w - 1], V[2 * w - 1]))
                Graph_X.add_edge((V[w - 1], V[2 * w - 2]))
            elif i == 0 and j != 0 and j != w - 1:
                Graph_X.add_edge((V[j], V[j - 1]))
                Graph_X.add_edge((V[j], V[j + 1]))
                Graph_X.add_edge((V[j], V[w + j]))
                Graph_X.add_edge((V[j], V[w + j - 1]))
                Graph_X.add_edge((V[j], V[w + j + 1]))
            elif i == h - 1 and j == 0:
                Graph_X.add_edge((V[h * w - w], V[h * w - w + 1]))
                Graph_X.add_edge((V[h * w - w], V[h * w - 2 * w]))
                Graph_X.add_edge((V[h * w - w], V[h * w - 2 * w + 1]))
            elif i != 0 and i != h - 1 and j == 0:
                Graph_X.add_edge((V[i * w], V[i * w + 1]))
                Graph_X.add_edge((V[i * w], V[(i - 1) * w]))
                Graph_X.add_edge((V[i * w], V[(i - 1) * w + 1]))
                Graph_X.add_edge((V[i * w], V[(i + 1) * w]))
                Graph_X.add_edge((V[i * w], V[(i + 1) * w + 1]))
            elif i == h - 1 and j == w - 1:
                Graph_X.add_edge((V[h * w - 1], V[h * w - 2]))
                Graph_X.add_edge((V[h * w - 1], V[h * w - w - 1]))
                Graph_X.add_edge((V[h * w - 1], V[h * w - w - 2]))
            elif i != 0 and i != h - 1 and j == w - 1:
                Graph_X.add_edge((V[i * w + w - 1], V[i * w + w - 2]))
                Graph_X.add_edge((V[i * w + w - 1], V[(i - 1) * w + w - 1]))
                Graph_X.add_edge((V[i * w + w - 1], V[(i - 1) * w + w - 2]))
                Graph_X.add_edge((V[i * w + w - 1], V[(i + 1) * w + w - 1]))
                Graph_X.add_edge((V[i * w + w - 1], V[(i + 1) * w + w - 2]))
            elif i == h - 1 and j != 0 and j != w - 1:
                Graph_X.add_edge((V[(h - 1) * w + j], V[(h - 1) * w + j - 1]))
                Graph_X.add_edge((V[(h - 1) * w + j], V[(h - 1) * w + j + 1]))
                Graph_X.add_edge((V[(h - 1) * w + j], V[(h - 2) * w + j]))
                Graph_X.add_edge((V[(h - 1) * w + j], V[(h - 2) * w + j - 1]))
                Graph_X.add_edge((V[(h - 1) * w + j], V[(h - 2) * w + j + 1]))
            else:
                Graph_X.add_edge((V[i * w + j], V[i * w + j - 1]))
                Graph_X.add_edge((V[i * w + j], V[i * w + j + 1]))
                Graph_X.add_edge((V[i * w + j], V[(i - 1) * w + j]))
                Graph_X.add_edge((V[i * w + j], V[(i - 1) * w + j - 1]))
                Graph_X.add_edge((V[i * w + j], V[(i - 1) * w + j + 1]))
                Graph_X.add_edge((V[i * w + j], V[(i + 1) * w + j]))
                Graph_X.add_edge((V[i * w + j], V[(i + 1) * w + j - 1]))
                Graph_X.add_edge((V[i * w + j], V[(i + 1) * w + j + 1]))
            # print(Graph_X)
    return Graph_X


def construct_graph2(X):
    import graph_with_dijkstra as graph
    h, w, c = X.shape
    Graph_X = graph.Graph(h * w)
    V = list(range(h * w))
    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                Graph_X.addEdge(V[0], V[1])
                Graph_X.addEdge(V[0], V[w])
                Graph_X.addEdge(V[0], V[w + 1])
            elif i == 0 and j == w - 1:
                Graph_X.addEdge(V[w - 1], V[w - 2])
                Graph_X.addEdge(V[w - 1], V[2 * w - 1])
                Graph_X.addEdge(V[w - 1], V[2 * w - 2])
            elif i == 0 and j != 0 and j != w - 1:
                Graph_X.addEdge(V[j], V[j - 1])
                Graph_X.addEdge(V[j], V[j + 1])
                Graph_X.addEdge(V[j], V[w + j])
                Graph_X.addEdge(V[j], V[w + j - 1])
                Graph_X.addEdge(V[j], V[w + j + 1])
            elif i == h - 1 and j == 0:
                Graph_X.addEdge(V[h * w - w], V[h * w - w + 1])
                Graph_X.addEdge(V[h * w - w], V[h * w - 2 * w])
                Graph_X.addEdge(V[h * w - w], V[h * w - 2 * w + 1])
            elif i != 0 and i != h - 1 and j == 0:
                Graph_X.addEdge(V[i * w], V[i * w + 1])
                Graph_X.addEdge(V[i * w], V[(i - 1) * w])
                Graph_X.addEdge(V[i * w], V[(i - 1) * w + 1])
                Graph_X.addEdge(V[i * w], V[(i + 1) * w])
                Graph_X.addEdge(V[i * w], V[(i + 1) * w + 1])
            elif i == h - 1 and j == w - 1:
                Graph_X.addEdge(V[h * w - 1], V[h * w - 2])
                Graph_X.addEdge(V[h * w - 1], V[h * w - w - 1])
                Graph_X.addEdge(V[h * w - 1], V[h * w - w - 2])
            elif i != 0 and i != h - 1 and j == w - 1:
                Graph_X.addEdge(V[i * w + w - 1], V[i * w + w - 2])
                Graph_X.addEdge(V[i * w + w - 1], V[(i - 1) * w + w - 1])
                Graph_X.addEdge(V[i * w + w - 1], V[(i - 1) * w + w - 2])
                Graph_X.addEdge(V[i * w + w - 1], V[(i + 1) * w + w - 1])
                Graph_X.addEdge(V[i * w + w - 1], V[(i + 1) * w + w - 2])
            elif i == h - 1 and j != 0 and j != w - 1:
                Graph_X.addEdge(V[(h - 1) * w + j], V[(h - 1) * w + j - 1])
                Graph_X.addEdge(V[(h - 1) * w + j], V[(h - 1) * w + j + 1])
                Graph_X.addEdge(V[(h - 1) * w + j], V[(h - 2) * w + j])
                Graph_X.addEdge(V[(h - 1) * w + j], V[(h - 2) * w + j - 1])
                Graph_X.addEdge(V[(h - 1) * w + j], V[(h - 2) * w + j + 1])
            else:
                Graph_X.addEdge(V[i * w + j], V[i * w + j - 1])
                Graph_X.addEdge(V[i * w + j], V[i * w + j + 1])
                Graph_X.addEdge(V[i * w + j], V[(i - 1) * w + j])
                Graph_X.addEdge(V[i * w + j], V[(i - 1) * w + j - 1])
                Graph_X.addEdge(V[i * w + j], V[(i - 1) * w + j + 1])
                Graph_X.addEdge(V[i * w + j], V[(i + 1) * w + j])
                Graph_X.addEdge(V[i * w + j], V[(i + 1) * w + j - 1])
                Graph_X.addEdge(V[i * w + j], V[(i + 1) * w + j + 1])
            # print(Graph_X)
    return Graph_X


def improved_css_img(X, K, max_iter, lmda_1):
    h, w, c = X.shape
    X_lab = color.rgb2lab(X)
    h_axis = np.repeat(np.arange(h, dtype='int').reshape(-1, 1), w, 1)
    w_axis = np.repeat(np.arange(w, dtype='int').reshape(1, -1), h, 0)
    aug_X = np.dstack((X_lab * lmda_1, h_axis))
    aug_X = np.dstack((aug_X, w_axis))  # 1

    V = list(range(h * w))
    Graph_X = construct_graph2(X)  # 2
    # A=Graph_X.dijkstra(V[6])

    c_point_init = V[np.random.random_integers(0, h * w - 1)]
    C_set = []
    C_set.append(c_point_init)  # 3

    i = 1
    while i < K:  # 4
        dist_vol = np.zeros((w * h, len(C_set)))

        for k in range(len(C_set)):
            dist=q_path(Graph_X,C_set,V)
            # dist = Graph_X.dijkstra(C_set[k])
            dist_vol[:, k] = np.array(dist)

        dist_vol = np.hstack((np.array(V).reshape((-1, 1)), dist_vol))  # dimension should be [w * h, len(C_set) + 1]
        dist_vol = np.delete(dist_vol, C_set, axis=0)

        shortest_dist = np.min(dist_vol[:, 1:], axis=1)  # dimension should be [w * h - len(C_set), 1]  # 5
        chosen_prob = shortest_dist / np.sum(shortest_dist)
        c_point = int(np.random.choice(dist_vol[:, 0], p=chosen_prob))  # 6
        C_set.append(c_point)
        i += 1  # 7, 8

    dist_record = np.zeros((w * h, len(C_set)))
    for k in range(len(C_set)):
        dist = Graph_X.dijkstra(C_set[k])
        dist_record[:, k] = np.array(dist)
    dist_record = np.hstack((np.array(V).reshape((-1, 1)), dist_record))
    dist_record = np.delete(dist_record, C_set, axis=0)
    dist_idx = dist_record[:, 0]
    dist_val = dist_record[:, 1:]

    C_bar = C_set.copy()
    C_init = C_set.copy()
    delta = -1.0
    iter = 1  # 9
    while delta < 0 and iter <= max_iter:  # 10
        C_set = C_bar  # 11

        # V_set = []
        for k in range(len(C_set)):
            # tt=np.repeat(dist_record[:, k].reshape((-1,1)), len(C_set) - 2, axis=1)
            # tt1=np.delete(dist_record, k, axis=1)
            dist_logic = np.where(np.repeat(dist_val[:, k].reshape((-1, 1)), len(C_set) - 1, axis=1) <
                                  np.delete(dist_val, k, axis=1), 1, 0)
            dist_logic = np.sum(dist_logic, axis=1).reshape((-1, 1))
            mask = (dist_logic[:, 0] == len(C_set) - 1)
            masked_dist_idx = dist_idx[mask]
            ii = np.ndarray.tolist((masked_dist_idx // w).astype(int))
            jj = np.ndarray.tolist((np.mod(masked_dist_idx, w)).astype(int))
            V_set = aug_X[ii, jj, :]  # 12

            # V_set.append(aug_X[masked_dist_idx // w, np.mod(masked_dist_idx, w), :])
            c_point_tilde = np.sum(V_set, axis=0) / V_set.shape[0]  # dimension should be 1x5  # 13
            # c_point_bar = aug_X[(np.round(c_point_tilde[3])).astype(int), (np.round(c_point_tilde[4])).astype(int), :]
            c_point_bar = (np.round(c_point_tilde[3]) * w + np.round(c_point_tilde[4])).astype(int)
            C_bar[k] = c_point_bar  # 14
            # C_bar.append(c_point_bar)

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


def q_path(graph, C_set, V):
    # params = ['dist', 'visit', 'pre']
    obj = mock.Mock()
    C_set = mock.Mock(C_set)
    obj.a = 5

    for k in range(len(C_set)):
        setattr(C_set[k], 'dist', 0)
        setattr(C_set[k], 'visit', True)

    for k in range(len(V)):
        setattr(V[k], 'dist', float("inf"))
        setattr(V[k], 'visit', False)


    for k in range(len(src_set)):
        src_set[k].dist = 0
        src_set[k].visit = True

    v.dist = float("inf")
    v.visit = False

    Q = src_set.copy()

    while len(Q) != 0:
        sdf

    return q_dist


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


im = np.float32(imread(fn('inputs/lion.jpg')))



num_clusters = [10]
for num_clusters in num_clusters:
    [src_set, cluster_centers] = improved_css_img(X=im, K=num_clusters, max_iter=100, lmda_1=1)
    imsave(fn('outputs/prob1a_' + str(num_clusters) + '_centers.jpg'),
           normalize_im(create_centers_im(im.copy(), cluster_centers)))
    # out_im = improved_css_img(X=im, K=10, max_iter=100, lmda_1=1)

    # Lr = np.random.permutation(num_clusters)
    # out_im = Lr[np.int32(out_im)]
    # dimg = cm.jet(np.minimum(1, np.float32(out_im.flatten()) / float(num_clusters)))[:, 0:3]
    # dimg = dimg.reshape([out_im.shape[0], out_im.shape[1], 3])
    # imsave(fn('outputs/prob1b_' + str(num_clusters) + '.jpg'), normalize_im(dimg))

"""
for i in range(h):
    for j in range(w):
        V.append(aug_X[i, j, :])
        # V.append(np.ndarray.tolist(aug_X[i, j, :]))
        E.append([np.array([aug_X[i, j, :], aug_X[np.maximum(0, i - 1), j, :]]),
                  np.array([aug_X[i, j, :], aug_X[np.minimum(h - 1, i + 1), j, :]]),
                  np.array([aug_X[i, j, :], aug_X[np.maximum(0, i - 1), np.maximum(0, j - 1), :]]),
                  np.array([aug_X[i, j, :], aug_X[i, np.maximum(0, j - 1), :]]),
                  np.array([aug_X[i, j, :], aug_X[np.minimum(h - 1, i + 1), np.maximum(0, j - 1), :]]),
                  np.array([aug_X[i, j, :], aug_X[np.maximum(0, i - 1), np.minimum(w - 1, j + 1), :]]),
                  np.array([aug_X[i, j, :], aug_X[i, np.minimum(w - 1, j + 1), :]]),
                  np.array([aug_X[i, j, :], aug_X[np.minimum(h - 1, i + 1), np.minimum(w - 1, j + 1), :]])])  # 2
"""
