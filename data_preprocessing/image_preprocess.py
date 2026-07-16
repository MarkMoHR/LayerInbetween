import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

from sklearn.cluster import KMeans


def kmeans_grouping(data, verbose=False):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    if verbose:
        print('labels', labels)
        print('centroids', centroids)

        plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, alpha=0.5)
        plt.title('KMeans Clustering')
        plt.xlabel('Data')
        plt.ylabel('Cluster')
        plt.show()

    return labels, centroids


def binarize(img):
    img = np.array(img, dtype=np.uint8)  # (H, W), [0-stroke, 255-BG]

    img_flatten = img.flatten().astype(np.float32)
    zeros = np.zeros_like(img_flatten)
    img_flatten_in = np.stack([img_flatten, zeros], axis=1)
    group_labels, group_centroids = kmeans_grouping(img_flatten_in, verbose=False)

    # img_a = img_flatten[group_labels == 0]
    # img_b = img_flatten[group_labels == 1]

    # img_a_min, img_a_max = np.min(img_a), np.max(img_a)
    # img_b_min, img_b_max = np.min(img_b), np.max(img_b)

    # if group_centroids[0][0] > group_centroids[1][0]:
    #     bin_threshold1 = (img_a_min + img_b_max) / 2.0
    # else:
    #     bin_threshold1 = (img_a_max + img_b_min) / 2.0

    bin_threshold2 = (group_centroids[0][0] + group_centroids[1][0]) / 2.0

    # print('bin_threshold1:', bin_threshold1)
    # print('bin_threshold2:', bin_threshold2)

    img[img > bin_threshold2] = 255  # (H, W), [0-stroke, 255-BG], uint8
    return img


def darken(img, dark_scaling=1.5):
    img = np.array(img, dtype=np.float32)  # (H, W), [0-stroke, 255-BG]

    img_darker = 255.0 - np.clip((255.0 - img) * dark_scaling, 0.0, 255.0)
    img_darker = img_darker.astype(np.uint8)  # (H, W), [0-stroke, 255-BG], uint8
    return img_darker


def squaring(img):
    height, width = img.shape[0], img.shape[1]

    max_dim = max(height, width)

    pad_top = (max_dim - height) // 2
    pad_down = max_dim - height - pad_top
    pad_left = (max_dim - width) // 2
    pad_right = max_dim - width - pad_left

    img_p = np.pad(img, ((pad_top, pad_down), (pad_left, pad_right)), 'constant', constant_values=255)
    return img_p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base', default="test_examples", type=str, help="data base path")
    parser.add_argument('--image_id', type=int, help='image ID for evaluation')
    args = parser.parse_args()

    img_base = os.path.join(args.data_base, 'raster_black')
    rough_base = os.path.join(img_base, 'rough_raw')

    for image_type in ['_ref', '_tar']:
        image_name = str(args.image_id) + image_type + '.png'
        img_path = os.path.join(rough_base, image_name)

        img = np.array(Image.open(img_path).convert('L'))
        img = binarize(img)
        img = darken(img)
        img = squaring(img)

        img = Image.fromarray(img, 'L')
        img.save(os.path.join(img_base, image_name))
