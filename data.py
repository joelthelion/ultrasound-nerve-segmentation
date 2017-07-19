from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread

data_path = '/mnt/VirtualBox/project_joel/data_joel/png/ED/'

image_rows = 96
image_cols = 96


def create_data(images, suffix):
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    print('-'*30)
    print('Creating %sing images...'% suffix)
    print('-'*30)
    for i, image_name in enumerate(images):
        image_mask_name = image_name.split('_')[0] + '_4CH_segmentation_ED.png'
        img = imread(os.path.join(data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(data_path, image_mask_name), as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
    print('Loading done.')

    np.save('imgs_%s.npy'%suffix, imgs)
    np.save('imgs_mask_%s.npy'%suffix, imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    images = os.glob("*4CH_ED.png")
    train_n = len(images) * 0.8
    create_data(images[:train_n], 'train')
    create_data(images[train_n:], 'test')
