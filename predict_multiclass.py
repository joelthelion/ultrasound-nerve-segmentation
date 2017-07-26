#!/usr/bin/env python3
""" Use weights.h5 to apply the model without training it """

import os
import scipy.misc
import numpy as np
from keras.utils import np_utils
from keras import backend as K

from data import load_train_data, load_test_data
from train_multiclass import get_unet

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 96
img_cols = 96

def preprocess(imgs):
    imgs_p = imgs[..., np.newaxis]
    return imgs_p


def predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train, imgs_label_train = load_train_data()
    # imgs_train = imgs_train[:32]
    # imgs_label_train = imgs_label_train[:32]

    imgs_train = preprocess(imgs_train)
    imgs_label_train = preprocess(imgs_label_train)
    imgs_label_train = np_utils.to_categorical(imgs_label_train)
    imgs_label_train = imgs_label_train.reshape((280,96,96,4))

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_mask_test, imgs_label_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)
    imgs_label_test = preprocess(imgs_label_test)
    imgs_label_test = np_utils.to_categorical(imgs_label_test)
    imgs_label_test = imgs_label_test.reshape((70,96,96,4))

    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = get_unet()
    # model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting labels on test data...')
    print('-'*30)
    predicted = model.predict(imgs_test, verbose=1)
    np.save('predicted.npy', predicted)

    print('-' * 30)
    print('Saving predicted labels to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(predicted.argmax(axis=3)[...,np.newaxis], imgs_id_test):
        image = (image[:, :, 0]).astype(np.uint8)
        scipy.misc.imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)
        # scipy.misc.toimage(image, cmin=0, cmax=255).save(os.path.join(pred_dir, str(image_id) + '_pred.png'))

if __name__ == '__main__':
    predict()
