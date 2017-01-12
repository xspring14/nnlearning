#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import os
import cv2


class Prefetcher():
    def __init__(self, img_list_path, imgs_dir, batch_size, offset=None):
        self.imgs_dirs = imgs_dir
        self.img_list = []
        self.labels = []
        with open(img_list_path, 'rb') as f:
            l = f.readlines()
        print("loaded {} samples from {}".format(len(l), img_list_path))

        self.chars = [str(x) for x in range(10)]
        for r in l:
            fields = r.strip().split(' ')
            self.img_list.append(fields[0])
            self.labels.append(int(fields[1]))
        self.batch_size = batch_size
        self.n_samples = len(self.img_list)
        # self.idxs = np.random.permutation(self.n_samples)
        self.idxs = range(self.n_samples)
        if offset is not None:
            self.cur = offset % self.n_samples
        else:
            self.cur = 0
        self.n_classes = len(self.chars)


    def next_batch(self):
        features = []
        labels = []
        # load into memory
        while len(features) < self.batch_size:
            if self.cur >= self.n_samples:
                self.cur = 0
                self.idxs = np.random.permutation(self.n_samples)
            img_path = self.img_list[self.idxs[self.cur]]
            full_img_path = os.path.join(self.imgs_dirs, img_path)
            im = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
            features.append(im.reshape(28, 28, 1)/255.)
            # features.append(im.reshape(784) / 255.)
            # label = np.zeros(10, dtype="int32")
            # label[self.labels[self.idxs[self.cur]]] = 1
            # labels.append(label)
            labels.append(self.labels[self.idxs[self.cur]])
            self.cur += 1
        # values = self._wrap(features, labels, self.batch_size, self.stride, self.patch_width, self.n_classes, self.is_shared, is_blank_y)
        values = [np.asarray(features).astype("float32"), np.asarray(labels).astype("int32")]
        return values


    """
    def _wrap(self, features, labels, batch_size, stride, patch_width, n_classes, is_shared, is_blank_y):
        # packing
        x_max_len = np.max([x.shape[1] for x in features])
        y_max_len = 50 # pre-difine
        height = features[0].shape[0]

        # transform
        x_max_len = np.ceil(x_max_len * 1. / stride)
        height = height * np.sum(patch_width)
        print("[prefetch]height: {}, x_max_step:{}, y_max_width:{}".format(height, x_max_len, y_max_len))

        # x and x_mask
        x = np.zeros((batch_size, 1, height, x_max_len)). astype(config.floatX)
        x_mask = np.zeros((batch_size, x_max_len)).astype(config.floatX)
        for i, xx in enumerate(features):
            shape = xx.shape
            l = int(np.ceil(xx.shape[1] * 1. / stride))
            for j in range(l):
                long_vec = []
                base = j * stride
                for patch in patch_width:
                    vec = np.zeros(shape[0] * patch).astype(config.floatX)
                    vec2 = xx[:, base:base+patch].T.flatten()
                    vec[:len(vec2)] = vec2
                    long_vec = np.concatenate([long_vec, vec])
                assert len(long_vec) == height
                x[i, :, :, j] = long_vec
            x_mask[i, :l] = 1.0

        # y and y_clip
        y = np.zeros((batch_size, y_max_len)).astype('int32')
        y_clip = np.zeros((batch_size)).astype('int32')
        if is_blank_y:
            for i, yy in enumerate(labels):
                y_extend = np.ones(2 * len(yy) + 1, dtype='int32') * n_classes
                for j in range(len(yy)):
                    y_extend[2 * j + 1] = yy[j]
                y[i, :len(y_extend)] = y_extend
                y_clip[i] = len(y_extend)
        else:
            for i, yy in enumerate(labels):
                y[i, :len(yy)] = yy;
                y_clip[i] = len(yy)

        values = [x, x_mask, y, y_clip]

        # is shared
        if is_shared:
            values = [theano.shared(value) for value in values]

        return values

    """
