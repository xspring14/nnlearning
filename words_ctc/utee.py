#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import os
import cv2


class Prefetcher():
    def __init__(self, imgs_dir, img_list_path, batch_size, max_length, offset = None):
        self.imgs_dirs = imgs_dir
        self.img_list = []
        self.labels = []
        self.max_length=max_length

        with open(img_list_path, 'rb') as f:
            l = f.readlines()
        print("loaded {} samples from {}".format(len(l) - 1, img_list_path))
        chars_from, chars_to = l[0].split(' ')
        self.chars = [chr(x) for x in range(int(chars_from), int(chars_to))]
        # print("class's size is {}".format(len(self.chars)))
        for r in l[1:]:
            fields = r.strip().split(' ')
            self.img_list.append(fields[0])
            self.labels.append([int(c) for c in fields[1:]])
        self.batch_size = batch_size
        self.n_samples = len(self.img_list)
        self.idxs = range(self.n_samples)
        if offset is not None:
            self.cur = offset % self.n_samples
        else:
            self.cur = 0

        self.n_classes = len(self.chars)


    def next_batch(self):
        feats=[]
        feat_lens = []
        target_vals = []
        target_indices = []

        # load into memory
        batch_cnts=0
        while batch_cnts < self.batch_size:
            if self.cur >= self.n_samples:
                self.cur = 0
                self.idxs = np.random.permutation(self.n_samples)
            img_path = self.img_list[self.idxs[self.cur]]
            full_img_path = os.path.join(self.imgs_dirs, img_path)
            im = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)

            feats.append(im/255.0)
            labels=self.labels[self.idxs[self.cur]]
            target_vals.extend(labels)
            target_indices.extend([[batch_cnts, i] for i in range(len(labels))])

            self.cur += 1
            batch_cnts += 1

        x = np.zeros((self.batch_size, self.max_length, feats[0].shape[0])).astype(np.float32)

        for i, feat in enumerate(feats):
            feat_lens.append(feat.shape[1])
            x[i,:feat_lens[-1],:]=feat.T

        values = [x, np.asarray(feat_lens).astype("int32"), np.asarray(target_vals).astype("int32"),
                  np.asarray(target_indices).astype("int64")]
        return values



"""
if __name__=="__main__":

    input_max_lenth=200

    data_dir="/home/mai/workspace/github/cnn-lstm-ctc/dataset/english_sentence/split_tiny_images"
    train_list='/home/mai/workspace/github/cnn-lstm-ctc/dataset/english_sentence/train_img_list.txt'
    val_list='/home/mai/workspace/github/cnn-lstm-ctc/dataset/english_sentence/val_img_list.txt'

    max_len=0

    with open(train_list,"r") as f:
        lines=f.readlines()

    for l in lines[1:]:
        l=l.strip().split(' ')
        full_img_path= os.path.join(data_dir, l[0])

        im=cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)

        if max_len<im.shape[1]:
            max_len=im.shape[1]

    with open(val_list,"r") as f:
        lines=f.readlines()

    for l in lines[1:]:
        l=l.strip().split(' ')
        full_img_path= os.path.join(data_dir, l[0])

        im=cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)

        if max_len<im.shape[1]:
            max_len=im.shape[1]

    print max_len



    train_prefetcher=Prefetcher(data_dir, train_list, 64, input_max_lenth)
    val_prefetcher=Prefetcher(data_dir, val_list, 64, input_max_lenth)

    x_val, x_len, y_val, y_idx=train_prefetcher.next_batch()

    print x_val.shape
    print x_len.shape
    print y_val.shape
    print y_idx.shape

"""