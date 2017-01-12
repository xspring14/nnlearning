import numpy as np
import scipy.io as sci

class Prefetcher(object):

	def __init__(self, mat_dir, mat_list, batch_size, max_len=150):

		lines = file(mat_list).readlines()
		self.n_samples = len(lines)
		self.max_len = max_len
		print("loading {} samples.".format(self.n_samples))

		self.mat_list = []
		self.label = np.empty((self.n_samples,), dtype="int32")
		for i, line in enumerate(lines):
			line = line.rstrip().split()
			self.mat_list.append(line[0])
			self.label[i] = int(line[1])

		self.cur = 0
		self.mat_dir = mat_dir
		self.batch_size = batch_size
		self.idx = range(self.n_samples)

	def next_batch(self):

		x_batches = np.zeros((self.batch_size, self.max_len, 2), dtype="float32")
		y_batches = np.empty((self.batch_size,), dtype="int32")
		l_batches = np.zeros((self.batch_size, self.max_len), dtype="float32")
		cur = 0;
		while(cur < self.batch_size):
			if self.cur >= self.n_samples:
				self.cur = 0
				self.idx = np.random.permutation(self.n_samples)
			points = sci.loadmat(self.mat_dir + self.mat_list[self.idx[self.cur]])['points']
			points = np.asarray(points, dtype="float32")
			l = points.shape[0]
			x_batches[cur, :l, :] = points;
			l_batches[cur, :l] = 1.0
			y_batches[cur] = self.label[self.idx[self.cur]]

			cur += 1
			self.cur += 1
		return x_batches, l_batches, y_batches

if __name__ == '__main__':

    imgs_dir = "/home/xu/data/"
    imgs_list = "/home/xu/data/val100.txt"
    prefetcher = Prefetcher(imgs_dir, imgs_list, 64, 150)
    x_batches, l_batches, y_batches = prefetcher.next_batch()

    print x_batches.shape
    print l_batches.shape
    print y_batches.shape
