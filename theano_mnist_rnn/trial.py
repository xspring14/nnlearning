# coding:utf-8
import theano
import theano.tensor as T
import numpy as np
from PIL import Image
from trial_model import Gru
from trial_optimizer import get_optimizer

def load_train_data():
	data = np.empty((60000,28,28),dtype="float32")
	label = np.empty((60000,),dtype="int32")

	imgs_list=file('/home/xu/data/mnist/train/total.txt').readlines()
	# imgs = os.listdir("./mnist")
	num = len(imgs_list)
	for i in range(num):
		line=imgs_list[i]
		line=line.rstrip().split()
		img = Image.open("/home/xu/data/"+line[0])
		arr = np.asarray(img,dtype="float32")
		data[i, ...] = arr
		label[i] = int(line[1])
        data /= 255
	return data,label

def load_test_data():
	data = np.empty((10000,28,28),dtype="float32")
	label = np.empty((10000,),dtype="int32")

	imgs_list=file('/home/xu/data/mnist/test/test.txt').readlines()
	# imgs = os.listdir("./mnist")
	num = len(imgs_list)
	for i in range(num):
		line=imgs_list[i]
		line=line.rstrip().split()
		img = Image.open("/home/xu/data/"+line[0])
		arr = np.asarray(img,dtype="float32")
		data[i, ...] = arr
		label[i] = int(line[1])
        data /= 255
	return data,label	

def train(n_h=200, optimizer='rmsprop', learning_rate=0.001, n_epochs=1, batch_size=64):

	n_x = 28 # for input_dim
	n_o = 10 # for output_dim, the class_num
	print("build actual model...")
	# index = T.iscalar('index')
	x = T.tensor3('x')
	y = T.ivector('y')

	model = Gru(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_o)
	cost = model.cross_entropy(y)
	acc  = model.accuracy(y)
	updates = get_optimizer(optimizer, cost, model.params, learning_rate)

	print("forming model...")
	train_model = theano.function(inputs=[x, y],
		                          outputs=cost,
		                          updates=updates
		                          )

	test_model = theano.function(inputs=[x, y], outputs=acc)

	
	print("loading data...")
	train_set_x, train_set_y = load_train_data()
	n_train_batches = train_set_x.shape[0] / batch_size
	test_set_x, test_set_y = load_test_data()
	n_test_batches = test_set_x.shape[0] / batch_size


	print("training...")
	epoch = 0
	while(epoch < n_epochs):

		epoch += 1
		for i in xrange(n_train_batches):

			x_batch = train_set_x[i * batch_size : (i + 1) * batch_size]
			y_batch = train_set_y[i * batch_size : (i + 1) * batch_size]
			train_cost = train_model(x_batch, y_batch)
			print("{}. {}/{} step's loss is {}.".format(epoch, i+1, n_train_batches, train_cost))

		for i in xrange(n_test_batches):
			x_batch = train_set_x[i * batch_size : (i + 1) * batch_size]
			y_batch = train_set_y[i * batch_size : (i + 1) * batch_size]
			test_accuracy = test_model(x_batch, y_batch)
			print("{}. {}/{} step's accuracy is {}.".format(epoch, i+1, n_test_batches, test_accuracy))
			
if __name__ == '__main__':

	train()


