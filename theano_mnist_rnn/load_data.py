import numpy as np
from PIL import Image


def load_train_data():
	data = np.empty((60000,28,28),dtype="float32")
	label = np.empty((60000,),dtype="int32")

	imgs_list=file('/home/xu/data/mnist/train/total.txt').readlines()
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

if __name__ == '__main__':

    data, label = load_test_data()
    print data.shape
    print label.shape