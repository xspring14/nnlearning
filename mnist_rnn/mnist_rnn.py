import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
from utee import Prefetcher


n_units=90
n_input=28
seq_len=28
n_epochs=10
n_classes=10
batch_size=60


data_dir='/home/mai/data'
train_list='/home/mai/data/mnist/train/train.txt'
val_list='/home/mai/data/mnist/train/val.txt'
test_list='/home/mai/data/mnist/test/test.txt'

train_prefetcher=Prefetcher(train_list, data_dir, batch_size)
val_prefetcher=Prefetcher(val_list, data_dir, batch_size)
test_prefetcher=Prefetcher(test_list, data_dir, batch_size)

n_train_steps=train_prefetcher.n_samples//batch_size
n_val_steps=val_prefetcher.n_samples//batch_size
n_test_steps=test_prefetcher.n_samples//batch_size

cell=rnn_cell.BasicLSTMCell(n_units)

inputs=[tf.placeholder(tf.float32, shape=[batch_size, n_input])
		for _ in range(seq_len)]
result=tf.placeholder(tf.int32, shape=[batch_size])

outputs, state=rnn.rnn(cell, inputs, dtype=tf.float32)

#outputs2=[tf.reduce_mean(seq, 0) for seq in outputs]
outputs2=outputs[-1]

W_o=tf.Variable(tf.random_normal([n_units, n_classes], stddev=0.01), name='W_0')
b_o=tf.Variable(tf.random_normal([n_classes], stddev=0.01), name='b_o')

outputs3=tf.matmul(outputs2, W_o)+b_o

# for train
cost=tf.nn.sparse_softmax_cross_entropy_with_logits(outputs3, result)
cost_mean=tf.reduce_mean(cost)

# for validation and test
prediction=tf.nn.softmax(outputs3)
correct=tf.nn.in_top_k(prediction, result, 1)
correct_sum=tf.reduce_sum(tf.cast(correct, tf.int32))
correct_mean=tf.cast(correct_sum, tf.float32)/batch_size

optimizer=tf.train.RMSPropOptimizer(0.005, 0.9)
global_step=tf.Variable(0, trainable=False)
train_op=optimizer.minimize(cost_mean, global_step=global_step)

saver=tf.train.Saver(tf.all_variables())

with tf.Session() as sess:

	sess.run(tf.initialize_all_variables())
	# saver.restore(sess, "./model.ckpt-15200")

	for epoch in range(n_epochs):

		for step in range(n_train_steps):			

			tempX, y_val=train_prefetcher.next_batch()

			X_val=[]

			for i in range(seq_len):
				X_val.append(tempX[:, i, :])

			temp_dict={inputs[i]:X_val[i] for i in range(seq_len)}
			temp_dict.update({result:y_val})

			_, cost_mean_val=sess.run([train_op, cost_mean], feed_dict=temp_dict)

			print("{}. {}/{} step, loss is {}".format(epoch+1, step+1, n_train_steps, cost_mean_val))

		for step in range(n_val_steps):

			tempX, y_val=val_prefetcher.next_batch()

			X_val=[]

			for i in range(seq_len):
				X_val.append(tempX[:, i, :])

			temp_dict={inputs[i]:X_val[i] for i in range(seq_len)}
			temp_dict.update({result:y_val})

			correct_mean_val=sess.run([correct_mean], feed_dict=temp_dict)

			print("{}. {}/{} step, loss is {}".format(epoch+1, step+1, n_val_steps, correct_mean_val))			

	print("{} epoch finished, begin testing".format(n_epochs))
	checkpoint_path='./model.ckpt'
	saver.save(sess, checkpoint_path, global_step=n_epochs*n_train_steps)
	print("saved model in {}".format(checkpoint_path))

	# for test
	cor_sum=0.0
	for step in range(n_test_steps):

		tempX, y_val=test_prefetcher.next_batch()

		X_val=[]

		for i in range(seq_len):
			X_val.append(tempX[:, i, :])

		temp_dict={inputs[i]:X_val[i] for i in range(seq_len)}
		temp_dict.update({result:y_val})

		correct_val=sess.run([correct_sum], feed_dict=temp_dict)		

		cor_sum+=sum(correct_val)

	print("tesing result is {}".format(cor_sum/test_prefetcher.n_samples))

	















