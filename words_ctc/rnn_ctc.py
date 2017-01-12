from utee import Prefetcher
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
import tensorflow.contrib.ctc as ctc
import numpy as np
import math


batch_size=325
input_dim=28
input_max_lenth=250
target_max_length=20
hidden_size=100
only_forward=True
dropout=0.5
n_layers=2
n_epochs=200
n_classes=96 	# added the blank
multi_iters=[50, 100, 150, 175]


data_dir="/home/mai/workspace/github/cnn-lstm-ctc/dataset/english_sentence/split_tiny_images"
train_list='/home/mai/workspace/github/cnn-lstm-ctc/dataset/english_sentence/train_img_list.txt'
val_list='/home/mai/workspace/github/cnn-lstm-ctc/dataset/english_sentence/test_img_list.txt'

train_prefetcher=Prefetcher(data_dir, train_list, batch_size, input_max_lenth)
n_train_steps=train_prefetcher.n_samples//batch_size
val_prefetcher=Prefetcher(data_dir, val_list, batch_size, input_max_lenth)
n_val_steps=val_prefetcher.n_samples//batch_size

chars=val_prefetcher.chars
# print(len(chars))


# define models

input_data=tf.placeholder(tf.float32, shape=[batch_size, input_max_lenth, input_dim])
input_lens=tf.placeholder(tf.int32, shape=[batch_size])
# target_lens=tf.placeholder(tf.int16, shape=[batch_size])
target_vals=tf.placeholder(tf.int32, shape=[None])
target_idxes=tf.placeholder(tf.int64, shape=[None, 2])
target_shape= [batch_size, target_max_length]	 # unsure

cell=rnn_cell.BasicLSTMCell(hidden_size)

if not only_forward and dropout<1.0:
	cell=rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=dropout)

if n_layers>1:
	cell=rnn_cell.MultiRNNCell([cell]*n_layers)

inputs=[tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, input_max_lenth, input_data)]

rnn_outputs, _ =rnn.dynamic_rnn(cell, tf.pack(inputs), sequence_length=input_lens, dtype=tf.float32, time_major=True)

w_o=tf.Variable(tf.truncated_normal([hidden_size, n_classes], stddev=1.0/np.sqrt(n_classes)))

b_o=tf.Variable(tf.zeros([n_classes]))

logits=[tf.matmul(tf.squeeze(i, squeeze_dims=[0]), w_o)+b_o for i in tf.split(0, input_max_lenth, rnn_outputs)]

logits=tf.pack(logits)


# if not only_forward:
sparse_labels=tf.SparseTensor(
	indices=target_idxes,
	values=target_vals,
	shape=[batch_size, target_max_length]
	)

if not only_forward:

	loss=ctc.ctc_loss(logits, sparse_labels, input_lens)

	loss_mean=tf.reduce_mean(loss)


	# updates
	learning_rate=tf.Variable(0.01, trainable=False)
	lr_decay_op=learning_rate.assign(learning_rate*0.1)

	max_grad_norm=5

	tvars=tf.trainable_variables()

	grads, _=tf.clip_by_global_norm(tf.gradients(loss_mean, tvars), max_grad_norm)

	optimizer=tf.train.AdamOptimizer(learning_rate)

	train_op=optimizer.apply_gradients(zip(grads, tvars))


# prediction
prediction = tf.to_int32(ctc.ctc_beam_search_decoder(logits, input_lens)[0][0])

error_rate = tf.reduce_sum(tf.edit_distance(prediction, sparse_labels, normalize=False)) / tf.to_float(tf.size(sparse_labels.values))

saver=tf.train.Saver(tf.all_variables())

with tf.Session() as sess:

	sess.run(tf.initialize_all_variables())
	# saver.restore(sess, "model.ckpt")

	train_step=0
	for epoch in range(1):
		
		for step in range(n_train_steps):
			
			x_val, x_len, y_val, y_idx=train_prefetcher.next_batch()

			input_feed={input_data:x_val, input_lens:x_len,
						target_vals:y_val, target_idxes:y_idx}

			output_feed=[train_op, loss_mean]

			_, cost=sess.run(output_feed, input_feed)
			print("{}. {}/{} step, loss is {}".format(epoch+1, step, n_train_steps, cost))

			if math.isnan(cost):
				exit()
		

		for step in range(n_val_steps):

			x_val, x_len, y_val, y_idx=val_prefetcher.next_batch()

			input_feed={input_data:x_val, input_lens:x_len,
						target_vals:y_val, target_idxes:y_idx}

			output_feed=[prediction, error_rate]

			predict, rate=sess.run(output_feed, input_feed)

			print("{}. {}/{} step, correct_rate is {}".format(epoch+1, step, n_val_steps, rate))
			print("predict's values's len is {}".format(len(predict.values)))

		print("the previous 10 prediction is :")
		for i in range(5):
			row1=np.where(y_idx[:,0]==i)[0]
			row2=np.where(predict.indices[:,0]==i)[0]
			# print rows
			seen=""
			pred=""
			for r in row1:
				seen+=chars[y_val[r]]
			for r in row2:
				pred+=chars[predict.values[r]]
			print("{}. seen:{}    predict:{}".format(i, seen, pred))


		train_step+=1
		if train_step in multi_iters:
	 		sess.run(lr_decay_op)


	model_path="./model.ckpt"
	print("save model in {}".format(saver.save(sess, model_path)))

