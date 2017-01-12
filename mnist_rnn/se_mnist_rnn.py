import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
from utee import Prefetcher


hidden_size=90
input_dim=28
max_input_seq_length=28
num_labels=10
batch_size=50
n_layers=2


# model 
input_data = tf.placeholder(tf.float32,shape=[None, max_input_seq_length, input_dim],
						name="inputs")

result=tf.placeholder(tf.int32, shape=[None])

input_seq_lengths = tf.placeholder(tf.int32, shape=[None], name="input_seq_lengths")

# input transform
inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, max_input_seq_length, input_data)]

cell = rnn_cell.BasicLSTMCell(hidden_size)

cell = rnn_cell.MultiRNNCell([cell] * n_layers)

init_state = cell.zero_state(batch_size, tf.float32)

rnn_output, hidden_state = rnn.dynamic_rnn(cell, tf.pack(inputs), sequence_length=input_seq_lengths,
										   initial_state=init_state, time_major=True)

output=tf.reduce_mean(rnn_output, 0)

# output transform
w_o = tf.Variable(tf.truncated_normal([hidden_size, num_labels], stddev=np.sqrt(1.0 / num_labels)),
                  name="output_w")
b_o = tf.Variable(tf.zeros([num_labels]), name="output_b")

logits=tf.matmul(output, w_o)+b_o

# for validation and test
prediction=tf.nn.softmax(logits)

correct=tf.nn.in_top_k(prediction, result, 1)

correct_sum=tf.reduce_sum(tf.cast(correct, tf.int32))

correct_mean=tf.cast(correct_sum, tf.float32)/batch_size

# for test
data_dir='/home/mai/data'
test_list='/home/mai/data/mnist/test/test.txt'
test_prefetcher=Prefetcher(test_list, data_dir, batch_size)

n_test_steps=test_prefetcher.n_samples//batch_size

saver=tf.train.Saver(tf.all_variables())

with tf.Session() as sess:

	# sess.run(tf.initialize_all_variables())
	saver.restore(sess, "./model_st.ckpt-16000")

	correct_num=0.0

	for step in range(n_test_steps):

		x_val, y_val, l_val=test_prefetcher.next_batch()
		# temp_dict={input_data:x_val, result:y_val}
		temp_dict={input_data:x_val, result:y_val, input_seq_lengths:l_val}
		# temp_dict.update({input_seq_lengths:l_val})

		corrects=sess.run([correct_sum], feed_dict=temp_dict)

		correct_num+=sum(corrects)

		print("{}/{} step, loss is {}".format(step+1, n_test_steps, corrects[0]))

	print("test accuracy is {}".format(correct_num/test_prefetcher.n_samples))