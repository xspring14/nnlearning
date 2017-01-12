import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
from utee import Prefetcher


hidden_size=90
input_dim=28
max_input_seq_length=28
n_epochs=10
num_labels=10
batch_size=60
n_layers=2


# model 
input_data = tf.placeholder(tf.float32,shape=[None, max_input_seq_length, input_dim],
						name="inputs")

result=tf.placeholder(tf.int32, shape=[None])

# input transform
inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, max_input_seq_length, input_data)]

cell_fw = rnn_cell.BasicLSTMCell(hidden_size)

cell_bw = rnn_cell.BasicLSTMCell(hidden_size)

# cell = rnn_cell.MultiRNNCell([cell] * n_layers)

rnn_output, _, _ = rnn.bidirectional_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32)
# method 1
# output=tf.reduce_mean(rnn_output, 0)
# method 2
# output=tf.squeeze(tf.split(0, max_input_seq_length, rnn_output)[-1], squeeze_dims=[0])
output=rnn_output[-1]

# output transform
w_o = tf.Variable(tf.truncated_normal([hidden_size*2, num_labels], stddev=np.sqrt(1.0 / num_labels)),
                  name="output_w")
b_o = tf.Variable(tf.zeros([num_labels]), name="output_b")

logits=tf.matmul(output, w_o)+b_o

cost=tf.nn.sparse_softmax_cross_entropy_with_logits(logits, result)

cost_mean=tf.reduce_mean(cost)

global_step=tf.Variable(0, trainable=False)

optimizer=tf.train.RMSPropOptimizer(0.005, 0.9)

train_op=optimizer.minimize(cost_mean, global_step=global_step)

# for validation and test
prediction=tf.nn.softmax(logits)

correct=tf.nn.in_top_k(prediction, result, 1)

correct_sum=tf.reduce_sum(tf.cast(correct, tf.int32))

correct_mean=tf.cast(correct_sum, tf.float32)/batch_size


data_dir='/home/mai/data'
train_list='/home/mai/data/mnist/train/train.txt'
val_list='/home/mai/data/mnist/train/val.txt'

train_prefetcher=Prefetcher(train_list, data_dir, batch_size)
val_prefetcher=Prefetcher(val_list, data_dir, batch_size)

n_train_steps=train_prefetcher.n_samples//batch_size
n_val_steps=val_prefetcher.n_samples//batch_size

saver=tf.train.Saver(tf.all_variables())
with tf.Session() as sess:

	sess.run(tf.initialize_all_variables())
	# saver.restore(sess, "./model_st.ckpt-8000")
	for epoch in range(n_epochs):

		for step in range(n_train_steps):
			#print type(outputs)

			x_val, y_val, l_val=train_prefetcher.next_batch()
			temp_dict={input_data:x_val, result:y_val}
			# temp_dict.update({input_seq_lengths:l_val})

			_, cost_mean_val=sess.run([train_op, cost_mean], feed_dict=temp_dict)

			print("{}. {}/{} step, loss is {}".format(epoch+1, step+1, n_train_steps, cost_mean_val))

		for step in range(n_val_steps):

			x_val, y_val, l_val=val_prefetcher.next_batch()
			temp_dict={input_data:x_val, result:y_val}
			# temp_dict.update({input_seq_lengths:l_val})

			correct_mean_val=sess.run([correct_mean], feed_dict=temp_dict)

			print("{}. {}/{} step, loss is {}".format(epoch+1, step+1, n_val_steps, correct_mean_val))


	model_path="./model_st.ckpt"
	saver.save(sess, model_path, global_step=n_epochs*n_train_steps*2)
