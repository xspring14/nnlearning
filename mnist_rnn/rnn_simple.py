import tensorflow as tf
import numpy as np

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

num_units=2
input_size=2

batch_size=50
seq_len=55
n_epochs=100


def gen_data(min_length=50, max_length=55, n_batch=5):

    X = np.concatenate([np.random.uniform(size=(n_batch, max_length, 1)),
                        np.zeros((n_batch, max_length, 1))],
                       axis=-1)
    y = np.zeros((n_batch,))
    # Compute masks and correct values
    for n in range(n_batch):
        # Randomly choose the sequence length
        length = np.random.randint(min_length, max_length)
        #i changed this to a constant
        #length=55

        # Zero out X after the end of the sequence
        X[n, length:, 0] = 0
        # Set the second dimension to 1 at the indices to add
        X[n, np.random.randint(length/2-1), 1] = 1
        X[n, np.random.randint(length/2, length), 1] = 1
        # Multiply and sum the dimensions of X to get the target value
        y[n] = np.sum(X[n, :, 0]*X[n, :, 1])
    # Center the inputs and outputs
    #X -= X.reshape(-1, 2).mean(axis=0)
    #y -= y.mean()
    return (X,y)


cell=rnn_cell.BasicLSTMCell(num_units)

# data placeholder
inputs=[tf.placeholder(tf.float32, shape=[batch_size, input_size])
		for _ in range(seq_len)]
result=tf.placeholder(tf.float32, shape=[batch_size])

outputs, states=rnn.rnn(cell, inputs, dtype=tf.float32)

outputs2=outputs[-1]

W_o=tf.Variable(tf.random_normal([2,1], stddev=0.01))
b_o=tf.Variable(tf.random_normal([1],stddev=0.01))

outputs3=tf.matmul(outputs2, W_o)+b_o

cost=tf.reduce_mean(tf.pow(outputs3-result, 2))

train_op=tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)




with tf.Session() as sess:

	sess.run(tf.initialize_all_variables())

	for epoch in range(n_epochs):

		# loading data
		tempX, y_val=gen_data(50, seq_len, batch_size)
		X_val=[]
		for i in range(seq_len):
			X_val.append(tempX[:, i, :])	# which save the data by step and step

		temp_dict={inputs[i]:X_val[i] for i in range(seq_len)}
		temp_dict.update({result:y_val})

		sess.run(train_op, feed_dict=temp_dict)

		val_dict={inputs[i]:X_val[i] for i in range(seq_len)}
		val_dict.update({result:y_val})

		c_val=sess.run(cost, feed_dict=val_dict)

		print("Validation cost: {}, on Epoch {}".format(c_val,epoch))







