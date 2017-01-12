import tensorflow as tf
SEED=66478	# for random initialize the weights
IMAGE_SIZE=28

def inference(images, Train=False):
	"""
	build the cnn net, with 2 convs
	"""

	with tf.name_scope("conv1"):
		conv_weights=tf.Variable(
			tf.truncated_normal([5, 5, 1, 32], stddev=0.1, seed=SEED),
			name='conv_weights'
			)

		conv_biases=tf.Variable(
			tf.zeros([32]),
			name='conv_biases')

		conv=tf.nn.conv2d(images,
						  conv_weights,
						  strides=[1, 1, 1, 1],
						  padding='SAME')

		relu=tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

		pool=tf.nn.max_pool(relu,
							ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1],
							padding='SAME')

	with tf.name_scope("conv2"):
		conv_weights=tf.Variable(
			tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED),
			name="conv_weights"
			)

		conv_biases=tf.Variable(
			tf.constant(0.1, shape=[64]),
		 	name="conv_biases"
		 	)

		conv=tf.nn.conv2d(pool,
						  conv_weights,
						  strides=[1, 1, 1, 1],
						  padding='SAME')

		relu=tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

		pool=tf.nn.max_pool(relu,
							ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1],
							padding='SAME')

		pool_shape=pool.get_shape().as_list()
		reshape=tf.reshape(
			pool,
			[pool_shape[0], pool_shape[1]*pool_shape[2]*pool_shape[3]])

	with tf.name_scope("fc1"):
		fc_weights=tf.Variable(
			tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512], stddev=0.1, seed=SEED),
			name="fc_weights"
			)

		fc_biases=tf.Variable(
			tf.constant(0.1, shape=[512]),
			name="fc_biases"
			)

		tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(fc_weights), 5e-4, name='weight_loss'))
		# tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(fc_biases), 5e-4, name='weight_loss'))

		hidden=tf.nn.relu(tf.matmul(reshape, fc_weights)+fc_biases)

		# add dropout
		if Train:
			hidden=tf.nn.dropout(hidden, 0.5, seed=SEED)

	with tf.name_scope("fc2"):
		fc_weights=tf.Variable(
			tf.truncated_normal([512, 10], stddev=0.1, seed=SEED),
			name="fc_weights"
			)

		fc_biases=tf.Variable(
			tf.constant(0.1, shape=[10]),
			name="fc_biases"
			)

		tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(fc_weights), 5e-4, name='weight_loss'))
		# tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(fc_biases), 5e-4, name='weight_loss'))

		logits=tf.matmul(hidden, fc_weights)+fc_biases

	return logits


def loss(logits, labels):

	cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) # dense labels
	cross_entropy_mean=tf.reduce_mean(cross_entropy)

	tf.add_to_collection('losses', cross_entropy_mean)

	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def testing(logits, labels):

	prediction=tf.nn.softmax(logits)

	correct=tf.nn.in_top_k(prediction, labels, 1)

	return tf.reduce_sum(tf.cast(correct, tf.int32))









