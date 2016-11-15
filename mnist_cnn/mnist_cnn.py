import tensorflow as tf
import cnn
from cnn_utee import Prefetcher


batch_size=100
SEED = 66478
NUM_CHANNELS=1
IMAGE_SIZE=28
NUM_LABELS=10
imgs_dir = '/home/mai/data'
train_list = '/home/mai/data/mnist/train/train.txt'
test_list = '/home/mai/data/mnist/test/test.txt'

train_prefetcher=Prefetcher(train_list, imgs_dir, batch_size)
test_prefetcher=Prefetcher(test_list, imgs_dir, batch_size)

train_size=train_prefetcher.n_samples
n_epochs=20
n_train_steps=train_prefetcher.n_samples//batch_size
n_test_steps=test_prefetcher.n_samples//batch_size


with tf.Graph().as_default():

	images_placeholder=tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
	labels_placeholder=tf.placeholder(tf.int64, shape=(batch_size,))

	logits=cnn.inference(images_placeholder, True)

	loss=cnn.loss(logits, labels_placeholder)

	# cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_placeholder)

	# loss=tf.reduce_mean(cross_entropy)

	# regularizers=(tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
    #               tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

	# loss += 5e-4 * regularizers

	batch=tf.Variable(0)

	learning_rate=tf.train.exponential_decay(
		0.01,
		batch*batch_size,
		train_size,
		0.95,
		staircase=True
		)

	optimizer=tf.train.MomentumOptimizer(learning_rate, 0.9)

	train_op=optimizer.minimize(loss, global_step=batch)

	#train_prediction=tf.nn.sofmax(logits)

	test_correct=cnn.testing(logits, labels_placeholder)

	with tf.Session() as sess:

		sess.run(tf.initialize_all_variables())

		for epoch in range(n_epochs):
			for step in range(n_train_steps):
				data=train_prefetcher.next_batch()
				_, loss_value=sess.run([train_op, loss], feed_dict={images_placeholder: data[0],
									   			                    labels_placeholder: data[1]})

				print("Training {}/{} step's loss is {}".format(step+1, n_train_steps, loss_value))
			print("Training {}/{} epoch finished... ".format(epoch+1, n_epochs))

			for step in range(n_test_steps):
				data=test_prefetcher.next_batch()
				acc_value=sess.run(test_correct, feed_dict={images_placeholder: data[0],
													        labels_placeholder: data[1]})
				print("Validation {}/{} step's accuracy is {}".format(step+1, n_test_steps, acc_value))
			print("Validation {}/{} epoch finished... ".format(epoch+1, n_epochs))