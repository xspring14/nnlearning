comparing with raw mnist_cnn, the difference is:
1. using the sparse_softmax_cross_entropy_logits() to compute the cross entropy, which auto transformed to the onehot vector

2. put the model i.e. the inference back to the cnn, and use tf.add_to_collection to put the fc weights's l2-loss in to the total loss

3. the method to using tf.add_to_collection is:
tf.add_to_collection(name, value) i.e tf.add_to_collection('losses', var).
by using the same name, the losses can be combined to the same collection

for example:
tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(fc_weights1), 5e-4, name='weight_loss'))
tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(fc_weights2), 5e-4, name='weight_loss'))
tf.add_to_collection('losses', cross_entropy_mean)

to return the losses, we use: tf.add_n(tf.get_collection('losses'), name='total_loss')
which tf.add_n() is the function to return all the losses combined togother, name can be ommited


