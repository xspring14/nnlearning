import theano
import theano.tensor as T
from load_data import *
from model import Gru

def train():

    x = T.tensor3('x')
    m = T.matrix('m')
    y = T.ivector('y')
    learning_rate = T.scalar('lr')

    print "initializing model..."
    model = Gru()

    print "building model..."
    model.build(x, m, y, learning_rate)

    print "loading data..."
    batch_size = 64
    imgs_dir = "/home/xu/data/"
    train_imgs_list = "/home/xu/data/train100.txt"
    test_imgs_list = "/home/xu/data/val100.txt"

    train_prefetcher = Prefetcher(imgs_dir, train_imgs_list, batch_size)
    test_prefetcher = Prefetcher(imgs_dir, test_imgs_list, batch_size)

    n_train_steps = train_prefetcher.n_samples / batch_size
    n_test_steps = test_prefetcher.n_samples / batch_size

    lr = 1e-3
    # total_step = 0L
    for epoch in range(20):

        if epoch == 10:
            lr = lr * 0.1
        for i in range(n_train_steps):

            x_batches, l_batches, y_batches = train_prefetcher.next_batch()
            x_batches = np.transpose(x_batches, (1, 0, 2))
            l_batches = np.transpose(l_batches, (1, 0))
            train_cost = model.sgd(x_batches, l_batches, y_batches, lr)
            print("{}. {}/{} step's loss is {}".format(epoch+1, i+1, n_train_steps, train_cost))

        for i in range(n_test_steps):

            x_batches, l_batches, y_batches = test_prefetcher.next_batch()
            x_batches = np.transpose(x_batches, (1, 0, 2))
            l_batches = np.transpose(l_batches, (1, 0))
            test_acc = model.predict(x_batches, l_batches, y_batches)
            print("{}. {}/{} step's accuracy is {}".format(epoch+1, i+1, n_test_steps, test_acc))

if __name__ == '__main__':

    train()
