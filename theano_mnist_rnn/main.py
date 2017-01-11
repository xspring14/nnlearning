import theano
import theano.tensor as T
from load_data import *
from gru_model import Gru
from optimizer import rmsprop

def train():

    x = T.tensor3('x')
    y = T.ivector('y')

    print "building model..."
    model = Gru(x)
    cost = model.crossentropy(y)
    acc = model.accuracy(y)

    updates = rmsprop(cost, model.params)
    print "forming model..."
    train_model = theano.function(inputs=[x, y],
                                  outputs=cost,
                                  updates=updates)

    test_model = theano.function(inputs=[x, y], outputs=acc)

    print "loading data..."
    train_set_x, train_set_y = load_train_data()
    test_set_x, test_set_y = load_test_data()

    batch_size = 64
    n_train_steps = train_set_x.shape[0] / batch_size
    n_test_steps = test_set_x.shape[0] / batch_size

    for epoch in range(20):

        for i in range(n_train_steps):

            x_batches = train_set_x[i * batch_size : (i+1) * batch_size]
            y_batches = train_set_y[i * batch_size : (i+1) * batch_size]

            train_cost = train_model(x_batches, y_batches)
            print("{}. {}/{} step's loss is {}".format(epoch+1, i+1, n_train_steps, train_cost))

        for i in range(n_test_steps):

            x_batches = test_set_x[i * batch_size : (i+1) * batch_size]
            y_batches = test_set_y[i * batch_size : (i+1) * batch_size]

            test_acc = test_model(x_batches, y_batches)
            print("{}. {}/{} step's loss is {}".format(epoch+1, i+1, n_test_steps, test_acc))

if __name__ == '__main__':

    train()