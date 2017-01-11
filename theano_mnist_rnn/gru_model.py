import  theano
import  theano.tensor as T
from initial import get

class Gru(object):

    def __init__(self, inputs, input_dim=28, hidden_dim=200, output_dim=10,
                 init='uniform', inner_init='orthonormal', activation=T.tanh,
                 inner_activation=T.nnet.hard_sigmoid):

        self.activation = activation
        self.inner_activation = inner_activation

        self.W_z = theano.shared(get(identifier=init, shape=(input_dim, hidden_dim)),
                                 name='W_z',
                                 borrow=True)
        self.U_z = theano.shared(get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                 name='U_z',
                                 borrow=True)
        self.b_z = theano.shared(get(identifier='zero', shape=(hidden_dim,)),
                                 name='b_z',
                                 borrow=True)

        self.W_r = theano.shared(get(identifier=init, shape=(input_dim, hidden_dim)),
                                 name='W_r',
                                 borrow=True)
        self.U_r = theano.shared(get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                 name='U_r',
                                 borrow=True)
        self.b_r = theano.shared(get(identifier='zero', shape=(hidden_dim,)),
                                 name='b_r',
                                 borrow=True)

        self.W = theano.shared(get(identifier=init, shape=(input_dim, hidden_dim)),
                                 name='W',
                                 borrow=True)
        self.U = theano.shared(get(identifier=inner_init, shape=(hidden_dim, hidden_dim)),
                                 name='U',
                                 borrow=True)
        self.b = theano.shared(get(identifier='zero', shape=(hidden_dim,)),
                                 name='b',
                                 borrow=True)

        self.V = theano.shared(get(identifier=init, shape=(hidden_dim, output_dim)),
                               name='V',
                               borrow=True)
        self.b_y = theano.shared(get(identifier='zero', shape=(output_dim,)),
                                 name='b_y',
                                 borrow=True)
        self.h_0 = theano.shared(get(identifier='zero', shape=(hidden_dim,)),
                                 name='h_0',
                                 borrow=True)

        self.params = [self.W_z, self.U_z, self.b_z,
                       self.W_r, self.U_r, self.b_r,
                       self.W, self.U, self.b,
                       self.V, self.b_y]

        self.__build__(inputs, hidden_dim)

    def __build__(self, inputs, hidden_dim):

        inputs = inputs.dimshuffle(1, 0, 2)

        def reccurence(x_t, h_tm_prev):

            x_z = T.dot(x_t, self.W_z) + self.b_z
            x_r = T.dot(x_t, self.W_r) + self.b_r
            x_h = T.dot(x_t, self.W) + self.b

            z_t = self.inner_activation(x_z + T.dot(h_tm_prev, self.U_z))
            r_t = self.inner_activation(x_r + T.dot(h_tm_prev, self.U_r))
            hh_t = self.activation(x_h + T.dot(r_t * h_tm_prev, self.U))
            h_t =(T.ones_like(z_t) - z_t) * hh_t + z_t * h_tm_prev

            return h_t

        self.h_t, _ = theano.scan(reccurence,
                                  sequences=inputs,
                                  outputs_info=[T.alloc(self.h_0, inputs.shape[1], hidden_dim)])

        self.y_t = T.nnet.softmax(T.dot(self.h_t[-1], self.V) + self.b_y)

        self.y = T.argmax(self.y_t, axis=1)

    def crossentropy(self, y):

        return T.mean(T.nnet.categorical_crossentropy(self.y_t, y))

    def accuracy(self, y):

        return T.mean(T.eq(self.y, y))


if __name__ == '__main__':

    inputs = T.tensor3('inputs')
    model = Gru(inputs)