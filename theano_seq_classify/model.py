import theano
import theano.tensor as T
from  initial import get

class Gru(object):

    def __init__(self, input_dim=2, hidden_dim=200, output_dim=100,
                 init='uniform', inner_init='orthonormal', activation=T.tanh,
                 inner_activation=T.nnet.hard_sigmoid):

        self.activation = activation
        self.inner_activation = inner_activation
        self.hidden_dim = hidden_dim

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

        # for sgd
        self.mW_z = theano.shared(value=get(identifier='zero', shape=(input_dim, hidden_dim)),
                                  name='mW_z',
                                  borrow=True)
        self.mU_z = theano.shared(value=get(identifier='zero', shape=(hidden_dim, hidden_dim)),
                                  name='mW_z',
                                  borrow=True)
        self.mb_z = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                  name='mb_z',
                                  borrow=True)

        self.mW_r = theano.shared(value=get(identifier='zero', shape=(input_dim, hidden_dim)),
                                  name='mW_z',
                                  borrow=True)
        self.mU_r = theano.shared(value=get(identifier='zero', shape=(hidden_dim, hidden_dim)),
                                  name='mW_z',
                                  borrow=True)
        self.mb_r = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                  name='mb_z',
                                  borrow=True)

        self.mW = theano.shared(value=get(identifier='zero', shape=(input_dim, hidden_dim)),
                                  name='mW',
                                  borrow=True)
        self.mU = theano.shared(value=get(identifier='zero', shape=(hidden_dim, hidden_dim)),
                                  name='mU',
                                  borrow=True)
        self.mb = theano.shared(value=get(identifier='zero', shape=(hidden_dim,)),
                                name='mb',
                                borrow=True)

        self.mV = theano.shared(value=get(identifier='zero', shape=(hidden_dim, output_dim)),
                                name='mV',
                                borrow=True)
        self.mb_y = theano.shared(value=get(identifier='zero', shape=(output_dim,)),
                                  name='mb_y',
                                  borrow=True)


        self.params = [self.W_z, self.U_z, self.b_z,
                       self.W_r, self.U_r, self.b_r,
                       self.W, self.U, self.b,
                       self.V, self.b_y]

        self.grads = [self.mW_z, self.mU_z, self.mb_z,
                      self.mW_r, self.mU_r, self.mb_r,
                      self.mW, self.mU, self.mb,
                      self.mV, self.mb_y]


    def build(self, x, m, y, learning_rate):

        # x = x.dimshuffle(1, 0, 2)

        def reccurence(x_t, m_t, h_tm_prev):

            x_z = T.dot(x_t, self.W_z) + self.b_z
            x_r = T.dot(x_t, self.W_r) + self.b_r
            x_h = T.dot(x_t, self.W) + self.b

            z_t = self.inner_activation(x_z + T.dot(h_tm_prev, self.U_z))
            r_t = self.inner_activation(x_r + T.dot(h_tm_prev, self.U_r))
            hh_t = self.activation(x_h + T.dot(r_t * h_tm_prev, self.U))
            h_t =(T.ones_like(z_t) - z_t) * hh_t + z_t * h_tm_prev

            # h_t = h_t * (T.ones_like(h_t) * m_t) + h_tm_prev * (T.ones_like(h_t) * (1 - m_t))
            # h_t = h_t * m_t + h_tm_prev * (1 - m_t)
            h_t = m_t[:, None] * h_t + (1. - m_t)[:, None] * h_tm_prev

            return h_t

        self.h_t, _ = theano.scan(reccurence,
                                  sequences=[x, m],
                                  outputs_info=[T.alloc(self.h_0, x.shape[1], self.hidden_dim)])

        self.y_t = T.nnet.softmax(T.dot(self.h_t[-1], self.V) + self.b_y)

        self.y = T.argmax(self.y_t, axis=1)

        cost = T.mean(T.nnet.categorical_crossentropy(self.y_t, y))
        accuracy = T.mean(T.eq(self.y ,y))

        # for updates
        decay = 0.9
        updates = []
        for p, g in zip(self.params, self.grads):
            dp = T.grad(cost, p)
            mg = decay * g + (1 - decay) * dp ** 2

            updates.append((p, p - learning_rate * dp / T.sqrt(mg + 1e-6)))
            updates.append((g, mg))

        self.sgd = theano.function(inputs=[x, m, y, learning_rate], outputs=cost, updates=updates)
        self.predict = theano.function(inputs=[x, m, y], outputs=accuracy)

        return self.sgd, self.predict

if __name__ == '__main__':

    print "init model..."
    model = Gru()

    x = T.tensor3('x')
    m = T.matrix('m')
    y = T.ivector('y')
    learning_rate = T.scalar('learning_rate')

    print "build model..."
    model.build(x, m, y, learning_rate)