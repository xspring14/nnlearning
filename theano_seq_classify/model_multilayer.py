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

        self.W1 = theano.shared(value=get(identifier=init, shape=(input_dim, hidden_dim, 3)),
                               name='W1',
                               borrow=True)
        self.U1 = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim, 3)),
                               name='U1',
                               borrow=True)

        self.W2 = theano.shared(value=get(identifier=init, shape=(hidden_dim, hidden_dim, 3)),
                                name='W2',
                                borrow=True)
        self.U2 = theano.shared(value=get(identifier=inner_init, shape=(hidden_dim, hidden_dim, 3)),
                                name='U2',
                                borrow=True)

        self.b = theano.shared(value=get(identifier='zero', shape=(hidden_dim, 6)),
                               name='b',
                               borrow=True)

        self.V = theano.shared(get(identifier=init, shape=(hidden_dim, output_dim)),
                               name='V',
                               borrow=True)
        self.b_y = theano.shared(get(identifier='zero', shape=(output_dim,)),
                                 name='b_y',
                                 borrow=True)

        self.h0 = theano.shared(get(identifier='zero', shape=(hidden_dim,)),
                                 name='h0',
                                 borrow=True)

        # for sgd
        self.mW1 = theano.shared(value=get(identifier='zero', shape=(input_dim, hidden_dim, 3)),
                                name='mW1',
                                borrow=True)
        self.mU1 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, hidden_dim, 3)),
                                name='mU1',
                                borrow=True)

        self.mW2 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, hidden_dim, 3)),
                                 name='mW2',
                                 borrow=True)
        self.mU2 = theano.shared(value=get(identifier='zero', shape=(hidden_dim, hidden_dim, 3)),
                                 name='mU2',
                                 borrow=True)

        self.mb = theano.shared(value=get(identifier='zero', shape=(hidden_dim, 6)),
                                name='mb',
                                borrow=True)

        self.mV = theano.shared(value=get(identifier='zero', shape=(hidden_dim, output_dim)),
                                name='mV',
                                borrow=True)
        self.mb_y = theano.shared(value=get(identifier='zero', shape=(output_dim,)),
                                  name='mb_y',
                                  borrow=True)


        self.params = [self.W1, self.U1,
                       self.W2, self.U2, self.b,
                       self.V, self.b_y]

        self.grads = [self.mW1, self.mU1,
                      self.mW2, self.mU2, self.mb,
                      self.mV, self.mb_y]


    def build(self, x, m, y, learning_rate):

        def reccurence(x_t, m_t, h_tm_prev1, h_tm_prev2):

            x_z1 = T.dot(x_t, self.W1[:,:,0]) + self.b[:,0]
            x_r1 = T.dot(x_t, self.W1[:,:,1]) + self.b[:,1]
            x_h1 = T.dot(x_t, self.W1[:,:,2]) + self.b[:,2]

            z_t1 = self.inner_activation(x_z1 + T.dot(h_tm_prev1, self.U1[:,:,0]))
            r_t1 = self.inner_activation(x_r1 + T.dot(h_tm_prev1, self.U1[:,:,1]))
            hh_t1 = self.activation(x_h1 + T.dot(r_t1 * h_tm_prev1, self.U1[:,:,2]))
            h_t1 =(T.ones_like(z_t1) - z_t1) * hh_t1 + z_t1 * h_tm_prev1

            h_t1 = m_t[:, None] * h_t1 + (1. - m_t)[:, None] * h_tm_prev1

            x_z2 = T.dot(h_t1, self.W2[:,:,0]) + self.b[:,3]
            x_r2 = T.dot(h_t1, self.W2[:,:,1]) + self.b[:,4]
            x_h2 = T.dot(h_t1, self.W2[:,:,2]) + self.b[:,5]

            z_t2 = self.inner_activation(x_z2 + T.dot(h_tm_prev2, self.U2[:,:,0]))
            r_t2 = self.inner_activation(x_r2 + T.dot(h_tm_prev2, self.U2[:,:,1]))
            hh_t2 = self.activation(x_h2 + T.dot(r_t2 * h_tm_prev2, self.U2[:,:,2]))
            h_t2 = (T.ones_like(z_t2) - z_t2) * hh_t2 + z_t2 * h_tm_prev2

            h_t2 = m_t[:, None] * h_t2 + (1. - m_t)[:, None] * h_tm_prev2

            return h_t1, h_t2
        # T.alloc(self.h_0, x.shape[1], self.hidden_dim)
        (self.h_t1, self.h_t2), _ = theano.scan(reccurence,
                                                sequences=[x, m],
                                                outputs_info=[T.alloc(self.h0, x.shape[1], self.hidden_dim),
                                                              T.alloc(self.h0, x.shape[1], self.hidden_dim)])
        self.y_t = T.nnet.softmax(T.dot(self.h_t2[-1], self.V) + self.b_y)

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