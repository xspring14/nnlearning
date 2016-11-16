### here use the rnn to train and test mnist

* the rnn network here is just a trial to train mnist by rnn. it has only one lstm layer without dropout.

* I use the last output of the seq in each picture as the input of the following hidden layer's input, after 20 epochs, the accuracy is 97.6%

* added: save and restore the model

