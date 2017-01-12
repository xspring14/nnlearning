### here use the rnn to train and test mnist

* the rnn network here is just a trial to train mnist by rnn. it has only one lstm layer without dropout.

* I use the last output of the seq in each picture as the input of the following hidden layer's input, after 10 epochs, the accuracy is 98.02%

* added: save and restore the model

* mnist\_rnn\_multi.py: using 2 stacked lstm layers, after 10 epochs the accuracy is 97.93%

* bit\_mnist\_rnn.py and bie\_mnist\_rnn.py: using bidirectional lstm to train mnist. after 10 epochs, the accuracy is 98.18%

* st\_mnist\_rnn.py and se\_mnist\_rnn.py: using multi-layers and dropout, and using the dynamic_rnn(which can handle the different length of seqs), after 20 epochs the accuracy is 98.46% by using the mean of output
