### RNN-CTC

* Here I used the rnn-ctc to classify the words in the pictures.
* At first, I'm so naive to just use the AdamOptimizer that after some epochs the training process always occurs the nan problem, then I used the grad_clip, which tackled this.
* For the training data, it's [here](https://github.com/aaron-xichen/cnn-lstm-ctc.git)



