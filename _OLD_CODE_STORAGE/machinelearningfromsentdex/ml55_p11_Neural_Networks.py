import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.python.ops import rnn, rnn_cell
#from tensorflow.nn import rnn, rnn_cell
from tensorflow.contrib import rnn
#from tensorflow.python.ops import rnn
'''
input > weight > hidden layer1(activation function)>
weights > hidden layer2(activation fun) > weights
>output layer
feed forward.

compare output to intended output > cost or loss function?(cross entropy)
optimization function(optimizer) > minimize cost (adamoptimizer...sgd, adagrad)

backpropagation

feed forward + backprop = epoch

'''

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

'''
0=[1,0,0,0,0,0,0,0,0,0]
1=[0,1,0,0,0,0,0,0,0,0]
2=[0,0,1,0,0,0,0,0,0,0]
3=[0,0,0,1,0,0,0,0,0,0]
4=[0,0,0,0,1,0,0,0,0,0]
'''

hm_epochs = 3
n_classes = 10
batch_size = 50#128
chunk_size = 28
n_chunks = 28
rnn_size = 128

# height x width
x = tf.placeholder('float',[None, n_chunks,chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):

    #(input_data* weights) + baises

    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))
            }
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    #x = tf.split(0, n_chunks, x)
    x = tf.split(x, n_chunks, 0)

    #lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    #outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    #looks like they changed how values passed to this func, softmax_cross_entropy_with_logits
    #                    learning rate = 0.001...
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles feed forward + backprop


    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                #reshape epoch_x
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch',epoch,'completed out of',hm_epochs,'los:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))


train_neural_network(x)
