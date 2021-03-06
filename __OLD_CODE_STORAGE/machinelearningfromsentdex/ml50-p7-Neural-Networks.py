import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from ml49_Neural_Networks import creat_feature_sets_and_labels
import numpy as np

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

#mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

'''
0=[1,0,0,0,0,0,0,0,0,0]
1=[0,1,0,0,0,0,0,0,0,0]
2=[0,0,1,0,0,0,0,0,0,0]
3=[0,0,0,1,0,0,0,0,0,0]
4=[0,0,0,0,1,0,0,0,0,0]
'''
train_x,train_y,test_x,test_y = creat_feature_sets_and_labels('pos.txt','neg.txt')

n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000

n_classes = 2
batch_size = 100

# height x width
x = tf.placeholder('float',[None, len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):

    #(input_data* weights) + baises

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))
                      }
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))
                      }
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))
                      }
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))
                      }

    #(input_data* weights) + baises

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) , hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    #looks like they changed how values passed to this func, softmax_cross_entropy_with_logits
    #                    learning rate = 0.001...
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles feed forward + backprop
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                ##ses = tf.Session(config=tf.ConfigProto(log_device_placement=True))
                epoch_loss += c
                i += batch_size
            print('Epoch',epoch,'completed out of',hm_epochs,'los:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))


train_neural_network(x)
