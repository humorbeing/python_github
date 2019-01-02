import tensorflow_playground as tf
from tensorflow_playground.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


image_size = 28
labels_size = 10
learning_rate = 0.05
steps_number = 1000
batch_size = 100

training_data = tf.placeholder(tf.float32, [None, image_size*image_size])
labels = tf.placeholder(tf.float32, [None, labels_size])

w = tf.Variable(tf.truncated_normal([image_size*image_size, labels_size], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[labels_size]))

output = tf.matmul(training_data, w) + b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(steps_number):
    input_batch, labels_batch = mnist.train.next_batch(batch_size)
    feed_dict = {
        training_data: input_batch,
        labels: labels_batch
    }
    train_step.run(feed_dict=feed_dict)

    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict=feed_dict)
        print('Step %d, training batch accuracy %g %%'%(i, train_accuracy*100))

feed_dict = {
    training_data: mnist.test.images,
    labels: mnist.test.labels
}
test_accuracy = accuracy.eval(feed_dict=feed_dict)
print("Test accuracy: %g %%"%(test_accuracy*100))
sess.close()