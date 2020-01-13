import tensorflow as tf
# import torch
# create a computational graph
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1,node2)
# run a computational graph
# sess = tf.Session()
# print(sess.run([node1,node2]))
# sess.close()

with tf.Session() as sess:
    output = sess.run([node1,node2])
    print(output)

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
this_path = '/media/ray/SSD/workspace/python/tensorflow_playground'
with tf.Session() as sess:
    File_Writer = tf.summary.FileWriter(this_path+'/graph', sess.graph)
    # $ tensorboard --logdir 'this_folder_path'
    # localhost:6006
    output = sess.run(c)
    print(output)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

with tf.Session() as sess:
    # output = sess.run(adder_node)
    # wrong. need to pass value to a and b
    output = sess.run(adder_node, {a:[1, 3], b:[2, 4]})
    print(output)

W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)
x = tf.placeholder(tf.float32)


linear_model = W * x + b

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    output = sess.run(linear_model,
                      {x: [1,2,3,4]})
    print(output)


W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)
x = tf.placeholder(tf.float32)


linear_model = W * x + b


y = tf.placeholder(tf.float32)
square_error = linear_model - y
square_error = tf.square(square_error)
loss = tf.reduce_sum(square_error)

init = tf.global_variables_initializer()

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
    # output = sess.run(loss,
    #          {x:[1,2,3,4],y:[0,-1,-2,-3]})
    # print(output)
        sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
    print(sess.run([W, b]))