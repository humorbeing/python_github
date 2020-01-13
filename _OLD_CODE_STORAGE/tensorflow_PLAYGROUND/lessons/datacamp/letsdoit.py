import tensorflow_playground as tf
import os
import skimage 
# Initialize two constants
# for element wise multiply
# x1 = tf.constant([1,2,3,4])
# x2 = tf.constant([5,6,7,8])
# Initialize two constants for matmul
# need a [1,4] and a [4,1]
x1 = tf.constant([[1,2,3,4]])
x2 = tf.constant([[5],[6],[7],[8]])

# Multiply
# result = tf.multiply(x1, x2)
result = tf.matmul(x1,x2)
print(result)

sess = tf.Session()

print(sess.run(result))

sess.close()

with tf.Session() as sess:
    output = sess.run(result)
    print(output)

# with tf.matmul(x1, x2) as result:
#     print(result)

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/media/ray/SSD/workspace/python/dataset/original/traffic_sign"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

images, labels = load_data(train_data_directory)

# Print the `images` dimensions
print(images.ndim)

# Print the number of `images`'s elements
print(images.size)

# Print the first instance of `images`
images[0]