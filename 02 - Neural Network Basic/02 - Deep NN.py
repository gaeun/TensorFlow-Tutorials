import tensorflow as tf
import numpy as np

# Read data from file
data = np.loadtxt('./data.csv', delimiter=',',
                  unpack=True, dtype='float32')

# Label 0th and 1st rows (features) as x_data
# and the remaining rows (classifications) as y_data
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])


#########
# Create a neural network model
######
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Create 2 hidden layers with 3 weighted matrix variables
# The hidden layers will each have 10 and 20 neurons, and they will be connected as follows:
# 2 -> 10 -> 20 -> 3
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))

# Generate first layers with input x and variable W1
# Generate second layer with first layer and variable W2
L1 = tf.nn.relu(tf.matmul(X, W1))
L2 = tf.nn.relu(tf.matmul(L1, W2))

# Multiply by W3 to generate output
model = tf.matmul(L2, W3)

# Tensorflow has a built in function called softmax_cross_entropy_with_logits that makes our lives easier
# We can set up the cost function without coding complicated equations
cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(model, Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)


#########
# Train the neural network
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in xrange(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print (step + 1), sess.run(cost, feed_dict={X: x_data, Y: y_data})


#########
# Check results
# 0: etc. 1: mammal, 2: bird
######
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print 'Prediction:', sess.run(prediction, feed_dict={X: x_data})
print 'Actual:', sess.run(target, feed_dict={Y: y_data})

check_prediction = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print 'Accuracy: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data})
