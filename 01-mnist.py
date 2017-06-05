#!/usr/bin/env python
#-*- coding:utf-8 -*-


# MNIST: Modified National Institute of Standards and Technology
# https://en.wikipedia.org/wiki/MNIST_database

# mnist data http://yann.lecun.com/exdb/mnist/



# loading the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



# SoftMax gives a list of values between 0 and 1 that add up to 1 !




import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# one line for the model definition
y = tf.nn.softmax(tf.matmul(x, W) + b)


# using cross entropy to implement the loss
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(.5).minimize(cross_entropy)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# batch of 100 makes it the Gradient Descent becomes Stochastic
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})


# model evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
