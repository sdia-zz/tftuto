#!/usr/bin/env python
#-*- coding:utf-8 -*-




import tensorflow as tf




# TF operations are arranged into a graph of nodes
# Nodes take tensors as inputs
# ... one type of node is Constant :
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)


# to actually evaluate the nodes the graph must run in a session
sess = tf.Session()
print(sess.run([node1, node2]))


# ... operations are also nodes
node3 = tf.add(node1, node2)
print('node3: ', node3)
print('sess.run(node3): ', sess.run(node3))


# use placeholders to parameterize a graph
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b


# think of the parameterize graph as a function
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b:[2,4]}))


# create new node out of existing one ...
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b:4.5}))

c = tf.placeholder(tf.float32)
add_and_mul = adder_node * c
print(sess.run(add_and_triple, {a: 3, b:4.5, c:3}))


# now adding tranable parameters to a graph
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b


# constants are initialized at creation,
# to initialize variables do:
init = tf.global_variables_initializer()
sess.run(init)  # until this call variables are not initialized


print(sess.run(linear_model, {x: range(10)}))


# given an expected output vector (y), let's see how the model performs
# error, loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))


# how to reassign variables... and move to the perfect model
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1,2,3,4,5], y: [0,-1,-2,-3,-4]}))


# introducing optimizer: how to slowly move variables to minimize the loss
optimizer = tf.train.GradientDescentOptimizer(.01)
train = optimizer.minimize(loss)

sess.run(init)   # reset values to incorrect defaults (before the reassign)
for i in range(1000):
    print(sess.run([W, b]))
    sess.run(train,{x:[1,2,3,4], y:[0,-1,-2,-3]})
  

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss],{x:[1,2,3,4], y:[0,-1,-2,-3]})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


######################################################
# abstract previous workflow using tf.contrib.learn ##
######################################################

import tensorflow as tf
import numpy as np

features = [tf.contrib.layers.real_valued_column('x', dimension=1)]
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
x = np.array([1.,2.,3.,4.])
y = np.array([0.,-1,-2.,-3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({'x':x}, y, batch_size=4, num_epochs=1000)
estimator.fit(input_fn=input_fn, steps=1000)
print(estimator.evaluate(input_fn=input_fn))


# ... custom model using tf.contrib.learn
import numpy as np
import tensorflow as tf

def model(features, labels, mode):
    W = tf.get_variable('W', [1], dtype=tf.float64)
    b = tf.get_variable('b', [1], dtype=tf.float64)
    y = W * features['x'] + b

    loss = tf.reduce_sum(tf.square(y - labels))

    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
x = np.array([1.,2.,3.,4.])
y = np.array([0.,-1.,-2.,-3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({'x': x}, y, 4, num_epochs=1000)
estimator.fit(input_fn=input_fn, steps=1000)
print(estimator.evaluate(input_fn=input_fn, steps=10))
