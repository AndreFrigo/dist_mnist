from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import sys
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#number of nodes
nodes = 5

#setup for parallelization with kubeflow
cpustring = "localhost:2223"
for i in range(1,nodes):
    cpustring += ",localhost:"+str(2223+i)

flags = tf.app.flags
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 0, "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("train_steps", 50,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 10e-4, "Learning rate")
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", cpustring,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = flags.FLAGS

IMAGE_PIXELS = 28   



time_start = time.time()
tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
task_config = tf_config.get('task', {})
task_type = task_config.get('type')
task_index = task_config.get('index')

FLAGS.job_name = task_type
FLAGS.task_index = task_index


mnist = input_data.read_data_sets("data/", one_hot=True) 

if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")

print("job name = %s" % FLAGS.job_name)
print("task index = %d" % FLAGS.task_index)

cluster_config = tf_config.get('cluster', {})
ps_hosts = cluster_config.get('ps')
worker_hosts = cluster_config.get('worker')
ps_hosts_str = ','.join(ps_hosts)
worker_hosts_str = ','.join(worker_hosts)

FLAGS.ps_hosts = ps_hosts_str
FLAGS.worker_hosts = worker_hosts_str

# Construct the cluster and start the server
ps_spec = FLAGS.ps_hosts.split(",")
worker_spec = FLAGS.worker_hosts.split(",")

# Get the number of workers.
num_workers = len(worker_spec)
print("Step to execute: "+str(FLAGS.train_steps))

cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})


# Not using existing servers. Create an in-process server.
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
if FLAGS.job_name == "ps":
    server.join()

is_chief = (FLAGS.task_index == 0)
if FLAGS.num_gpus > 0:
    # Avoid gpu allocation conflict: now allocate task_num -> #gpu
    # for each worker in the corresponding machine
    gpu = (FLAGS.task_index % FLAGS.num_gpus)
    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
elif FLAGS.num_gpus == 0:
    # Just allocate the CPU to worker server
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    # The ps use CPU and workers use corresponding GPU

print("WORKING DEVICE: "+worker_device)

""" Initialize weight """
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

""" Initialize bias """
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

""" Define convolution """
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

""" Define pooling """
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.device(tf.train.replica_device_setter(
        worker_device=worker_device,
        ps_device="/job:ps/cpu:0",
        cluster=cluster)):

    global_step = tf.Variable(0, name="global_step", trainable=False)


    """ Define placeholder: Where the data will be placed.
    Create a two-dimensional tensor for images and correct labels.
    None means no limits in length. """
    # Placeholder for image data
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    # Placeholder for a correct answer label
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Redefine input data with 4D tensor.
    # The second/third parameter specifies the width/height of the image.
    # Since it is a monochrome image, the number of color channels of the last parameter is 1.
    x_image = tf.reshape(x, [-1,28,28,1])

    """ Define first convolutional layer """
    # Weight tensor definition (patch size, patch size, input channel, output channel)
    # Use 32 features (kernel, filter) with a 5x5 window (also called patch) size.
    # Since the image is monochrome, the input channel is 1.
    W_conv1 = weight_variable([5, 5, 1, 32])
    # Define bias tensor
    b_conv1 = bias_variable([32])
    # Apply convolution to the x_image and the weight tensor, add the bias, and then apply ReLU function.
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # Apply max pooling to get an output value.
    h_pool1 = max_pool_2x2(h_conv1)

    """ Define second convolutional layer """
    # Define weight tensor (patch size, patch size, input channel, output channel)
    # Use 64 features with a 5x5 window (also called patch) size.
    # The size of the output channel of the previous layer is 32, which is the input channel here.
    W_conv2 = weight_variable([5, 5, 32, 64])
    # Define bias tensor
    b_conv2 = bias_variable([64])
    # Apply convolution to the h_pool1 output value of the first convolutional layer and the weight tensor, add the bias, and then apply ReLU function.
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # Apply max pooling to get an output value.
    h_pool2 = max_pool_2x2(h_conv2)

    """ Define fully-connected layer """
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    """ Define dropout """
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    """ Define final softmax hierarchy """
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    # Model training and evaluation
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy, global_step=global_step)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.global_variables_initializer()
    train_dir = tempfile.mkdtemp()
    sv = tf.train.Supervisor(
        is_chief=is_chief,
        logdir=train_dir,
        init_op=init_op,
        recovery_wait_secs=1,
        global_step=global_step)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps",
                        "/job:worker/task:%d" % FLAGS.task_index])

    # The chief worker (task_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
      print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
      print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)

    
    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    print("Worker %d: Session initialization complete." % FLAGS.task_index)

    # Perform training
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)
    print("Nodes="+str(nodes))
    while True:
        batch = mnist.train.next_batch(FLAGS.batch_size)
        _, step = sess.run([train_step, global_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        now = time.time()
        if step%10==0:
            print("time: %f, step: %d" % (now, step-(step%10)))
        if step >= FLAGS.train_steps: break
        

    time_end = time.time()
    print("Training ends @ %f" % time_end)
    print("Steps: %d, Batch size: %d" % (FLAGS.train_steps, FLAGS.batch_size))
    
    training_time = time_end - time_begin
    starting_time = time_begin - time_start
    print("Starting time (time lost before starting training): %f s" % starting_time)
    print("Training elapsed time: %f s" % training_time)

    if is_chief:
        # # Validation feed
        # val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # val_acc = sess.run(accuracy, feed_dict=val_feed)
        # print("Validation accuracy = %g" % (val_acc*100))


        # # Test feed
        # test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        # test_acc = sess.run(accuracy, feed_dict=test_feed)
        # print("Test accuracy = %g" % (test_acc*100))
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print("Test accuracy = %g" % (test_acc * 100))


    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())

    #     # Sample every 100 pieces of data to perform learning 2000 times.
    #     for i in range(nsteps):
    #         batch = mnist.train.next_batch(100)
    #         if i % 100 == 0:
    #             # train_accuracy = accuracy.eval(feed_dict={
    #             #     x: batch[0], y_: batch[1], keep_prob: 1.0})
    #             # print 'step %d, training accuracy %g' % (i, train_accuracy)
    #             print("Step "+str(i))
    #         train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    #     print 'test accuracy %g' % accuracy.eval(feed_dict={
    #         x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
