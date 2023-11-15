# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Distributed MNIST training and validation, with model replicas.

A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on one parameter server (ps), while the ops
are executed on two worker nodes by default. The TF sessions also run on the
worker node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.

The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""

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

#number of CPU nodes (same as the number of k8s computing nodes)
nodes = 2
cpustring = "localhost:2223"
for i in range(1,nodes):
  cpustring += ",localhost:"+str(2223+i)

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 0, "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 20000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", cpustring,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = flags.FLAGS

IMAGE_PIXELS = 28

# Example:
#   cluster = {'ps': ['host1:2222', 'host2:2222'],
#              'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
#   os.environ['TF_CONFIG'] = json.dumps(
#       {'cluster': cluster,
#        'task': {'type': 'worker', 'index': 1}})

def main(unused_argv):

  time_start = time.time()
  # Parse environment variable TF_CONFIG to get job_name and task_index

  # If not explicitly specified in the constructor and the TF_CONFIG
  # environment variable is present, load cluster_spec from TF_CONFIG.
  tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
  task_config = tf_config.get('task', {})
  task_type = task_config.get('type')
  task_index = task_config.get('index')

  FLAGS.job_name = task_type
  FLAGS.task_index = task_index

  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

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

  cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})

  if not FLAGS.existing_servers:
    # Not using existing servers. Create an in-process server.
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
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
  with tf.device(
      tf.train.replica_device_setter(
          worker_device=worker_device,
          ps_device="/job:ps/cpu:0",
          cluster=cluster)):
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Variables of the hidden layer
    hid_w = tf.Variable(
        tf.truncated_normal(
            [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
            stddev=1.0 / IMAGE_PIXELS),
        name="hid_w")
    hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

    # Variables of the softmax layer
    sm_w = tf.Variable(
        tf.truncated_normal(
            [FLAGS.hidden_units, 10],
            stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
        name="sm_w")
    sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

    # Ops: located on the worker specified with FLAGS.task_index
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, [None, 10])

    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)

    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)


    train_step = opt.minimize(cross_entropy, global_step=global_step)

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

    if FLAGS.existing_servers:
      server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
      print("Using existing server at: %s" % server_grpc_url)

      sess = sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config)
    else:
      sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    print("Worker %d: Session initialization complete." % FLAGS.task_index)
       
    
    # Perform training
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)
    print("Nodes="+str(nodes))
    local_step = 0
    #control to print data one time every 1000 steps
    hasPrinted = False
    while True:
      # Training feed
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
      train_feed = {x: batch_xs, y_: batch_ys}

      _, step = sess.run([train_step, global_step], feed_dict=train_feed)
      local_step += 1
      now = time.time()
      if (step % 1000 <= 200 and not hasPrinted and step >= 1000):
        print("time: %f, step: %d" % (now, step-(step%1000)))
        hasPrinted = True
      if (step % 1000 > 200):
        hasPrinted = False
      if step >= FLAGS.train_steps:
        break

    time_end = time.time()
    print("Training ends @ %f" % time_end)
    print("Steps: %d, Batch size: %d" % (FLAGS.train_steps, FLAGS.batch_size))
    training_time = time_end - time_begin
    starting_time = time_begin - time_start
    print("Starting time (time lost before starting training): %f s" % starting_time)
    print("Training elapsed time: %f s" % training_time)
    
   
    # Validation feed
    val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    val_acc = sess.run(accuracy, feed_dict=val_feed)
    print("Validation accuracy = %g" %
          (val_acc*100))


    # Test feed
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}
    test_acc = sess.run(accuracy, feed_dict=test_feed)
    print("Test accuracy = %g" %
          (test_acc*100))



if __name__ == "__main__":
  tf.app.run()
