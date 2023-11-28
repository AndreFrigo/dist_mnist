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
flags.DEFINE_integer("train_steps", 300,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean(
    "sync_replicas", True,
    "Use the sync_replicas (synchronized replicas) mode, "
    "wherein the parameter updates from workers are aggregated "
    "before applied to avoid stale gradients")
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






# We'll use 3 hidden layers. The number of hidden layers is a trade off between speed, cost and accuracy.
# After the output layer, to evaluate the errors between the predictions and the labels, we'll use a loss (cost)
# function.
# Here, we'll just check how many classes we have correctly predicted. We apply an optimizer (Adam) to reduce
# the cost/error at each epoch.
def multilayer_perceptron(x, weights, biases):
    """
    3-layer perceptron for the MNIST dataset.

    :param x: Placeholder for the data input
    :param weights: dict of weights
    :param biases: dict of bias values
    :return: the output layer
    """
    # first hidden layer with RELU activation function
    # X * W + B
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # RELU(X * W + B) -> f(x) = max(0, x)
    layer_1 = tf.nn.relu(layer_1)

    # second hidden layer with RELU activation function
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # third hidden layer with RELU activation function
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # output layer
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']

    return out_layer


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

  # Define some parameters
  learning_rate = 0.001
  training_epochs = 100  # how many training cycles we'll go through
  batch_size = 100  # size of the batches of the training data
  
  n_classes = 10  # number of classes for the output (-> digits from 0 to 9)
  n_samples = mnist.train.num_examples  # number of samples (55 000)
  n_input = 784  # shape of one input (array of 784 floats)
  
  n_hidden_1 = 256  # number of neurons for the 1st hidden layer. 256 is common because of the 8-bit color storing method
  n_hidden_2 = 256  # number of neurons for the 2nd hidden layer
  n_hidden_3 = 256  # number of neurons for the 3rd hidden layer

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
  with tf.device(
      tf.train.replica_device_setter(
          worker_device=worker_device,
          ps_device="/job:ps/cpu:0",
          cluster=cluster)):
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # # Variables of the hidden layer
    # hid_w = tf.Variable(
    #     tf.truncated_normal(
    #         [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
    #         stddev=1.0 / IMAGE_PIXELS),
    #     name="hid_w")
    # hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

    # # Variables of the softmax layer
    # sm_w = tf.Variable(
    #     tf.truncated_normal(
    #         [FLAGS.hidden_units, 10],
    #         stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
    #     name="sm_w")
    # sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

    # # Ops: located on the worker specified with FLAGS.task_index
    # x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    # y_ = tf.placeholder(tf.float32, [None, 10])

    # define the weights for the nodes of each layer : 784 weights for each node in the first layer,
    # then 256 for the 2 next layers, then 10 for the output layer
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),  # matrix of normally distributed random values for H1.
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
    }

    # define the biases for each nodes in each layer : 1 bias per node
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # placeholders for the input data & the labels
    x = tf.placeholder('float', [None, n_input])
    y = tf.placeholder('float', [None, n_classes])


    pred = multilayer_perceptron(x, weights, biases)

    # Define costs and optimization functions. We'll use tf built-in functions
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)




    # hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    # hid = tf.nn.relu(hid_lin)

    # y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    # correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

    if FLAGS.sync_replicas:
      replicas_to_aggregate = num_workers
      opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=replicas_to_aggregate,
          total_num_replicas=num_workers,
          name="mnist_sync_replicas")

    # train_step = opt.minimize(cross_entropy, global_step=global_step)

    if FLAGS.sync_replicas:
      local_init_op = opt.local_step_init_op
      if is_chief:
        local_init_op = opt.chief_init_op

      ready_for_local_init_op = opt.ready_for_local_init_op

      # Initial token and chief queue runners required by the sync_replicas mode
      chief_queue_runner = opt.get_chief_queue_runner()
      sync_init_op = opt.get_init_tokens_op()




    init_op = tf.global_variables_initializer()
    train_dir = tempfile.mkdtemp()

    if FLAGS.sync_replicas:
      sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir=train_dir,
          init_op=init_op,
          local_init_op=local_init_op,
          ready_for_local_init_op=ready_for_local_init_op,
          recovery_wait_secs=1,
          global_step=global_step)
    else:
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
       
    
    # 'training_epochs' training cycles
    for epoch in range(training_epochs):

        # Cost
        avg_cost = 0.0

        total_batch = int(n_samples/batch_size)

        for i in range(total_batch):

            batch_x, batch_y = mnist.train.next_batch(batch_size)

            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

            avg_cost += c/total_batch

        print("Epoch : {} -> cost : {:.4f}".format(epoch, avg_cost))

    print("Model has completed {} Epochs of training".format(training_epochs))

    if is_chief:
        # Model evaluations
        correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # Tensor of bool
        correct_predictions = tf.cast(correct_predictions, 'float')  # Cast it to a tensor of floats sor that we can get the
        # mean

        accuracy = tf.reduce_mean(correct_predictions)

        # We now evaluate this accuracy on the test dataset
        print('\nTest Dataset accuracy: {:.4f}'.format(accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))


    
if __name__ == "__main__":
  tf.app.run()
