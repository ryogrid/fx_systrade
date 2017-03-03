import tensorflow as tf
import numpy as np
import random


num_of_input_nodes = 1
num_of_hidden_nodes = 80
num_of_output_nodes = 1
length_of_sequences = 10
num_of_training_epochs = 5000
size_of_mini_batch = 100
num_of_prediction_epochs = 100
learning_rate = 0.01
forget_bias = 0.8
num_of_sample = 1000


def get_batch(batch_size, X, t):
    rnum = [random.randint(0, len(X) - 1) for x in range(batch_size)]
    xs = np.array([[[y] for y in list(X[r])] for r in rnum])
    ts = np.array([[t[r]] for r in rnum])
    return xs, ts


def create_data(nb_of_samples, sequence_len):
    X = np.zeros((nb_of_samples, sequence_len))
    for row_idx in range(nb_of_samples):
        X[row_idx, :] = np.around(np.random.rand(sequence_len)).astype(int)
    # Create the targets for each sequence
    t = np.sum(X, axis=1)
    return X, t


def make_prediction(nb_of_samples):
    sequence_len = 10
    xs, ts = create_data(nb_of_samples, sequence_len)
    return np.array([[[y] for y in x] for x in xs]), np.array([[x] for x in ts])


def inference(input_ph, istate_ph):
    with tf.name_scope("inference") as scope:
        weight1_var = tf.Variable(tf.truncated_normal(
            [num_of_input_nodes, num_of_hidden_nodes], stddev=0.1), name="weight1")
        weight2_var = tf.Variable(tf.truncated_normal(
            [num_of_hidden_nodes, num_of_output_nodes], stddev=0.1), name="weight2")
        bias1_var = tf.Variable(tf.truncated_normal([num_of_hidden_nodes], stddev=0.1), name="bias1")
        bias2_var = tf.Variable(tf.truncated_normal([num_of_output_nodes], stddev=0.1), name="bias2")

        in1 = tf.transpose(input_ph, [1, 0, 2])
        in2 = tf.reshape(in1, [-1, num_of_input_nodes])
        in3 = tf.matmul(in2, weight1_var) + bias1_var
        in4 = tf.split(0, length_of_sequences, in3)

        cell = tf.nn.rnn_cell.BasicLSTMCell(
            num_of_hidden_nodes, forget_bias=forget_bias, state_is_tuple=False)
        rnn_output, states_op = tf.nn.rnn(cell, in4, initial_state=istate_ph)
        output_op = tf.matmul(rnn_output[-1], weight2_var) + bias2_var

        # Add summary ops to collect data
        w1_hist = tf.histogram_summary("weights1", weight1_var)
        w2_hist = tf.histogram_summary("weights2", weight2_var)
        b1_hist = tf.histogram_summary("biases1", bias1_var)
        b2_hist = tf.histogram_summary("biases2", bias2_var)
        output_hist = tf.histogram_summary("output",  output_op)
        results = [weight1_var, weight2_var, bias1_var,  bias2_var]
        return output_op, states_op, results


def loss(output_op, supervisor_ph):
    with tf.name_scope("loss") as scope:
        square_error = tf.reduce_mean(tf.square(output_op - supervisor_ph))
        loss_op = square_error
        tf.scalar_summary("loss", loss_op)
        return loss_op


def training(loss_op):
    with tf.name_scope("training") as scope:
        training_op = optimizer.minimize(loss_op)
        return training_op


def calc_accuracy(output_op, prints=False):
    inputs, ts = make_prediction(num_of_prediction_epochs)
    pred_dict = {
        input_ph:  inputs,
        supervisor_ph: ts,
        istate_ph:    np.zeros((num_of_prediction_epochs, num_of_hidden_nodes * 2)),
    }
    output = sess.run([output_op], feed_dict=pred_dict)

    def print_result(i, p, q):
        [print(list(x)[0]) for x in i]
        print("output: %f, correct: %d" % (p, q))
    if prints:
        [print_result(i, p, q) for i, p, q in zip(inputs, output[0], ts)]

    opt = abs(output - ts)[0]
    total = sum([1 if x[0] < 0.05 else 0 for x in opt])
    print("accuracy %f" % (total / float(len(ts))))
    return output

random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

X, t = create_data(num_of_sample, length_of_sequences)

with tf.Graph().as_default():
    input_ph = tf.placeholder(tf.float32, [None, length_of_sequences, num_of_input_nodes], name="input")
    supervisor_ph = tf.placeholder(tf.float32, [None, num_of_output_nodes], name="supervisor")
    istate_ph = tf.placeholder(tf.float32, [None, num_of_hidden_nodes * 2], name="istate")

    output_op, states_op, datas_op = inference(input_ph, istate_ph)
    loss_op = loss(output_op, supervisor_ph)
    training_op = training(loss_op)

    summary_op = tf.merge_all_summaries()
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        summary_writer = tf.train.SummaryWriter("/tmp/tensorflow_log", graph=sess.graph)
        sess.run(init)

        for epoch in range(num_of_training_epochs):
            inputs, supervisors = get_batch(size_of_mini_batch, X, t)
            train_dict = {
                input_ph:      inputs,
                supervisor_ph: supervisors,
                istate_ph:     np.zeros((size_of_mini_batch, num_of_hidden_nodes * 2)),
            }
            sess.run(training_op, feed_dict=train_dict)

            if (epoch) % 100 == 0:
                summary_str, train_loss = sess.run([summary_op, loss_op], feed_dict=train_dict)
                print("train#%d, train loss: %e" % (epoch, train_loss))
                summary_writer.add_summary(summary_str, epoch)
                if (epoch) % 500 == 0:
                    calc_accuracy(output_op)

        calc_accuracy(output_op, prints=True)
        datas = sess.run(datas_op)
        saver.save(sess, "model.ckpt")
