"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee

Orthogonal RNN based on https://arxiv.org/abs/1612.00188
Efficient Orthogonal Parametrisation of Recurrent Neural Networks Using Householder Reflections
Zakaria Mhammedi, Andrew Hellicar, Ashfaqur Rahman, James Bailey, 2016
"""

import tensorflow as tf
import numpy as np
import stimulus
import time
import analysis
from parameters import *

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

"""
Model setup and execution
"""

class Model:

    def __init__(self, input_data, target_data, mask):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=1)
        self.target_data = tf.unstack(target_data, axis=1)
        self.mask = tf.unstack(mask, axis=0)

        # Load the initial hidden state activity to be used at the start of each trial
        self.hidden_init = tf.constant(par['h_init'])

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):

        """
        Run the reccurent network
        History of hidden state activity stored in self.hidden_state_hist
        """
        self.rnn_cell_loop(self.input_data, self.hidden_init)

        with tf.variable_scope('output'):
            W_out = tf.get_variable('W_out', initializer = par['w_out0'], trainable=True)
            b_out = tf.get_variable('b_out', initializer = par['b_out0'], trainable=True)

        """
        Network output
        Only use excitatory projections from the RNN to the output layer
        """
        self.y_hat = [tf.matmul(W_out,h)+b_out for h in self.hidden_state_hist]


    def rnn_cell_loop(self, x_unstacked, h):

        """
        Initialize weights and biases
        """
        with tf.variable_scope('rnn_cell'):
            W_in = tf.get_variable('W_in', initializer = par['w_in0'], trainable=False)
            U = tf.get_variable('U', initializer = par['u0'], trainable=True)
            b_rnn = tf.get_variable('b_rnn', initializer = par['b_rnn0'], trainable=True)

        self.hidden_state_hist = []

        """
        Loop through the neural inputs to the RNN, indexed in time
        """
        for rnn_input in x_unstacked:
            h = self.rnn_cell(rnn_input, h)
            self.hidden_state_hist.append(h)


    def rnn_cell(self, rnn_input, h):

        """
        Main computation of the recurrent network
        """
        with tf.variable_scope('rnn_cell', reuse=True):
            W_in = tf.get_variable('W_in')
            U = tf.get_variable('U')
            b_rnn = tf.get_variable('b_rnn')

        U1 = U/tf.norm(U, axis=0)

        TU = tf.constant(par['triu'])
        D = tf.constant(par['diag'])

        #print('TU', TU)
        #print('D', D)
        #print('U', U)
        # Calculating  h - U T^{-1} U h, where T = triu(U^TU, 1) + 1/2 diag(U^TU).

        UU = tf.matmul(tf.transpose(U1), U1)
        #print('UU', UU)
        T = tf.matrix_inverse(TU*UU + D*UU/2)
        self.WY = tf.eye(par['n_hidden'], dtype=tf.float32) - tf.matmul(U1, tf.matmul(T, tf.transpose(U1)))

        if par['n_reflect'] < par['n_hidden']:
            """
            h = tf.nn.relu(h*(1-par['alpha_neuron']) \
                + par['alpha_neuron']*(tf.matmul(W_in, rnn_input) + tf.matmul(self.WY, h) + b_rnn) \
                + tf.random_normal([par['n_hidden'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float32))
            """
            h = tf.nn.relu(tf.matmul(W_in, rnn_input) + tf.matmul(self.WY, h) + b_rnn \
                + tf.random_normal([par['n_hidden'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float32))

        else:
            print('Not supported!')
            quit()

        return h


    def optimize(self):

        """
        Calculate the loss functions and optimize the weights
        """
        #perf_loss = [mask*tf.reduce_mean(tf.square(y_hat-desired_output),axis=0)
        #             for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]

        """
        cross_entropy
        """
        perf_loss = [mask*tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = desired_output, dim=0) \
                for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]


        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        spike_loss = [par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for h in self.hidden_state_hist]

        self.perf_loss = tf.reduce_mean(tf.stack(perf_loss, axis=0))
        self.spike_loss = tf.reduce_mean(tf.stack(spike_loss, axis=0))

        #self.wiring_loss = 0.5*tf.reduce_mean(tf.square(self.WY))


        self.loss = self.perf_loss + self.spike_loss

        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        grads_and_vars = opt.compute_gradients(self.loss)

        """
        Apply any applicable weights masks to the gradient and clip
        """
        capped_gvs = []
        for grad, var in grads_and_vars:

            if var.name == "rnn_cell/U:0":
                grad *= par['u_mask']
                print('Applied weight mask to U.')
            """
            elif var.name == "output/W_out:0":
                grad *= par['w_out_mask']
                print('Applied weight mask to w_out.')
            """
            if not str(type(grad)) == "<class 'NoneType'>":
                capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))

        self.train_op = opt.apply_gradients(capped_gvs)


def train_and_analyze():

    """
    Train the network model given the paramaters, then re-run the model at finer
    temporal resoultion and with more trials, and then analyze the model results.
    Paramaters used for analysis purposes found in analysis_par.
    """

    main()
    update_parameters(analysis_par)
    tf.reset_default_graph()
    main()


    update_parameters(revert_analysis_par)


def main():

    """
    Reset TensorFlow before running anything
    """
    tf.reset_default_graph()

    """
    Create the stimulus class to generate trial paramaters and input activity
    """
    stim = stimulus.Stimulus()

    n_input, n_hidden, n_output = par['shape']
    N = par['batch_train_size'] * par['num_batches'] # trials per iteration, calculate gradients after batch_train_size

    """
    Define all placeholder
    """
    mask = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_train_size']])
    x = tf.placeholder(tf.float32, shape=[n_input, par['num_time_steps'], par['batch_train_size']])  # input data
    y = tf.placeholder(tf.float32, shape=[n_output, par['num_time_steps'], par['batch_train_size']]) # target data


    # enter "config=tf.ConfigProto(log_device_placement=True)" inside Session to check whether CPU/GPU in use
    with tf.Session() as sess:

        #with tf.device("/gpu:0"):
        model = Model(x, y, mask)
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()

        saver = tf.train.Saver()
        # Restore variables from previous model if desired
        if par['load_previous_model']:
            saver.restore(sess, par['save_dir'] + par['ckpt_load_fn'])
            print('Model ' +  par['ckpt_load_fn'] + ' restored.')

        # keep track of the model performance across training
        model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], 'trial': [], 'time': []}

        for i in range(par['num_iterations']):

            # generate batch of N (batch_train_size X num_batches) trials
            trial_info = stim.generate_trial()

            # keep track of the model performance for this batch
            loss = np.zeros((par['num_batches']))
            perf_loss = np.zeros((par['num_batches']))
            spike_loss = np.zeros((par['num_batches']))
            accuracy = np.zeros((par['num_batches']))

            for j in range(par['num_batches']):

                """
                Select batches of size batch_train_size
                """
                ind = range(j*par['batch_train_size'],(j+1)*par['batch_train_size'])
                target_data = trial_info['desired_output'][:,:,ind]
                input_data = trial_info['neural_input'][:,:,ind]
                train_mask = trial_info['train_mask'][:,ind]

                """
                Run the model
                If learning rate > 0, then also run the optimizer;
                if learning rate = 0, then skip optimizer
                """
                if par['learning_rate']>0:
                    _, loss[j], perf_loss[j], spike_loss[j], y_hat, state_hist, W_rnn,  = \
                        sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, model.y_hat, \
                        model.hidden_state_hist, model.WY], {x: input_data, y: target_data, mask: train_mask})
                else:
                    loss[j], perf_loss[j], spike_loss[j], y_hat, state_hist, W_rnn = \
                        sess.run([model.loss, model.perf_loss, model.spike_loss, model.y_hat, model.hidden_state_hist, \
                        model.WY], {x: input_data, y: target_data, mask: train_mask})

                accuracy[j] = analysis.get_perf(target_data, y_hat, train_mask)

            iteration_time = time.time() - t_start
            model_performance = append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, (i+1)*N, iteration_time)

            """
            Save the network model and output model performance to screen
            """
            if (i+1)%par['iters_between_outputs']==0 or i+1==par['num_iterations']:
                print_results(i, N, iteration_time, perf_loss, spike_loss, state_hist, accuracy)
                save_path = saver.save(sess, par['save_dir'] + par['ckpt_save_fn'])

        """
        Analyze the network model and save the results
        """
        if par['analyze_model']:
            weights = eval_weights(W_rnn)
            analysis.analyze_model(trial_info, y_hat, state_hist, model_performance, weights)

def append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, trial_num, iteration_time):

    model_performance['accuracy'].append(np.mean(accuracy))
    model_performance['loss'].append(np.mean(loss))
    model_performance['perf_loss'].append(np.mean(perf_loss))
    model_performance['spike_loss'].append(np.mean(spike_loss))
    model_performance['trial'].append(trial_num)
    model_performance['time'].append(iteration_time)

    return model_performance

def eval_weights(W_rnn):

    with tf.variable_scope('rnn_cell', reuse=True):
        W_in = tf.get_variable('W_in')
        U = tf.get_variable('U')
        b_rnn = tf.get_variable('b_rnn')

    with tf.variable_scope('output', reuse=True):
        W_out = tf.get_variable('W_out')
        b_out = tf.get_variable('b_out')

    weights = {
        'w_in'  : W_in.eval(),
        'w_rnn' : W_rnn,
        'U' : U.eval(),
        'w_out' : W_out.eval(),
        'b_rnn' : b_rnn.eval(),
        'b_out'  : b_out.eval()
    }

    return weights

def print_results(iter_num, trials_per_iter, iteration_time, perf_loss, spike_loss, state_hist, accuracy):

    print('Trial {:7d}'.format((iter_num+1)*trials_per_iter) + ' | Time {:0.2f} s'.format(iteration_time) +
      ' | Perf loss {:0.4f}'.format(np.mean(perf_loss)) + ' | Spike loss {:0.4f}'.format(np.mean(spike_loss)) +
      ' | Mean activity {:0.4f}'.format(np.mean(state_hist)) + ' | Accuracy {:0.4f}'.format(np.mean(accuracy)))
