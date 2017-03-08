# This module contains the tensorflow model for DCM-RNN.
import tensorflow as tf
import numpy as np
from DCM_RNN.toolboxes import Initialization


def reset_interactive_sesssion(isess):
    try:
        isess
    except:
        isess = tf.InteractiveSession()
    else:
        isess.close()
        isess = tf.InteractiveSession()


class DcmRnn(Initialization):
    def __init__(self,
                 n_recurrent_step=None,
                 learning_rate=None,
                 variable_scope_name_x_parameter=None,
                 variable_scope_name_x_initial=None,
                 variable_scope_name_x=None,
                 variable_scope_name_x_final=None,
                 variable_scope_name_x_stacked=None,
                 variable_scope_name_h_parameter=None,
                 variable_scope_name_h_initial=None,
                 variable_scope_name_h=None,
                 variable_scope_name_h_final=None,
                 variable_scope_name_h_stacked=None,
                 variable_scope_name_y=None,
                 variable_scope_name_y_stacked=None,
                 variable_scope_name_loss=None,
                 log_directory=None):
        Initialization.__init__(self)
        self.n_recurrent_step = n_recurrent_step or 12
        self.learning_rate = learning_rate or 0.005

        self.variable_scope_name_x_parameter = variable_scope_name_x_parameter or 'para_x'
        self.variable_scope_name_x_initial = variable_scope_name_x_initial or 'cell_x_initial'
        self.variable_scope_name_x = variable_scope_name_x or 'cell_x'
        self.variable_scope_name_x_final = variable_scope_name_x_final or 'cell_x_final'
        self.variable_scope_name_x_stacked = variable_scope_name_x_stacked or 'x_stacked'

        self.variable_scope_name_h_parameter = variable_scope_name_h_parameter or 'para_h'
        self.variable_scope_name_h_initial = variable_scope_name_h_initial or 'cell_h_initial'
        self.variable_scope_name_h = variable_scope_name_h or 'cell_h'
        self.variable_scope_name_h_final = variable_scope_name_h_final or 'cell_h_final'
        self.variable_scope_name_h_stacked = variable_scope_name_h_stacked or 'h_stacked'

        self.variable_scope_name_y = variable_scope_name_y or 'cell_y'
        self.variable_scope_name_y_stacked = variable_scope_name_y_stacked or 'y_stacked'

        self.variable_scope_name_loss = variable_scope_name_loss or 'loss'

        self.log_directory = log_directory or './logs'

        self.set_up_hyperparameter_values()
        self.trainable_flags_h = {'alpha': True,
                                  'E0': True,
                                  'k': True,
                                  'gamma': True,
                                  'tao': True,
                                  'epsilon': False,
                                  'V0': False,
                                  'TE': False,
                                  'r0': False,
                                  'theta0': False,
                                  'x_h_coupling': False
                                  }

    def collect_parameters(self, du):
        """
        Collect needed parameters from a DataUnit instant.
        :param du: a DataUnit instant
        :return: a dictionary containing needed paramters
        """
        deliverables = {}
        needed_parameters = {'n_node', 'n_stimuli', 't_delta'}
        for para in needed_parameters:
            deliverables[para] = du.get(para)
        self.load_parameters(deliverables)
        return deliverables

    def load_parameters(self, parameter_package):
        self.n_region = parameter_package['n_node']
        self.t_delta = parameter_package['t_delta']
        self.n_stimuli = parameter_package['n_stimuli']

    def set_up_hyperparameter_values(self):
        hyperparameter_values = {}
        hyperparameter_values[self.variable_scope_name_x_parameter] = {'gradient': 1., 'sparse': 1., 'prior': 0.}
        hyperparameter_values[self.variable_scope_name_h_parameter] = {'gradient': 1., 'sparse': 0., 'prior': 1.}
        self.hyperparameter_values = hyperparameter_values

    def create_shared_variables_h(self, initial_values):
        """
        Create shared hemodynamic variables.
        :param initial_values: a pandas.DataFrame, containing the initial values of hemodynamic variables
        :return:
        """
        n_region, n_para = initial_values.shape
        hemodynamic_parameter = []
        with tf.variable_scope(self.variable_scope_name_h_parameter):
            for idx_r, region_label in enumerate(list(initial_values.index)):
                temp_list = []
                for para in initial_values.columns:
                    temp = tf.get_variable(para + '_r' + str(idx_r),
                                           dtype=tf.float32,
                                           initializer=initial_values[para][region_label],
                                           trainable=self.trainable_flags_h[para])
                    temp_list.append(temp)
                temp_tensor = tf.stack(temp_list, 0)
                hemodynamic_parameter.append(temp_tensor)
            hemodynamic_parameter = tf.stack(hemodynamic_parameter, 1, 'hemodynamic_parameter')
            hemodynamic_parameter = tf.transpose(hemodynamic_parameter)
        self.hemodynamic_parameter = hemodynamic_parameter
        return hemodynamic_parameter

    def create_placeholders(self):
        input_u = tf.placeholder(tf.float32, [self.n_stimuli, self.n_recurrent_step], name='input_u')
        input_y_true = tf.placeholder(tf.float32, [self.n_region, self.n_recurrent_step], name='input_y_true')
        return [input_u, input_y_true]

    def phi_h(self, h_state_current, alpha, E0):
        """
        Used to map hemodynamic states into higher dimension
        :param h_state_current:
        :param alpha:
        :param E0:
        :return:
        """
        h_state_augmented = []
        for i in range(4):
            h_state_augmented.append(h_state_current[i])
        h_state_augmented.append(tf.pow(h_state_current[2], tf.div(1., alpha)))
        h_state_augmented.append(tf.multiply(tf.div(h_state_current[3], h_state_current[2]), h_state_augmented[4]))
        tmp = tf.subtract(1., tf.pow(tf.subtract(1., E0), tf.div(1., h_state_current[1])))
        tmp = tf.multiply(tf.div(tmp, E0), h_state_current[1])
        h_state_augmented.append(tmp)
        return h_state_augmented

    def get_h_para_tensor_for_one_region(self, i_region):
        """
        Get hemodynamic parameter tensors for a particular region
        :param i_region:
        :return: a dictionary
        """
        output = {}
        for para in self.hemo_parameter_keys:
            output[para] = tf.get_variable(para + '_r' + str(i_region))
        return output


    def add_one_cell_h(self, h_state_current, x_state_current, h_parameter):
        """
        Model the evolving of hemodynamic states {s,f,v,q}
        This is independent for each region
        Here x_state_current is r scalar for r particular region
        :param h_state_current:
        :param x_state_current:
        :param h_parameter: a dictionary with all hemodynamic parameter
        :return:
        """
        # with tf.variable_scope(self.variable_scope_name_h, reuse=True):
        alpha = h_parameter["alpha"]
        k = h_parameter["k"]
        E0 = h_parameter["E0"]
        gamma = h_parameter["gamma"]
        tao = h_parameter["tao"]
        x_h_coupling = h_parameter["x_h_coupling"]
        t_delta = self.t_delta

        h_state_augmented = self.phi_h(h_state_current, alpha, E0)
        h_state_next = []
        # s
        tmp1 = tf.multiply(t_delta, x_h_coupling * x_state_current)
        tmp2 = tf.multiply(tf.subtract(tf.multiply(t_delta, k), 1.), h_state_augmented[0])
        tmp3 = tf.multiply(t_delta, tf.multiply(gamma, tf.subtract(h_state_augmented[1], 1.)))
        tmp = tf.subtract(tmp1, tf.add(tmp2, tmp3))
        h_state_next.append(tmp)
        # f
        tmp = tf.add(h_state_augmented[1], tf.multiply(t_delta, h_state_augmented[0]))
        h_state_next.append(tmp)
        # v
        tmp = t_delta * h_state_augmented[1] / tao \
              - t_delta / tao * h_state_augmented[4] \
              + h_state_augmented[2]
        h_state_next.append(tmp)
        # q
        tmp = h_state_augmented[3] \
              + t_delta / tao * h_state_augmented[6] \
              - t_delta / tao * h_state_augmented[5]
        h_state_next.append(tmp)
        # concantenate into a tensor
        h_state_next = tf.stack(h_state_next, 0)
        return h_state_next

    def add_hemodynamic_layer(self, x_state=None, h_state_initial=None):
        """
        :param x_state:
        :return:
        """
        if x_state is None:
            x_state = self.x_state_predicted
        if h_state_initial is None:
            h_state_initial = self.h_state_initial

        # format: h_state_predicted[time][region, 4]
        # format: h_state_predicted_stacked[time, region, 4]
        self.h_state_predicted = []
        self.h_state_predicted.append(h_state_initial)
        for i in range(1, self.n_recurrent_step):
            with tf.variable_scope(self.variable_scope_name_h_parameter, reuse=True):
                para_packages = []
                for i_region in range(self.n_region):
                    para_packages.append(self.get_h_para_tensor_for_one_region(i_region))
            with tf.variable_scope(self.variable_scope_name_h, reuse=True):
                h_temp = []
                for i_region in range(self.n_region):
                    h_temp.append(self.add_one_cell_h(
                        self.h_state_predicted[i - 1][i_region, :], x_state[i - 1][i_region], para_packages[i_region]))
                self.h_state_predicted.append(tf.stack(h_temp, 0))

        i = self.n_recurrent_step
        with tf.variable_scope(self.variable_scope_name_h_parameter, reuse=True):
            para_packages = []
            for i_region in range(self.n_region):
                para_packages.append(self.get_h_para_tensor_for_one_region(i_region))
        with tf.variable_scope(self.variable_scope_name_h_final, reuse=True):
            h_temp = []
            for i_region in range(self.n_region):
                h_temp.append(self.add_one_cell_h(
                    self.h_state_predicted[i - 1][i_region, :], x_state[i - 1][i_region], para_packages[i_region]))
            self.h_state_final = tf.stack(h_temp, 0)

        with tf.variable_scope(self.variable_scope_name_h_stacked):
            self.h_state_predicted_stacked = tf.stack(self.h_state_predicted, 0)

    def phi_o(self, h_state_current):
        """
        Used to map hemodynamic states into higher dimension to calculate fMRI signal
        :param h_state_current:
        :return:
        """
        o_state_augmented = [h_state_current[i + 2] for i in range(2)]
        tmp = tf.div(o_state_augmented[1], o_state_augmented[0])
        o_state_augmented.append(tmp)
        return o_state_augmented

    def output_mapping(self, h_state_current, parameter_package):

        E0 = parameter_package['E0']
        epsilon = parameter_package['epsilon']
        V0 = parameter_package['V0']
        TE = parameter_package['TE']
        r0 = parameter_package['r0']
        theta0 = parameter_package['theta0']

        k1 = 4.3 * theta0 * E0 * TE
        k2 = epsilon * r0 * E0 * TE
        k3 = 1 - epsilon

        o_state_augmented = self.phi_o(h_state_current)

        y = V0 * k1 * (1 - o_state_augmented[1]) \
            + V0 * k2 * (1 - o_state_augmented[2]) \
            + V0 * k3 * (1 - o_state_augmented[0])

        return y

    def add_output_layer(self, h_state_predicted=None):
        h_state_predicted = h_state_predicted or self.h_state_predicted
        self.y_predicted = []

        for i in range(0, self.n_recurrent_step):
            with tf.variable_scope(self.variable_scope_name_h_parameter, reuse=True):
                para_packages = []
                for i_region in range(self.n_region):
                    para_packages.append(self.get_h_para_tensor_for_one_region(i_region))
            with tf.variable_scope(self.variable_scope_name_y):
                y_temp = []
                for i_region in range(self.n_region):
                    y_temp.append(self.output_mapping(h_state_predicted[i][i_region, :], para_packages[i_region]))
                y_temp = tf.stack(y_temp, 0)
                self.y_predicted.append(y_temp)
        with tf.variable_scope(self.variable_scope_name_y_stacked):
            self.y_predicted_stacked = tf.stack(self.y_predicted, 0, name='y_predicted')

    def build_an_initializer_graph(self, hemodynamic_parameter_initial=None):
        """
        Build a model to estimate neural states from functional signal
        :return:
        """

        self.hemodynamic_parameter_initial = hemodynamic_parameter_initial or \
                                             self.get_standard_hemodynamic_parameters(self.n_region).astype(np.float32)

        # create place holder and variables
        self.y_true = tf.placeholder(dtype=tf.float32, shape=[self.n_recurrent_step, self.n_region], name="y_true")
        with tf.variable_scope(self.variable_scope_name_x):
            self.x_state = \
                tf.get_variable(name='x_state', dtype=tf.float32, shape=[self.n_recurrent_step, self.n_region])
            self.x_state_placeholder = \
                tf.placeholder(dtype=tf.float32, shape=[self.n_recurrent_step, self.n_region], name='x_state_placeholder')
            self.assign_x = tf.assign(self.x_state, self.x_state_placeholder, name='assign_x_state')
            self.x_state_list = []

        for n in range(self.n_recurrent_step):
            with tf.variable_scope(self.variable_scope_name_x):
                self.x_state_list.append(self.x_state[n, :])

        with tf.variable_scope(self.variable_scope_name_h):
            self.h_state_initial_default = \
                self.h_state_initial_default = \
                self.set_initial_hemodynamic_state_as_inactivated(n_node=self.n_region).astype(np.float32)
            self.h_state_initial = \
                tf.get_variable('h_state_initial',
                                initializer=self.h_state_initial_default,
                                trainable=True)


        self.create_shared_variables_h(self.hemodynamic_parameter_initial)

        # build model
        self.add_hemodynamic_layer(self.x_state_list, self.h_state_initial)
        self.add_output_layer()

        # define loss
        with tf.variable_scope(self.variable_scope_name_loss):
            self.loss = self.mse(self.y_true, self.y_predicted, "loss")
        with tf.variable_scope('accumulate_' + self.variable_scope_name_loss):
            self.loss_total = tf.get_variable('loss_total', initializer=0., trainable=False)
            # self.loss_total = self.loss_total + self.loss
            self.sum_loss = tf.assign_add(self.loss_total, self.loss, name='accumulate_loss')
            self.clear_loss_total = tf.assign(self.loss_total, 0., name='clear_loss_total')

        # define optimiser
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # define summarizer
        self.variable_summaries(self.loss_total)
        self.merged_summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.log_directory, tf.get_default_graph())

    def run_initializer_graph(self, sess, h_state_initial, data_x):
        """
        Run forward the initializer graph
        :param sess:
        :param h_state_initial:
        :param data_x: a list of neural activity signal segment
        :return: [y_predicted, h_state_predicted]
        """
        h_state_predicted = []
        y_predicted = []
        h_state_initial_segment = h_state_initial
        for x_segment in data_x:
            y_segment, h_segment, h_state_final = \
                sess.run([self.y_predicted_stacked, self.h_state_predicted_stacked, self.h_state_final],
                         feed_dict={self.x_state: x_segment, self.h_state_initial: h_state_initial_segment})
            h_state_initial_segment = h_state_final
            h_state_predicted.append(h_segment)
            y_predicted.append(y_segment)
        h_state_predicted = np.concatenate(h_state_predicted)
        y_predicted = np.concatenate(y_predicted)
        return [y_predicted, h_state_predicted]


    # unitilies
    def get_element_count(self, tensor):
        return np.prod(tensor.get_shape().as_list())

    def mse(self, tensor1, tensor2, name=None):
        with tf.variable_scope('MSE'):
            temp = tf.reduce_sum((tf.reshape(tensor1, [-1]) - tf.reshape(tensor2, [-1]) ) ** 2)
            mse = temp / self.get_element_count(tensor1)
            if name is not None:
                tf.identity(mse, name=name)
            return mse

    def variable_summaries(self, tensor):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        scope_name = 'Summary_' + tensor.op.name
        with tf.name_scope(scope_name):
            mean = tf.reduce_mean(tensor)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(tensor))
            tf.summary.scalar('min', tf.reduce_min(tensor))
            tf.summary.histogram('histogram', tensor)

