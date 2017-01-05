import numpy as np
import tensorflow as tf
import math as mth
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import inspect
import tensorflow as tf


class DCM_RNN:
    def __init__(self, m, n_recurrent_step=None, learning_rate=None):
        # self.batch_size = 1
        n_recurrent_step = n_recurrent_step or 8
        learning_rate = learning_rate or 0.01
        n_region = m.n_region
        t_delta = m.t_delta
        n_stimuli = m.n_stimuli

        self.n_recurrent_step = n_recurrent_step
        self.learning_rate = learning_rate
        self.n_region = n_region
        self.t_delta = t_delta
        self.n_stimuli = n_stimuli

        self.variable_scope_name_x = 'rnn_cell_x'
        self.variable_scope_name_h = 'rnn_cell_h'

        # set connection initial value
        with tf.variable_scope(self.variable_scope_name_x):
            self.Wxx_init = np.array([[-1, -0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32) * m.t_delta + np.eye(
                m.n_region, m.n_region, 0, dtype=np.float32)
            self.Wxxu_init = [np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32) * m.t_delta for _ in
                              range(n_stimuli)]
            self.Wxu_init = np.array([0.5, 0, 0], dtype=np.float32).reshape(3, 1) * m.t_delta

        self.set_up_parameter_prior()
        self.set_up_hyperparameter_values()

    def build_a_model(self):
        function_name = inspect.stack()[0][3]

        [self.input_u, self.input_y_true] = self.add_placeholders()

        # create shared variables in computation graph
        self.create_shared_variables()
        print(function_name + ': variables created.')

        # build layers
        self.add_neural_layer()
        self.add_hemodynamic_layer()
        self.add_output_layer()
        print(function_name + ': layers created.')

        self.names_in_graph = [item.name for item in tf.trainable_variables()]
        self.set_up_parameter_profile()

        # masks for penalty weighting and connection support
        # self.define_masks()

        # define loss
        self.add_loss_prediction()
        self.add_loss_sparse()
        self.add_loss_prior()
        self.collect_losses()
        print(function_name + ': losses added.')

        self.calculate_gradients()
        self.process_gradients()
        self.apply_gradients()
        print(function_name + ': finished.')

    def set_up_hyperparameter_values(self):
        hyperparameter_values = {}
        hyperparameter_values[self.variable_scope_name_x] = {'gradient': 1., 'sparse': 1., 'prior': 0.}
        hyperparameter_values[self.variable_scope_name_h] = {'gradient': 1., 'sparse': 0., 'prior': 1.}
        self.hyperparameter_values = hyperparameter_values

    def set_connection_matrices_from_initial(self, isess):
        with tf.variable_scope(self.variable_scope_name_x):
            isess.run(self.Wxx.assign(self.Wxx_init))
            isess.run(self.Wxxu.assign(self.Wxxu_init))
            isess.run(self.Wxu.assign(self.Wxu_init))

    def set_connection_matrices(self, isess, Wxx, Wxxu, Wxu):
        with tf.variable_scope(self.variable_scope_name_x):
            isess.run(self.Wxx.assign(Wxx))
            for idx, item in enumerate(Wxxu):
                isess.run(self.Wxxu[idx].assign(item))
            isess.run(self.Wxu.assign(Wxu))

    def rnn_cell(self, u_current, x_state_previous):
        n_region = x_state_previous.get_shape()[0]
        n_stimuli = self.n_stimuli
        with tf.variable_scope(self.variable_scope_name_x, reuse=True):
            Wxx = tf.get_variable("Wxx", [n_region, n_region])
            Wxxu = [tf.get_variable("Wxxu_s" + str(n)) for n in range(n_stimuli)]
            Wxu = tf.get_variable("Wxu", [n_region, n_stimuli])

            tmp1 = tf.matmul(Wxx, x_state_previous)
            tmp2 = [tf.matmul(Wxxu[n] * u_current[n], x_state_previous) for n in range(n_stimuli)]
            tmp2 = tf.add_n(tmp2)
            tmp3 = tf.mul(Wxu, u_current)
            # return tf.add(tf.add(tmp1,tmp2),tmp3)
            return tmp1 + tmp2 + tmp3

    def phi_h(self, h_state_current, alpha, E0):
        # used to map hemodynamic states into higher dimension
        # for hemodynamic states evolvement
        h_state_augmented = []
        for i in range(4):
            h_state_augmented.append(h_state_current[i])

        h_state_augmented.append(tf.pow(h_state_current[2], tf.div(1., alpha)))
        h_state_augmented.append(tf.mul(tf.div(h_state_current[3], h_state_current[2]), h_state_augmented[4]))
        tmp = tf.sub(1., tf.pow(tf.sub(1., E0), tf.div(1., h_state_current[1])))
        tmp = tf.mul(tf.div(tmp, E0), h_state_current[1])
        h_state_augmented.append(tmp)
        return h_state_augmented

    def rnn_cell_h(self, h_state_current, x_state_current, i_region):
        # model the evolving of hemodynamic states {s,f,v,q}
        # this is independent for each region
        # here x_state_current is a scalar for a particular region
        with tf.variable_scope(self.variable_scope_name_h, reuse=True):
            alpha = tf.get_variable('alpha_r' + str(i_region))
            E0 = tf.get_variable('E0_r' + str(i_region))
            k = tf.get_variable('k_r' + str(i_region))
            gamma = tf.get_variable('gamma_r' + str(i_region))
            tao = tf.get_variable('tao_r' + str(i_region))
            t_delta = self.t_delta

            h_state_augmented = self.phi_h(h_state_current, alpha, E0)
            tmp_list = []
            # s
            tmp1 = tf.mul(t_delta, x_state_current)
            tmp2 = tf.mul(tf.sub(tf.mul(t_delta, k), 1.), h_state_augmented[0])
            tmp3 = tf.mul(t_delta, tf.mul(gamma, tf.sub(h_state_augmented[1], 1.)))
            tmp = tf.sub(tmp1, tf.add(tmp2, tmp3))
            tmp_list.append(tf.reshape(tmp, [1, 1]))
            # f
            tmp = tf.add(h_state_augmented[1], tf.mul(t_delta, h_state_augmented[0]))
            tmp_list.append(tf.reshape(tmp, [1, 1]))
            # v
            tmp = t_delta * h_state_augmented[1] / tao \
                  - t_delta / tao * h_state_augmented[4] \
                  + h_state_augmented[2]
            tmp_list.append(tf.reshape(tmp, [1, 1]))
            # q
            tmp = h_state_augmented[3] \
                  + t_delta / tao * h_state_augmented[6] \
                  - t_delta / tao * h_state_augmented[5]
            tmp_list.append(tf.reshape(tmp, [1, 1]))
            # concantenate into a tensor
            tmp_list = tf.concat(0, tmp_list)
            tmp_list = tf.reshape(tmp_list, [4, 1])
            return tmp_list

    def phi_o(self, h_state_current):
        # used to map hemodynamic states into higher dimension
        # for fMRI output
        o_state_augmented = [h_state_current[i + 2] for i in range(2)]
        tmp = tf.div(o_state_augmented[1], o_state_augmented[0])
        o_state_augmented.append(tmp)
        return o_state_augmented

    def output_mapping(self, h_state_current, i_region):
        with tf.variable_scope(self.variable_scope_name_h, reuse=True):
            E0 = tf.get_variable('E0_r' + str(i_region))
            epsilon = tf.get_variable('epsilon_r' + str(i_region))
            V0 = tf.get_variable('V0_r' + str(i_region))
            TE = tf.get_variable('TE_r' + str(i_region))
            r0 = tf.get_variable('r0_r' + str(i_region))
            theta0 = tf.get_variable('theta0_r' + str(i_region))
            k1 = 4.3 * theta0 * E0 * TE
            k2 = epsilon * r0 * E0 * TE
            k3 = 1 - epsilon

            o_state_augmented = self.phi_o(h_state_current)

            y = V0 * k1 * (1 - o_state_augmented[1]) \
                + V0 * k2 * (1 - o_state_augmented[2]) \
                + V0 * k3 * (1 - o_state_augmented[0])

            return y

    def add_placeholders(self):
        input_u = tf.placeholder(tf.float32, [self.n_stimuli, self.n_recurrent_step], name='input_u')
        input_y_true = tf.placeholder(tf.float32, [self.n_region, self.n_recurrent_step], name='input_y_true')
        return [input_u, input_y_true]

    def get_random_h_state_initial(self):
        h_state_initial = np.zeros((4, 1))
        h_state_initial[0] = np.random.normal(loc=0, scale=0.3)
        h_state_initial[1] = np.random.normal(loc=1.5, scale=0.5)
        h_state_initial[2] = np.random.normal(loc=1.15, scale=0.15)
        h_state_initial[3] = np.random.normal(loc=0.85, scale=0.15)

    def set_up_parameter_prior(self):
        names = ['alpha', 'E0', 'k', 'gamma', 'tao', 'epsilon', 'V0', 'TE', 'r0', 'theta0']
        means = [0.32, 0.34, 0.65, 0.41, 0.98, 0.4, 100., 0.03, 25., 40.3]
        stds = [np.sqrt(0.0015), np.sqrt(0.0024), np.sqrt(0.015), np.sqrt(0.002), np.sqrt(0.0568), None, None, None,
                None, None]
        distributions = ['Gaussian', 'Gaussian', 'Gaussian', 'Gaussian', 'Gaussian', 'Gaussian', 'Gaussian', 'Gaussian',
                         'Gaussian', 'Gaussian']
        parameter_prior = {}
        for idx in range(len(names)):
            parameter_prior[names[idx]] = {}
            parameter_prior[names[idx]]['distribution'] = distributions[idx]
            parameter_prior[names[idx]]['mean'] = means[idx]
            parameter_prior[names[idx]]['std'] = stds[idx]
        self.parameter_prior = parameter_prior

    def create_shared_variables(self):
        # for neural level
        with tf.variable_scope(self.variable_scope_name_x):
            trainable_flag = True
            self.Wxx = tf.get_variable('Wxx', initializer=self.Wxx_init, trainable=trainable_flag)
            self.Wxxu = [tf.get_variable('Wxxu_s' + str(n), initializer=self.Wxxu_init[n], trainable=trainable_flag) for
                         n in range(self.n_stimuli)]
            self.Wxu = tf.get_variable('Wxu', initializer=self.Wxu_init, trainable=trainable_flag)

        # for hemodynamic level
        with tf.variable_scope(self.variable_scope_name_h):
            self.alpha = {}
            self.E0 = {}
            self.k = {}
            self.gamma = {}
            self.tao = {}
            self.epsilon = {}
            self.V0 = {}
            self.TE = {}
            self.r0 = {}
            self.theta0 = {}
            trainable_flag = True
            for n in range(self.n_region):
                self.alpha['alpha_r' + str(n)] = tf.get_variable('alpha_r' + str(n),
                                                                 initializer=self.parameter_prior['alpha']['mean'],
                                                                 trainable=trainable_flag)
                self.E0['E0_r' + str(n)] = tf.get_variable('E0_r' + str(n),
                                                           initializer=self.parameter_prior['E0']['mean'],
                                                           trainable=trainable_flag)
                self.k['k_r' + str(n)] = tf.get_variable('k_r' + str(n), initializer=self.parameter_prior['k']['mean'],
                                                         trainable=trainable_flag)
                self.gamma['gamma_r' + str(n)] = tf.get_variable('gamma_r' + str(n),
                                                                 initializer=self.parameter_prior['gamma']['mean'],
                                                                 trainable=trainable_flag)
                self.tao['tao_r' + str(n)] = tf.get_variable('tao_r' + str(n),
                                                             initializer=self.parameter_prior['tao']['mean'],
                                                             trainable=trainable_flag)
                self.epsilon['epsilon_r' + str(n)] = tf.get_variable('epsilon_r' + str(n),
                                                                     initializer=self.parameter_prior['epsilon'][
                                                                         'mean'],
                                                                     trainable=False)  # This is set to untrainable by design
                self.V0['V0_r' + str(n)] = tf.get_variable('V0_r' + str(n),
                                                           initializer=self.parameter_prior['V0']['mean'],
                                                           trainable=False)  # This is set to untrainable by design
                self.TE['TE_r' + str(n)] = tf.get_variable('TE_r' + str(n),
                                                           initializer=self.parameter_prior['TE']['mean'],
                                                           trainable=False)  # This is set to untrainable by design
                self.r0['r0_r' + str(n)] = tf.get_variable('r0_r' + str(n),
                                                           initializer=self.parameter_prior['r0']['mean'],
                                                           trainable=False)  # This is set to untrainable by design
                self.theta0['theta0_r' + str(n)] = tf.get_variable('theta0_r' + str(n),
                                                                   initializer=self.parameter_prior['theta0']['mean'],
                                                                   trainable=False)  # This is set to untrainable by design

    def add_neural_layer(self):
        self.x_state_initial = tf.zeros((self.n_region, 1), dtype=np.float32)

        self.x_state_predicted = []
        self.x_state_predicted.append(self.x_state_initial)
        for i in range(1, self.n_recurrent_step):
            tmp = self.rnn_cell(self.input_u[0, i - 1], self.x_state_predicted[i - 1])
            self.x_state_predicted.append(tmp)
        # the last element needs special handling
        i = self.n_recurrent_step
        self.x_state_final = self.rnn_cell(self.input_u[0, i - 1], self.x_state_predicted[i - 1])

    def add_hemodynamic_layer(self):
        n_region = self.n_region
        self.h_state_initial = [
            tf.get_variable('h_state_initial_r' + str(n), shape=[4, 1], initializer=self.get_random_h_state_initial(),
                            trainable=False) \
            for n in range(n_region)]

        # format: h_state_predicted[region][time]
        self.h_state_predicted = [[] for _ in range(n_region)]
        for n in range(n_region):
            self.h_state_predicted[n].append(self.h_state_initial[n])
        for i in range(1, self.n_recurrent_step):
            for n in range(n_region):
                self.h_state_predicted[n].append( \
                    self.rnn_cell_h(self.h_state_predicted[n][i - 1], \
                                    self.x_state_predicted[i - 1][n], n))
        # the last element needs special handling
        i = self.n_recurrent_step
        self.h_state_final = []
        for n in range(n_region):
            self.h_state_final.append( \
                self.rnn_cell_h(self.h_state_predicted[n][i - 1], \
                                self.x_state_predicted[i - 1][n], n))

    def add_output_layer(self):
        self.y_state_predicted = []
        for i in range(0, self.n_recurrent_step):
            tmp = []
            for n in range(self.n_region):
                tmp.append(self.output_mapping(self.h_state_predicted[n][i], n))
            tmp = tf.pack(tmp)
            self.y_state_predicted.append(tmp)

    def define_masks(self):
        names = self.names_in_graph
        self.masks = type('container', (object,), {})()
        self.masks.gradient = {}
        self.masks.sparse = {}
        self.masks.prior = {}
        for idx, name in enumerate(names):
            tmp = tf.get_default_graph().get_tensor_by_name(name).get_shape()
            self.masks.gradient[name] = tf.placeholder(tf.float32, tmp, name='mask_gradient_' + str(idx))
            self.masks.sparse[name] = tf.placeholder(tf.float32, tmp, name='mask_sparse_' + str(idx))
            self.masks.prior[name] = tf.placeholder(tf.float32, tmp, name='mask_prior_' + str(idx))

    def set_up_parameter_profile(self):
        # create hyperparameter masks associated with each parameter
        # including BP masks, sparse masks, prior mask, variable_scope
        hyperparameter_values = self.hyperparameter_values
        variable_names = self.names_in_graph
        hyperparameter_masks = {}
        for name in variable_names:
            hyperparameter_masks[name] = {}
            if self.variable_scope_name_x in name:
                self.add_a_parameter_profile(name, self.variable_scope_name_x, hyperparameter_values,
                                             hyperparameter_masks)
            elif self.variable_scope_name_h in name:
                self.add_a_parameter_profile(name, self.variable_scope_name_h, hyperparameter_values,
                                             hyperparameter_masks)
            else:
                function_name = inspect.stack()[0][3]
                raise ValueError(function_name + '() unknown variable scope')
        self.parameter_profile = hyperparameter_masks

    def find_keyword(self, name):
        if 'Wxx' in name:
            keyword = 'Wxx'
        elif 'Wxxu' in name:
            keyword = 'Wxxu'
        elif 'Wxu' in name:
            keyword = 'Wxu'
        elif 'alpha' in name:
            keyword = 'alpha'
        elif 'E0' in name:
            keyword = 'E0'
        elif 'k' in name:
            keyword = 'k'
        elif 'gamma' in name:
            keyword = 'gamma'
        elif 'tao' in name:
            keyword = 'tao'
        elif 'epsilon' in name:
            keyword = 'epsilon'
        elif 'V0' in name:
            keyword = 'V0'
        elif 'TE' in name:
            keyword = 'TE'
        elif 'r0' in name:
            keyword = 'r0'
        elif 'theta0' in name:
            keyword = 'theta0'
        else:
            function_name = inspect.stack()[0][3]
            raise ValueError(function_name + '() cannot find proper keyword')
        return keyword

    def clip_name(self, name):
        start_index = name.rfind('/')
        end_index = name.find(':')
        return name[start_index + 1:end_index]

    def add_a_parameter_profile(self, name, scope, hyperparameter_values, hyperparameter_masks):
        with tf.variable_scope(scope):
            hyperparameter_masks[name]['scope'] = scope
            variable_shape = tf.get_default_graph().get_tensor_by_name(name).get_shape()
            hyperparameter_masks[name]['shape'] = variable_shape
            hyperparameter_masks[name]['keyword'] = self.find_keyword(name)
            hyperparameter_masks[name]['mask_gradient'] = tf.get_variable(self.clip_name(name) + '_mask_gradient', \
                                                                          initializer=np.float32(
                                                                              np.ones(variable_shape) *
                                                                              hyperparameter_values[
                                                                                  hyperparameter_masks[name]['scope']][
                                                                                  'gradient']), \
                                                                          trainable=False)
            hyperparameter_masks[name]['mask_sparse'] = tf.get_variable(self.clip_name(name) + '_mask_sparse', \
                                                                        initializer=np.float32(np.ones(variable_shape) *
                                                                                               hyperparameter_values[
                                                                                                   hyperparameter_masks[
                                                                                                       name]['scope']][
                                                                                                   'sparse']), \
                                                                        trainable=False)
            hyperparameter_masks[name]['mask_prior'] = tf.get_variable(self.clip_name(name) + '_mask_prior', \
                                                                       initializer=np.float32(np.ones(variable_shape) *
                                                                                              hyperparameter_values[
                                                                                                  hyperparameter_masks[
                                                                                                      name]['scope']][
                                                                                                  'prior']), \
                                                                       trainable=False)

    def add_loss_prediction(self):
        y_true_as_list = [tf.reshape(self.input_y_true[:, i], (self.n_region, 1)) for i in range(self.n_recurrent_step)]
        self.loss_y_list = [(tf.reduce_mean(tf.square(tf.sub(y_pred, y_true)))) \
                            for y_pred, y_true in zip(self.y_state_predicted, y_true_as_list)]
        self.loss_y = tf.reduce_mean(self.loss_y_list)

    def add_loss_sparse(self):
        # check mask value, if all 0, then don't add the variable to the loss
        parameter_profile = self.parameter_profile
        variable_names = self.names_in_graph
        variable_values = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        self.loss_sparse_list = []
        for name, value in zip(variable_names, variable_values):
            if np.sum(abs(parameter_profile[name]['mask_sparse'])) == 0:
                pass
            else:
                self.loss_sparse_list.append(
                    tf.reduce_sum(tf.reshape(tf.abs(value * parameter_profile[name]['mask_sparse']), [-1])))
        if not self.loss_sparse_list:
            self.loss_sparse = tf.constant([0.])
        else:
            self.loss_sparse = tf.add_n(self.loss_sparse_list)

    def add_loss_prior(self):
        # check mask value, if all 0, then don't add the variable to the loss
        parameter_profile = self.parameter_profile
        variable_names = self.names_in_graph
        parameter_prior = self.parameter_prior

        self.loss_prior_list = []
        for name in variable_names:
            if self.check_prior_effectiveness(name):
                self.loss_prior_list.append(self.calculate_prior_loss(name))
        if not self.loss_prior_list:
            self.loss_prior = tf.constant([0.])
        else:
            self.loss_prior = tf.add_n(self.loss_prior_list)

    def check_prior_effectiveness(self, name):
        parameter_profile = self.parameter_profile
        parameter_prior = self.parameter_prior

        keyword = parameter_profile[name]['keyword']
        if np.sum(abs(parameter_profile[name]['mask_prior'])) == 0:
            return False
        elif not keyword in parameter_prior:
            return False
        elif None in parameter_prior[keyword]:
            return False
        else:
            return True

    def calculate_prior_loss(self, name):
        parameter_profile = self.parameter_profile
        parameter_prior = self.parameter_prior

        variable = tf.get_default_graph().get_tensor_by_name(name)
        keyword = parameter_profile[name]['keyword']
        if parameter_prior[keyword]['distribution'] == 'Gaussian':
            loss_unmasked = ((variable - parameter_prior[keyword]['mean']) / parameter_prior[keyword]['std']) ** 2
            loss = loss_unmasked * parameter_profile[name]['mask_prior']
        return loss

    def collect_losses(self):
        self.loss_total = self.loss_y + self.loss_sparse + self.loss_prior

    def calculate_gradients(self):
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.grads_and_vars = self.opt.compute_gradients(self.loss_total)

    def process_gradients(self):
        parameter_profile = self.parameter_profile
        variable_names = self.names_in_graph
        # self.processed_grads_and_vars = [(gv[0]*self.masks.gradient[self.names[idx]], gv[1]) for idx,gv in enumerate(self.grads_and_vars)]
        self.processed_grads_and_vars = [(gv[0] * self.parameter_profile[variable_names[idx]]['mask_gradient'], gv[1])
                                         for idx, gv in enumerate(self.grads_and_vars)]

    def apply_gradients(self):
        self.apply_gradient = self.opt.apply_gradients(self.processed_grads_and_vars)


class Utilities:
    def __init__(self, default_model=None, default_session=None, n_region=None, n_recurrence=None, learning_rate=None):
        '''
		self.n_region = n_region or 3
		self.n_recurrent_step = n_recurrence or 8 
		self.learning_rate = learning_rate or 0.01
		self.n_region=m.n_region
		'''
        # self.default_dr = dr
        self.parameter_key_list = ['Wxx', 'Wxxu', 'Wxu', 'alpha', 'E0', 'k', 'gamma', 'tao', 'epsilon', 'V0', 'TE',
                                   'r0', 'theta0']
        self.default_model = default_model
        self.default_sess = default_session

    def run_forward_segment(self, dr, sess, feed_dict_in):
        x_state, x_state_final = sess.run([dr.x_state_predicted, dr.x_state_final], \
                                          feed_dict={dr.input_u: feed_dict_in['input_u'], \
                                                     dr.rnn_x: feed_dict_in['rnn_x'],
                                                     dr.x_state_initial: feed_dict_in['x_state_initial']})
        return [x_state, x_state_final]

    def forward_pass_x(self, dr, dh, isess, x_state_initial=None):
        training_state = x_state_initial or np.zeros((dr.n_region, 1))

        x_state_predicted = []
        for i in range(len(dh.u_list)):
            tmp, training_state = isess.run([dr.x_state_predicted, dr.x_state_final], \
                                            feed_dict={dr.input_u: dh.u_list[i],
                                                       # dr.rnn_x:dh.x_list[i],
                                                       dr.x_state_initial: training_state})
            tmp = np.asarray(tmp)
            tmp = tmp[:, :, 0]
            x_state_predicted.append(tmp)
        x_state_predicted = np.concatenate(x_state_predicted).transpose()
        self.x_state_predicted = x_state_predicted[:]
        return x_state_predicted

    def show_x(self, t_delta=1, length=None):
        length = length or self.x_state_predicted.shape[1]
        plt.plot(np.arange(length) * t_delta, self.x_state_predicted[0, 0:length].transpose())

    def forward_pass_h(self, dr, dh, isess, x_state_initial=None, h_state_initial=None):
        x_state_feed = x_state_initial or np.zeros((dr.n_region, 1))
        h_state_feed = h_state_initial or [np.array([0, 1, 1, 1], dtype=np.float32).reshape(4, 1) for _ in
                                           range(dr.n_region)]

        h_state_predicted = [[] for _ in range(3)]

        for i in range(len(dh.u_list)):
            # build feed_dictionary
            feed_dict = {i: d for i, d in zip(dr.h_state_initial, h_state_feed)}
            feed_dict[dr.x_state_initial] = x_state_feed
            feed_dict[dr.input_u] = dh.u_list[i]
            # run
            h_current_segment, h_state_feed, x_state_feed = isess.run(
                [dr.h_state_predicted, dr.h_state_final, dr.x_state_final], feed_dict=feed_dict)

            for n in range(dr.n_region):
                # concatenate h_current_segment in to list of 3 element, each a np.narray
                h_current_segment[n] = np.squeeze(np.asarray(h_current_segment[n])).transpose()
                # h_state_predicted[n].append(h_current_segment[n])
                if i == 0:
                    h_state_predicted[n] = h_current_segment[n]
                else:
                    h_state_predicted[n] = np.concatenate((h_state_predicted[n], h_current_segment[n]), axis=1)
        return h_state_predicted

    def forward_pass_y(self, dr, dh, isess, x_state_initial=None, h_state_initial=None):
        x_state_feed = x_state_initial or np.zeros((dr.n_region, 1))
        h_state_feed = h_state_initial or [np.array([0, 1, 1, 1], dtype=np.float32).reshape(4, 1) for _ in
                                           range(dr.n_region)]

        y_output_predicted = []
        for i in range(len(dh.u_list)):
            # build feed_dictionary
            feed_dict = {i: d for i, d in zip(dr.h_state_initial, h_state_feed)}
            feed_dict[dr.x_state_initial] = x_state_feed
            feed_dict[dr.input_u] = dh.u_list[i]
            # run
            y_current_segment, h_state_feed, x_state_feed = isess.run(
                [dr.y_state_predicted, dr.h_state_final, dr.x_state_final], \
                feed_dict=feed_dict)
            # orgnize output
            # print(len(y_current_segment))
            # print(len(y_current_segment[0]))
            y_current_segment = np.asarray(y_current_segment)
            y_current_segment = np.squeeze(y_current_segment)
            y_output_predicted.append(y_current_segment)
        y_output_predicted = np.concatenate(y_output_predicted).transpose()
        self.y_output_predicted = y_output_predicted[:]
        return y_output_predicted

    def show_all_variable_value(self, dr, isess, visFlag=False):
        output = []
        output_buff = pd.DataFrame()
        # variables= self.parameter_key_list
        # print(variables)
        # values=eval('isess.run(['+', '.join(variables)+'])')
        for idx, key in enumerate(self.parameter_key_list):
            if key == 'Wxx':
                values = isess.run(dr.Wxx)
                tmp = pd.DataFrame(values, index=['To_r' + str(i) for i in range(dr.n_region)], \
                                   columns=['From_r' + str(i) for i in range(dr.n_region)])
                tmp.name = key
                output.append(tmp)
            elif key == 'Wxxu':
                values = isess.run(dr.Wxxu)
                for n in range(dr.n_stimuli):
                    tmp = pd.DataFrame(values[n], index=['To_r' + str(i) for i in range(dr.n_region)], \
                                       columns=['From_r' + str(i) for i in range(dr.n_region)])
                    tmp.name = key + '_s' + str(n)
                    output.append(tmp)
            elif key == 'Wxu':
                values = isess.run(dr.Wxu)
                tmp = pd.DataFrame(values, index=['To_r' + str(i) for i in range(dr.n_region)], \
                                   columns=['stimuli_' + str(i) for i in range(dr.n_stimuli)])
                tmp.name = key
                output.append(tmp)
            else:
                values = eval('isess.run(dr.' + key + ')')
                # print(key)
                # print(values)
                tmp = [values[key + '_r' + str(i)] for i in range(dr.n_region)]
                tmp = pd.Series(tmp, index=['region_' + str(i) for i in range(dr.n_region)])
                output_buff[key] = tmp
        output_buff.name = 'hemodynamic_parameters'
        output.append(output_buff)
        if visFlag:
            for item in output:
                print(item.name)
                display(item)
        return output

    def compare_parameters(self, set1, set2, visFlag=True, parameter_list=None):
        if parameter_list == None:
            name_list1 = [set1[i].name for i in range(len(set1))]
            name_list2 = [set2[i].name for i in range(len(set2))]
            name_list = list(set(name_list1) & set(name_list2))  # common name list
        else:
            name_list = parameter_list
        output = []
        for name in name_list:
            tmp1 = next((x for x in set1 if x.name == name), None)
            tmp2 = next((x for x in set2 if x.name == name), None)
            if tmp1.shape[0] >= tmp1.shape[1]:
                tmp = pd.concat([tmp1, tmp2, tmp1 - tmp2], axis=1, join_axes=[tmp1.index],
                                keys=['set1', 'set2', 'difference'])
            else:
                tmp = pd.concat([tmp1, tmp2, tmp1 - tmp2], axis=0, join_axes=[tmp1.columns],
                                keys=['set1', 'set2', 'difference'])
            tmp.name = name
            output.append(tmp)
        if visFlag:
            for item in output:
                print(item.name)
                display(item)
        return output

    def set_connection_matrices(self, dr, isess, Wxx, Wxxu, Wxu):
        with tf.variable_scope(self.variable_scope_name_x):
            isess.run(dr.Wxx.assign(Wxx))
            for idx, item in enumerate(Wxxu):
                isess.run(dr.Wxxu[idx].assign(item))
            isess.run(dr.Wxu.assign(Wxu))

    def update_parameter_profile(self, dr=None, session=None):
        dr = dr or self.default_model
        session = session or self.default_session

        hyperparameter_values = dr.hyperparameter_values
        parameter_profile = dr.parameter_profile

        for name in parameter_profile:
            item = parameter_profile[name]
            variable_shape = item['shape']
            with tf.variable_scope(item['scope']):
                session.run(item['mask_gradient'].assign(
                    np.float32(np.ones(variable_shape) * hyperparameter_values[item['scope']]['gradient'])))
                session.run(item['mask_sparse'].assign(
                    np.float32(np.ones(variable_shape) * hyperparameter_values[item['scope']]['sparse'])))
                session.run(item['mask_prior'].assign(
                    np.float32(np.ones(variable_shape) * hyperparameter_values[item['scope']]['prior'])))

    def check_parameter_profile_item(self, parameter_profile, item_name, session=None):
        session = session or self.default_session
        item = parameter_profile[item_name]
        print('\'' + item_name + '\'' + ' profile')
        for keyword in item:
            if isinstance(item[keyword], tf.Variable):
                print(keyword + ': ', session.run(item[keyword]))
            else:
                print(keyword + ': ', item[keyword])

    def get_trainable_parameter_names_in_graph(self):
        return [item.name for item in tf.trainable_variables()]

    # return [var.name for (_,var) in opt_calculate_gradient]

    def set_up_parameter_profile(self, graph, names, mask_value_gradient=None, mask_value_sparse=None,
                                 mask_value_prior=None):

        if mask_value_gradient == None:
            mask_value_gradient = 1
        if mask_value_sparse == None:
            mask_value_sparse = 0
        if mask_value_prior == None:
            mask_value_prior = 0

        parameters = {}
        for name in names:
            tmp = type('container', (object,), {})()
            tmp.name = name
            tmp.shape = graph.get_tensor_by_name(name).get_shape()
            tmp.masks = type('container', (object,), {})()
            tmp.masks.gradient = np.ones(tmp.shape) * mask_value_gradient
            tmp.masks.sparse = np.ones(tmp.shape) * mask_value_sparse
            tmp.masks.prior = np.ones(tmp.shape) * mask_value_prior
            parameters[name] = tmp
        return parameters

    def append_masks_to_feed_dict(self, dr, feed_dict, variable_profile_dict):
        for idx, name in enumerate(variable_profile_dict):
            feed_dict[dr.masks.gradient[name]] = variable_profile_dict[name].masks.gradient
            feed_dict[dr.masks.sparse[name]] = variable_profile_dict[name].masks.sparse

    def MSE_loss_np(self, array1, array2):
        MSE = np.mean((array1.flatten() - array2.flatten()) ** 2)
        return MSE

    def rMSE(self, x_hat, x_true):
        return tf.div(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x_hat, x_true)))), \
                      tf.sqrt(tf.reduce_mean(tf.square(tf.constant(x_true, dtype=tf.float32)))))
