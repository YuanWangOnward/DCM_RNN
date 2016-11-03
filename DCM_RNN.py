import numpy as np
import tensorflow as tf
import math as mth
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

class DCM_RNN:

	def __init__(self,m,n_recurrent_step=None, learning_rate=None):
		#self.batch_size = 1
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
		
		# set connection initial value
		with tf.variable_scope('rnn_cell'):
			self.Wxx_init=np.array([[-1,-0,0],[0,-1,0],[0,0,-1]],dtype=np.float32)*m.t_delta+np.eye(m.n_region,m.n_region,0,dtype=np.float32)
			self.Wxxu_init=[np.array([[0,0,0],[0,0,0],[0,0,0]],dtype=np.float32)*m.t_delta for _ in range(n_stimuli)]
			self.Wxu_init=np.array([0.5,0,0],dtype=np.float32).reshape(3,1)*m.t_delta

	def build_a_model(self):
		[self.input_u, self.input_y_true] = self.add_placeholders()

		# create shared variables in computation graph
		self.hemodynamic_parameters = self.set_up_hemodynamic_parameter_prior()
		self.create_shared_variables()

		# build layers
		self.add_neural_layer()
		self.add_hemodynamic_layer()
		self.add_output_layer()

		# optimizer with gradient manipulate
		# self.names = [item.name for item in tf.all_variables()]
		self.names = [item.name for item in tf.trainable_variables()]

		# masks for penalty weighting and connection support
		self.define_masks()
		
		# define loss
		self.add_loss_prediction()
		self.add_loss_sparse()
		self.add_loss_prior()
		self.collect_losses()
		
		self.calculate_gradients()
		self.process_gradients()
		self.apply_gradients()

	def set_connection_matrices_from_initial(self, isess):
		with tf.variable_scope('rnn_cell'):
			isess.run(self.Wxx.assign(self.Wxx_init))
			isess.run(self.Wxxu.assign(self.Wxxu_init))
			isess.run(self.Wxu.assign(self.Wxu_init))

	def set_connection_matrices(self, isess, Wxx, Wxxu, Wxu):
		with tf.variable_scope('rnn_cell'):
			isess.run(self.Wxx.assign(Wxx))
			for idx, item in enumerate(Wxxu):
				isess.run(self.Wxxu[idx].assign(item))
			isess.run(self.Wxu.assign(Wxu))

	def rnn_cell(self,u_current, x_state_previous):
		n_region = x_state_previous.get_shape()[0]
		n_stimuli = self.n_stimuli
		with tf.variable_scope('rnn_cell', reuse=True):
			Wxx = tf.get_variable("Wxx",[n_region,n_region])
			Wxxu = [tf.get_variable("Wxxu_s"+str(n)) for n in range(n_stimuli)]
			Wxu = tf.get_variable("Wxu",[n_region,n_stimuli])

			tmp1 = tf.matmul(Wxx,x_state_previous)
			tmp2 = [tf.matmul(Wxxu[n]*u_current[n],x_state_previous) for n in range(n_stimuli)]
			tmp2 = tf.add_n(tmp2)
			tmp3 = tf.mul(Wxu,u_current)
			#return tf.add(tf.add(tmp1,tmp2),tmp3)
			return tmp1+tmp2+tmp3

	def phi_h(self, h_state_current,alpha,E0):
		# used to map hemodynamic states into higher dimension
		# for hemodynamic states evolvement
		h_state_augmented=[]
		for i in range(4):
			h_state_augmented.append(h_state_current[i])

		h_state_augmented.append(tf.pow(h_state_current[2],tf.div(1.,alpha)))
		h_state_augmented.append(tf.mul(tf.div(h_state_current[3],h_state_current[2]),h_state_augmented[4]))
		tmp=tf.sub(1.,tf.pow(tf.sub(1.,E0),tf.div(1.,h_state_current[1])))
		tmp=tf.mul(tf.div(tmp,E0),h_state_current[1])
		h_state_augmented.append(tmp)
		return h_state_augmented

	def rnn_cell_h(self,h_state_current,x_state_current,i_region):
		# model the evolving of hemodynamic states {s,f,v,q}
		# this is independent for each region
		# here x_state_current is a scalar for a particular region
		with tf.variable_scope('rnn_cell_h', reuse=True):
			alpha = tf.get_variable('alpha_r'+str(i_region))
			E0 = tf.get_variable('E0_r'+str(i_region))
			k = tf.get_variable('k_r'+str(i_region))
			gamma = tf.get_variable('gamma_r'+str(i_region))
			tao = tf.get_variable('tao_r'+str(i_region))
			t_delta = self.t_delta

			h_state_augmented=self.phi_h(h_state_current,alpha,E0)
			tmp_list = []
			# s
			tmp1 = tf.mul(t_delta,x_state_current)
			tmp2 = tf.mul(tf.sub(tf.mul(t_delta,k),1.),h_state_augmented[0])
			tmp3 = tf.mul(t_delta,tf.mul(gamma,tf.sub(h_state_augmented[1],1.)))
			tmp = tf.sub(tmp1,tf.add(tmp2,tmp3))
			tmp_list.append(tf.reshape(tmp,[1,1]))
			# f
			tmp = tf.add(h_state_augmented[1],tf.mul(t_delta,h_state_augmented[0]))
			tmp_list.append(tf.reshape(tmp,[1,1]))
			# v
			tmp = t_delta*h_state_augmented[1]/tao \
			- t_delta/tao*h_state_augmented[4] \
			+ h_state_augmented[2] 
			tmp_list.append(tf.reshape(tmp,[1,1]))
			# q
			tmp = h_state_augmented[3] \
			+ t_delta/tao*h_state_augmented[6] \
			- t_delta/tao*h_state_augmented[5]
			tmp_list.append(tf.reshape(tmp,[1,1]))
			# concantenate into a tensor
			tmp_list = tf.concat(0,tmp_list)
			tmp_list = tf.reshape(tmp_list,[4,1])
			return tmp_list

	def phi_o(self, h_state_current):
		# used to map hemodynamic states into higher dimension
		# for fMRI output
		o_state_augmented = [h_state_current[i+2] for i in range(2)]
		tmp = tf.div(o_state_augmented[1],o_state_augmented[0])
		o_state_augmented.append(tmp)
		return o_state_augmented

	def output_mapping(self,h_state_current,i_region):
		with tf.variable_scope('rnn_cell_h', reuse=True):
			E0 = tf.get_variable('E0_r'+str(i_region))
			epsilon = tf.get_variable('epsilon_r'+str(i_region))
			V0 = tf.get_variable('V0_r'+str(i_region))
			TE = tf.get_variable('TE_r'+str(i_region))
			r0 = tf.get_variable('r0_r'+str(i_region))
			theta0 = tf.get_variable('theta0_r'+str(i_region))
			k1 = 4.3*theta0*E0*TE
			k2 = epsilon*r0*E0*TE
			k3 = 1-epsilon

			o_state_augmented = self.phi_o(h_state_current)

			y = V0*k1*(1-o_state_augmented[1])\
			+ V0*k2*(1-o_state_augmented[2])\
			+ V0*k3*(1-o_state_augmented[0])

			return y

	def add_placeholders(self):
		input_u = tf.placeholder(tf.float32, [self.n_stimuli, self.n_recurrent_step], name='input_u')
		input_y_true = tf.placeholder(tf.float32, [self.n_region, self.n_recurrent_step], name='input_y_true')
		return [input_u, input_y_true]

	def get_random_h_state_initial(self):
		h_state_initial = np.zeros((4,1))
		h_state_initial[0] = np.random.normal(loc=0, scale=0.3)	
		h_state_initial[1] = np.random.normal(loc=1.5, scale=0.5)	
		h_state_initial[2] = np.random.normal(loc=1.15, scale=0.15)	
		h_state_initial[3] = np.random.normal(loc=0.85, scale=0.15)	

	def set_up_hemodynamic_parameter_prior(self):
		hemodynamic_parameters={}
		hemodynamic_parameters['alpha']=type('container', (object,), {})()
		hemodynamic_parameters['alpha'].mean = 0.32
		hemodynamic_parameters['alpha'].std = np.sqrt(0.0015)
		hemodynamic_parameters['E0']=type('container', (object,), {})()
		hemodynamic_parameters['E0'].mean = 0.34
		hemodynamic_parameters['E0'].std = np.sqrt(0.0024)
		hemodynamic_parameters['k']=type('container', (object,), {})()
		hemodynamic_parameters['k'].mean = 0.65
		hemodynamic_parameters['k'].std = np.sqrt(0.015)
		hemodynamic_parameters['gamma']=type('container', (object,), {})()
		hemodynamic_parameters['gamma'].mean = 0.41
		hemodynamic_parameters['gamma'].std = np.sqrt(0.002)
		hemodynamic_parameters['tao']=type('container', (object,), {})()
		hemodynamic_parameters['tao'].mean = 0.98
		hemodynamic_parameters['tao'].std = np.sqrt(0.0568)
		hemodynamic_parameters['epsilon']=type('container', (object,), {})()
		hemodynamic_parameters['epsilon'].mean = 0.4
		hemodynamic_parameters['epsilon'].std = np.finfo(float).eps
		hemodynamic_parameters['V0']=type('container', (object,), {})()
		hemodynamic_parameters['V0'].mean = 100.
		hemodynamic_parameters['V0'].std = np.finfo(float).eps
		hemodynamic_parameters['TE']=type('container', (object,), {})()
		hemodynamic_parameters['TE'].mean = 0.03
		hemodynamic_parameters['TE'].std = np.finfo(float).eps
		hemodynamic_parameters['r0']=type('container', (object,), {})()
		hemodynamic_parameters['r0'].mean = 25.
		hemodynamic_parameters['r0'].std = np.finfo(float).eps
		hemodynamic_parameters['theta0']=type('container', (object,), {})()
		hemodynamic_parameters['theta0'].mean = 40.3
		hemodynamic_parameters['theta0'].std = np.finfo(float).eps
		return hemodynamic_parameters

	def create_shared_variables(self):
		# for neural level
		with tf.variable_scope('rnn_cell'):
			trainable_flag=True
			self.Wxx = tf.get_variable('Wxx',initializer=self.Wxx_init,trainable=trainable_flag)
			self.Wxxu = [tf.get_variable('Wxxu_s'+str(n),initializer=self.Wxxu_init[n],trainable=trainable_flag) for n in range(self.n_stimuli)]
			self.Wxu = tf.get_variable('Wxu',initializer=self.Wxu_init,trainable=trainable_flag)

		# for hemodynamic level
		with tf.variable_scope('rnn_cell_h'):
			self.alpha={}
			self.E0={}
			self.k={}
			self.gamma={}
			self.tao={}
			self.epsilon={}
			self.V0={}
			self.TE={}
			self.r0={}
			self.theta0={}
			trainable_flag=False
			for n in range(self.n_region):
				self.alpha['alpha_r'+str(n)]=tf.get_variable('alpha_r'+str(n),initializer=self.hemodynamic_parameters['alpha'].mean,trainable=trainable_flag)
				self.E0['E0_r'+str(n)]=tf.get_variable('E0_r'+str(n),initializer=self.hemodynamic_parameters['E0'].mean,trainable=trainable_flag)
				self.k['k_r'+str(n)]=tf.get_variable('k_r'+str(n),initializer=self.hemodynamic_parameters['k'].mean,trainable=trainable_flag)
				self.gamma['gamma_r'+str(n)]=tf.get_variable('gamma_r'+str(n),initializer=self.hemodynamic_parameters['gamma'].mean,trainable=trainable_flag)
				self.tao['tao_r'+str(n)]=tf.get_variable('tao_r'+str(n),initializer=self.hemodynamic_parameters['tao'].mean,trainable=trainable_flag)
				self.epsilon['epsilon_r'+str(n)]=tf.get_variable('epsilon_r'+str(n),initializer=self.hemodynamic_parameters['epsilon'].mean,trainable=False) # This is set to untrainable by design
				self.V0['V0_r'+str(n)]=tf.get_variable('V0_r'+str(n),initializer=self.hemodynamic_parameters['V0'].mean,trainable=False) # This is set to untrainable by design
				self.TE['TE_r'+str(n)]=tf.get_variable('TE_r'+str(n),initializer=self.hemodynamic_parameters['TE'].mean,trainable=False)	# This is set to untrainable by design
				self.r0['r0_r'+str(n)]=tf.get_variable('r0_r'+str(n),initializer=self.hemodynamic_parameters['r0'].mean,trainable=False) # This is set to untrainable by design
				self.theta0['theta0_r'+str(n)]=tf.get_variable('theta0_r'+str(n),initializer=self.hemodynamic_parameters['theta0'].mean,trainable=False) # This is set to untrainable by design

	def add_neural_layer(self):
		self.x_state_initial = tf.zeros((self.n_region,1),dtype=np.float32)

		self.x_state_predicted =[]
		self.x_state_predicted.append(self.x_state_initial)
		for i in range(1,self.n_recurrent_step):
			tmp = self.rnn_cell(self.input_u[0,i-1], self.x_state_predicted[i-1])
			self.x_state_predicted.append(tmp)
		# the last element needs special handling
		i=self.n_recurrent_step
		self.x_state_final = self.rnn_cell(self.input_u[0,i-1], self.x_state_predicted[i-1])

	def add_hemodynamic_layer(self):
		n_region = self.n_region
		self.h_state_initial = [tf.get_variable('h_state_initial_r'+str(n),shape=[4,1],initializer=self.get_random_h_state_initial(),trainable=False) \
								for n in range(n_region)]

		# format: h_state_predicted[region][time]
		self.h_state_predicted = [[] for _ in range(n_region)]
		for n in range(n_region):
			self.h_state_predicted[n].append(self.h_state_initial[n])
		for i in range(1,self.n_recurrent_step):
			for n in range(n_region):
				self.h_state_predicted[n].append(\
					self.rnn_cell_h(self.h_state_predicted[n][i-1],\
						self.x_state_predicted[i-1][n],n))
		# the last element needs special handling
		i=self.n_recurrent_step
		self.h_state_final=[]
		for n in range(n_region):
			self.h_state_final.append(\
				self.rnn_cell_h(self.h_state_predicted[n][i-1],\
					self.x_state_predicted[i-1][n],n))

	def add_output_layer(self):
		self.y_state_predicted = []
		for i in range(0,self.n_recurrent_step):
			tmp = []
			for n in range(self.n_region):
				tmp.append(self.output_mapping(self.h_state_predicted[n][i],n))
			tmp = tf.pack(tmp)
			self.y_state_predicted.append(tmp)

	def define_masks(self):
		names=self.names
		self.masks = type('container', (object,), {})()
		self.masks.gradient = {}
		self.masks.sparse = {}
		self.masks.prior = {}
		for idx,name in enumerate(names):
			tmp=tf.get_default_graph().get_tensor_by_name(name).get_shape()
			self.masks.gradient[name] = tf.placeholder(tf.float32, tmp, name='mask_gradient_'+str(idx))
			self.masks.sparse[name] = tf.placeholder(tf.float32, tmp, name='mask_sparse_'+str(idx))
			self.masks.prior[name] = tf.placeholder(tf.float32, tmp, name='mask_prior_'+str(idx))	

	def add_loss_prediction(self):
		self.y_true_as_list =[tf.reshape(self.input_y_true[:,i],(self.n_region,1)) for i in range(self.n_recurrent_step)]
		self.loss_y_list= [(tf.reduce_mean(tf.square(tf.sub(y_pred, y_true)))) \
				for y_pred, y_true in zip(self.y_state_predicted,self.y_true_as_list)]
		self.loss_y = tf.reduce_mean(self.loss_y_list)

	def add_loss_sparse(self):
		# got all variable values
		variable_names = self.names
		variable_values = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		self.loss_sparse_list = [tf.reduce_sum(tf.reshape(tf.abs(value*self.masks.sparse[name]),[-1])) for name,value in zip(variable_names, variable_values)]	
		self.loss_sparse = tf.add_n(self.loss_sparse_list)

	def add_loss_prior(self):
		self.loss_prior=0
		self.hemo_parameter_mean_list=[0.32, 0.34, 0.65, 0.41, 0.98]
		self.hemo_parameter_std_list=np.diag(np.sqrt([0.0015, 0.0024, 0.015, 0.002, 0.0568]))
		for name in self.names:
			tmp=tf.get_default_graph().get_tensor_by_name(name)
			#if 'alpha' in name:
			#	pass

	def collect_losses(self):
		self.loss_total = self.loss_y + self.loss_sparse + self.loss_prior

	def calculate_gradients(self):
		self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
		self.grads_and_vars = self.opt.compute_gradients(self.loss_total)

	def process_gradients(self):
		self.processed_grads_and_vars = [(gv[0]*self.masks.gradient[self.names[idx]], gv[1]) for idx,gv in enumerate(self.grads_and_vars)]

	def apply_gradients(self):
		self.apply_gradient = self.opt.apply_gradients(self.processed_grads_and_vars)


class utilities:

	def __init__(self,n_region=None,n_recurrence=None, learning_rate=None):
		'''
		self.n_region = n_region or 3
		self.n_recurrent_step = n_recurrence or 8 
		self.learning_rate = learning_rate or 0.01
		self.n_region=m.n_region
		'''
		#self.default_dr = dr
		self.parameter_key_list = ['Wxx','Wxxu','Wxu','alpha','E0','k','gamma','tao','epsilon','V0','TE','r0','theta0']

	def run_forward_segment(self,dr,sess,feed_dict_in):
		x_state, x_state_final = sess.run([dr.x_state_predicted,dr.x_state_final],\
		                                   feed_dict={dr.input_u:feed_dict_in['input_u'],\
		                                   dr.rnn_x:feed_dict_in['rnn_x'],
		                                   dr.x_state_initial:feed_dict_in['x_state_initial']})
		return [x_state, x_state_final]

	def forward_pass_x(self, dr, dh, isess, x_state_initial=None):
		training_state = x_state_initial or np.zeros((dr.n_region,1))

		x_state_predicted=[]
		for i in range(len(dh.u_list)):
			tmp,training_state = isess.run([dr.x_state_predicted,dr.x_state_final],\
			feed_dict={dr.input_u:dh.u_list[i],
			#dr.rnn_x:dh.x_list[i],
			dr.x_state_initial:training_state})
			tmp=np.asarray(tmp)
			tmp=tmp[:,:,0]
			x_state_predicted.append(tmp)	
		x_state_predicted=np.concatenate(x_state_predicted).transpose()
		self.x_state_predicted=x_state_predicted[:]
		return x_state_predicted
	
	def show_x(self,t_delta=1,length=None):
		length = length or self.x_state_predicted.shape[1]
		plt.plot(np.arange(length)*t_delta,self.x_state_predicted[0,0:length].transpose())

	def forward_pass_h(self, dr, dh, isess, x_state_initial=None, h_state_initial=None):
		x_state_feed = x_state_initial or np.zeros((dr.n_region,1))
		h_state_feed = h_state_initial or [np.array([0,1,1,1],dtype=np.float32).reshape(4,1) for _ in range(dr.n_region)]
		
		h_state_predicted = [[] for _ in range(3)]

		for i in range(len(dh.u_list)):
			# build feed_dictionary
			feed_dict={i: d for i, d in zip(dr.h_state_initial, h_state_feed)}
			feed_dict[dr.x_state_initial]=x_state_feed
			feed_dict[dr.input_u]=dh.u_list[i]
			# run 
			h_current_segment,h_state_feed,x_state_feed = isess.run([dr.h_state_predicted,dr.h_state_final,dr.x_state_final],\
			feed_dict=feed_dict)
			

			for n in range(dr.n_region):
				# concatenate h_current_segment in to list of 3 element, each a np.narray
				h_current_segment[n]=np.squeeze(np.asarray(h_current_segment[n])).transpose()
				#h_state_predicted[n].append(h_current_segment[n])
				if i==0:
					h_state_predicted[n]=h_current_segment[n]
				else:
					h_state_predicted[n]=np.concatenate((h_state_predicted[n],h_current_segment[n]),axis=1)
		return h_state_predicted

	def forward_pass_y(self,dr,dh,isess,x_state_initial=None, h_state_initial=None):
		x_state_feed = x_state_initial or np.zeros((dr.n_region,1))
		h_state_feed = h_state_initial or [np.array([0,1,1,1],dtype=np.float32).reshape(4,1) for _ in range(dr.n_region)]
		
		y_output_predicted = []
		for i in range(len(dh.u_list)):
			# build feed_dictionary
			feed_dict={i: d for i, d in zip(dr.h_state_initial, h_state_feed)}
			feed_dict[dr.x_state_initial]=x_state_feed
			feed_dict[dr.input_u]=dh.u_list[i]
			# run 
			y_current_segment,h_state_feed,x_state_feed = isess.run([dr.y_state_predicted,dr.h_state_final,dr.x_state_final],\
			feed_dict=feed_dict)
			# orgnize output
			# print(len(y_current_segment))
			# print(len(y_current_segment[0]))
			y_current_segment = np.asarray(y_current_segment)
			y_current_segment = np.squeeze(y_current_segment)
			y_output_predicted.append(y_current_segment)
		y_output_predicted=np.concatenate(y_output_predicted).transpose()
		self.y_output_predicted=y_output_predicted[:]
		return y_output_predicted

	def show_all_variable_value(self, dr, isess, visFlag=False):
		output=[]
		output_buff = pd.DataFrame()
		#variables= self.parameter_key_list
		#print(variables)
		#values=eval('isess.run(['+', '.join(variables)+'])')
		for idx, key in enumerate(self.parameter_key_list):
			if key == 'Wxx':
				values = isess.run(dr.Wxx)
				tmp=pd.DataFrame(values,index=['To_r'+str(i) for i in range(dr.n_region)],\
                   columns=['From_r'+str(i) for i in range(dr.n_region)])
				tmp.name=key
				output.append(tmp)
			elif key == 'Wxxu':
				values = isess.run(dr.Wxxu)
				for n in range(dr.n_stimuli):
					tmp=pd.DataFrame(values[n],index=['To_r'+str(i) for i in range(dr.n_region)],\
	                   columns=['From_r'+str(i) for i in range(dr.n_region)])
					tmp.name=key+'_s'+str(n)
					output.append(tmp)
			elif key == 'Wxu':
				values = isess.run(dr.Wxu)
				tmp = pd.DataFrame(values,index=['To_r'+str(i) for i in range(dr.n_region)],\
                   columns=['stimuli_'+str(i) for i in range(dr.n_stimuli)])
				tmp.name=key
				output.append(tmp)
			else:
				values = eval('isess.run(dr.'+key+')')
				#print(key)
				#print(values)
				tmp = [values[key+'_r'+str(i)] for i in range(dr.n_region)]
				tmp = pd.Series(tmp,index=['region_'+str(i) for i in range(dr.n_region)])
				output_buff[key] = tmp
		output_buff.name='hemodynamic_parameters'
		output.append(output_buff)
		if visFlag:
			for item in output:
				print(item.name)
				display(item)
		return output

	def compare_parameters(self, set1, set2, visFlag=True,parameter_list=None):
		if parameter_list ==None:
			name_list1 = [set1[i].name for i in range(len(set1))]
			name_list2 = [set2[i].name for i in range(len(set2))]
			name_list = list(set(name_list1)&set(name_list2))	#common name list
		else:
			name_list = parameter_list
		output = []
		for name in name_list:
			tmp1 = next((x for x in set1 if x.name == name), None)
			tmp2 = next((x for x in set2 if x.name == name), None)
			if tmp1.shape[0]>=tmp1.shape[1]:
				tmp = pd.concat([tmp1, tmp2, tmp1-tmp2], axis=1, join_axes=[tmp1.index],keys=['set1', 'set2','difference'])
			else:
				tmp = pd.concat([tmp1, tmp2, tmp1-tmp2], axis=0, join_axes=[tmp1.columns],keys=['set1', 'set2','difference'])
			tmp.name=name
			output.append(tmp)
		if visFlag:
			for item in output:
				print(item.name)
				display(item)
		return output

	def set_connection_matrices(self, dr,isess, Wxx, Wxxu, Wxu):
		with tf.variable_scope('rnn_cell'):
			isess.run(dr.Wxx.assign(Wxx))
			for idx, item in enumerate(Wxxu):
				isess.run(dr.Wxxu[idx].assign(item))
			isess.run(dr.Wxu.assign(Wxu))

	def get_trainable_parameter_names_in_graph(self):
		return [item.name for item in tf.trainable_variables()]
		#return [var.name for (_,var) in opt_calculate_gradient]

	def set_up_parameter_profile(self,graph,names,mask_value_gradient=None,mask_value_sparse=None,mask_value_prior=None):
		
		if mask_value_gradient == None:
			mask_value_gradient = 1
		if mask_value_sparse == None:
			mask_value_sparse = 0
		if mask_value_prior == None:
			mask_value_prior = 0

		parameters={}
		for name in names:
			tmp=type('container', (object,), {})()
			tmp.name=name
			tmp.shape=graph.get_tensor_by_name(name).get_shape()
			tmp.masks = type('container', (object,), {})()
			tmp.masks.gradient = np.ones(tmp.shape)*mask_value_gradient
			tmp.masks.sparse = np.ones(tmp.shape)*mask_value_sparse
			tmp.masks.prior = np.ones(tmp.shape)*mask_value_prior
			parameters[name] = tmp
		return parameters

	def append_masks_to_feed_dict(self,dr,feed_dict,variable_profile_dict):
		for idx,name in enumerate(variable_profile_dict):
			feed_dict[dr.masks.gradient[name]]=variable_profile_dict[name].masks.gradient
			feed_dict[dr.masks.sparse[name]]=variable_profile_dict[name].masks.sparse

	def MSE_loss_np(self,array1,array2):
		MSE=np.mean((array1.flatten()- array2.flatten()) ** 2)
		return MSE

	def rMSE(self, x_hat, x_true):
		return tf.div(tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x_hat, x_true)))),\
                 tf.sqrt(tf.reduce_mean(tf.square(tf.constant(x_true,dtype=tf.float32)))))
		




	









