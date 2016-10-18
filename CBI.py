import numpy as np
import tensorflow as tf
import math as mth

class configure_a_scanner:

	def __init__(self):
		###############################
		# set parameters 
		# Global config variables
		self.n_region=3  # number of brain regions in consideration
		self.n_steps = 10 # number of truncated backprop steps ('n' in the discussion above)
		self.n_stimuli=1
		self.state_size = 4
		#self.learning_rate = 0.1
		self.t_delta=0.25
		self.n_time_point=int(5*60/self.t_delta)
		self.n_time_point=int(mth.ceil(self.n_time_point/128.)*128)

		# used for create stimuli
		self.t_exitation=1 # the duration of a stimulus
		self.t_blank_interval=5 # the resting interval between stimuli
		self.u_probability=0.5 # the probability a stimulus is given at a time point
		self.n_exitation=int(self.t_exitation/self.t_delta) 
		self.n_blank_interval=int(self.t_blank_interval/self.t_delta) 

		# standard stimulus
		self.u_stand=np.zeros(self.n_time_point)

		# configured stimulus
		self.u=np.zeros(self.n_time_point)
		self.u=self.create_stimulus()
	
	###############################
	# function definition
	def create_stimulus(self,n_stimuli=None,n_time_point=None,u_probability=None,\
		n_exitation=None,n_blank_interval=None):
		# create a random input u
		n_stimuli = n_stimuli or self.n_stimuli
		n_time_point = n_time_point or self.n_time_point
		u_probability = u_probability or self.u_probability
		n_exitation = n_exitation or self.n_exitation
		n_blank_interval = n_blank_interval or self.n_blank_interval
		
		for i in range(n_blank_interval,n_time_point,n_blank_interval):
			tmp=np.random.random(1)
			if tmp > 1-u_probability:
				self.u[i-n_exitation:i]=1
		self.u[0:n_exitation]=1
		u=self.u[:]
		return u
		

	def print_test(self):
		print(self.n_region)
		print('print works')
	
	def subject_preparation(self,sub):
		sub.__init__(t_delta=self.t_delta,n_region=self.n_region,n_time_point=self.n_time_point)

	
	def x_evolve(self, sub, u=None, x_state_initial=None):
		# neural response under the stimuli
		u = u or self.u
		x_state_initial = x_state_initial or sub.x_state_initial
		x_state = sub.x_state

		x_state[:,0,0]=x_state_initial
		for i in range(1,self.n_time_point):
		    # easy approximation
		    x_state[:,0,i]=np.matmul(sub.Wxx,x_state[:,0,i-1])+\
		    np.matmul(sub.Wxxu,x_state[:,0,i-1]*u[i-1])+\
		    sub.Wxu*u[i-1]
		output=x_state[:]
		return output
	
	
	def h_evolve(self, sub, h_state_initial=None, x_state=None):
		# hemodynamic response 
		h_state_initial = h_state_initial or sub.h_state_initial
		x_state = x_state or sub.x_state
		h_state = sub.h_state

		h_state[:,:,0]=h_state_initial
		for t in range(1,self.n_time_point):
		    for n in range(0,self.n_region):
		        h_state[n,:,t]=np.matmul(sub.Whh,sub.phi_h(h_state[n,:,t-1]))+\
		        sub.Whx*x_state[n,0,t-1]+sub.bh
		output=h_state[:]
		return output
	
	def f_evolve(self, sub, h_state=None):
		# given h_state, find observable fMRI signal
		h_state = h_state or sub.h_state
		f_output=sub.f_output

		f_output[:,0,0]=sub.f_output_initial
		for t in range(1,self.n_time_point):
		    for n in range(0,self.n_region): 
		        f_output[n,0,t]=np.matmul(sub.Wo,sub.phi_o(h_state[n,:,t]))+sub.bo
		output=f_output[:]
		return output

	def quick_scan(self, sub, return_x=False, return_h=False):
		# scan with all default setting
		self.subject_preparation(sub)
		self.x_evolve(sub)
		self.h_evolve(sub)
		self.f_evolve(sub)
		output = [self.u, sub.f_output]
		if return_x:
			output.append(sub.x_state)
		if return_h:
			output.append(sub.h_state)
		return output

	'''
	def quick_scan_with_downsample(self, sub, t_delta=2, return_x=False, return_h=False):
		# scan with all default setting
		# down sample to practical temporal resolution
		self.subject_preparation(sub)
		self.x_evolve(sub)
		self.h_evolve(sub)
		self.f_evolve(sub)


	'''

	



	'''
	####################################################
	# class definition



	####################################
	# initial run when imported or called as a script

	'''
	





