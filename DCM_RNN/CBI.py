import numpy as np
import tensorflow as tf
import math as mth
import pandas as pd
from colorama import Fore, Style
from IPython.display import display
import statistics as st

'''
unified states representation:
state[i_region][i_state][i_time]
'''


class Project(dict):
    def __init__(self, parameters):
        para = parameters
        if 'key1' in dict.keys():
            pass



    def plan_an_experiment(self):
        pass




class configure_a_scanner:

	def __init__(self, t_delta=None,n_stimuli=None):
		###############################
		# set parameters 
		# Global config variables
		self.n_region=3  # number of brain regions in consideration
		self.n_steps = 10 # number of truncated backprop steps ('n' in the discussion above)
		self.n_stimuli = n_stimuli or 1
		# self.state_size = 4
		# self.learning_rate = 0.1
		self.t_delta = t_delta or 0.25
		self.n_time_point=int(6*60/self.t_delta)
		self.n_time_point=int(mth.ceil(self.n_time_point/48.)*48)	# make sure length can be cut into different length

		# used for create stimuli
		self.t_exitation=4 # the duration of a stimulus
		self.t_blank_interval=6 # the resting interval between stimuli
		self.u_probability=0.5
		 # the probability a stimulus is given at a time point
		self.n_exitation=int(self.t_exitation/self.t_delta) 
		self.n_blank_interval=int(self.t_blank_interval/self.t_delta) 

		# standard stimulus
		self.u_stand=np.zeros((self.n_stimuli,self.n_time_point))

		# configured stimulus
		self.u=np.zeros((self.n_stimuli,self.n_time_point))
		self.u=self.create_stimulus()
	
	###############################
	# function definition
	def create_stimulus(self,n_stimuli=None,n_time_point=None,u_probability=None,
		n_exitation=None,n_blank_interval=None):
		# create a random input u
		n_stimuli = n_stimuli or self.n_stimuli
		n_time_point = n_time_point or self.n_time_point
		u_probability = u_probability or self.u_probability
		n_exitation = n_exitation or self.n_exitation
		n_blank_interval = n_blank_interval or self.n_blank_interval
		for stim in range(n_stimuli):
			for i in range(n_blank_interval,n_time_point,n_blank_interval):
				tmp=np.random.random(1)
				if tmp > 1-u_probability/n_stimuli:
					self.u[stim,i-n_exitation:i]=1
		self.u[0,0:n_exitation]=1
		u=self.u[:]
		return u
	
	def subject_preparation(self,sub):
		sub.__init__(t_delta=self.t_delta,n_region=self.n_region,\
			n_time_point=self.n_time_point,n_stimuli=self.n_stimuli,flags=sub.flags)

	def x_evolve(self, sub, u=None, x_state_initial=None):
		# neural response under the stimuli
		u = u or self.u
		x_state_initial = x_state_initial or sub.x_state_initial
		x_state = sub.x_state

		x_state[:,0,0]=x_state_initial
		for i in range(1,self.n_time_point):
			tmp1 = np.matmul(sub.Wxx,x_state[:,0,i-1]);
			tmp2 = [np.matmul(sub.Wxxu[idx],x_state[:,0,i-1]*u[idx,i-1]) for idx in range(sub.n_stimuli)]
			tmp2 = np.sum(np.asarray(tmp2),0)
			tmp3 = np.matmul(sub.Wxu,u[:,i-1])
			x_state[:,0,i]=tmp1+tmp2+tmp3
			'''
		    x_state[:,0,i]=np.matmul(sub.Wxx,x_state[:,0,i-1])+\
		    np.matmul(sub.Wxxu,x_state[:,0,i-1]*u[i-1])+\
		    sub.Wxu*u[i-1]
		    '''
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
				alpha = sub.hemodynamic_parameters.loc['region_'+str(n),'alpha']
				E0 = sub.hemodynamic_parameters.loc['region_'+str(n),'E0']
				h_state[n,:,t]=(np.matmul(sub.Whh[n],sub.phi_h(h_state[n,:,t-1],alpha,E0)).reshape(4,1)\
				+sub.Whx[n]*x_state[n,0,t-1]+sub.bh[n]).reshape(4)
		output=h_state[:]
		return output
	
	def add_noise(self,target_signal,SNR=2):
		signal_shape = target_signal.shape
		noise = np.random.normal(0,np.sqrt(st.variance(target_signal.flatten()))/SNR,signal_shape)
		return target_signal + noise

	def h_impulse_response(self,sub):
		t_duration = 15.
		n_duration = int(t_duration/self.t_delta)
		h_state_initial=np.ones((self.n_region,4))
		h_state_initial[:,0]=0
		x_state = np.zeros((self.n_region,1,n_duration))
		x_state[:,0,0] = 1
		h_state = np.zeros((self.n_region,4,n_duration))

		h_state[:,:,0]=h_state_initial
		h_state[:,:,0]=h_state_initial
		for t in range(1,n_duration):
			for n in range(0,self.n_region):
				alpha = sub.hemodynamic_parameters.loc['region_'+str(n),'alpha']
				E0 = sub.hemodynamic_parameters.loc['region_'+str(n),'E0']
				h_state[n,:,t]=(np.matmul(sub.Whh[n],sub.phi_h(h_state[n,:,t-1],alpha,E0)).reshape(4,1)\
				+sub.Whx[n]*x_state[n,0,t-1]+sub.bh[n]).reshape(4)

		y_output = np.zeros((self.n_region,1,n_duration))
		for t in range(0,n_duration):
		    for n in range(0,self.n_region): 
		        y_output[n,0,t]=np.matmul(sub.Wo[n],sub.phi_o(h_state[n,:,t]))+sub.bo[n]
		return y_output

	def f_evolve(self, sub, h_state=None):
		# given h_state, find observable fMRI signal
		h_state = h_state or sub.h_state
		f_output=sub.f_output

		#f_output[:,0,0]=sub.f_output_initial
		for t in range(0,self.n_time_point):
		    for n in range(0,self.n_region): 
		        f_output[n,0,t]=np.matmul(sub.Wo[n],sub.phi_o(h_state[n,:,t]))+sub.bo[n]
		output=f_output[:]
		return output

	def quick_scan(self, sub, return_x=False, return_h=False):
		# scan with all default setting
		self.subject_preparation(sub)
		self.x_evolve(sub)
		self.h_evolve(sub)
		self.f_evolve(sub)
		fmri_noised = self.add_noise(sub.f_output,)
		sub.f_output_noised = fmri_noised
		output = [self.u, sub.f_output, fmri_noised]
		if return_x:
			output.append(sub.x_state)
		if return_h:
			output.append(sub.h_state)
		return output

	def h_state_suspicious(self, h_state):
		flag_list = [h_state[1]<0.2,
		h_state[2]<0.2,
		h_state[3]<0.2,
		h_state[1]>4,
		h_state[2]>2,
		h_state[3]>2]
		return True in flag_list





	



	'''
	####################################################
	# class definition



	####################################
	# initial run when imported or called as a script

	'''
	





