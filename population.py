import numpy as np
import scipy as sp
#import tensorflow as tf
import math as mth
import pandas as pd
import scipy.stats
from IPython.display import display


class get_a_subject:
	
	# global parameters, need to be changed before scanning
	t_delta=0.25
	n_region=3
	n_time_point=128
	n_stimuli = 1


	def __init__(self,flags=None,t_delta=None,n_region=None,n_time_point=None,n_stimuli=None):
		self.t_delta = t_delta or 0.25
		self.n_region = n_region or 3
		self.n_time_point = n_time_point or 128
		self.n_stimuli = n_stimuli or 1

		self.flags = self.add_default_flags(flags)

		t_delta = self.t_delta
		n_region = self.n_region
		n_time_point = self.n_time_point
		n_stimuli = self.n_stimuli

	
		# neural parameters
		self.Wxx=np.array([	[-1,0,0],
		              		[0.8,-1,0.4],
		              		[0.4,0.8,-1]],dtype=np.float32)*t_delta+np.eye(n_region,n_region,0,dtype=np.float32)
		self.Wxxu=[np.array([[0,0,0],
		              [0,0,0],
		              [0,0,-0.4]],dtype=np.float32)*t_delta for _ in range(n_stimuli)]
		self.Wxu=np.eye(n_region,n_stimuli,dtype=np.float32)*0.4*t_delta 
		#np.zeros((n_region,n_stimuli),dtype=np.float32)
		#self.Wxu[]
		#np.array([0.8,0,0],dtype=np.float32).reshape(n_region,n_stimuli)*t_delta 

		# Hemodynamic parameters	
		self.hemodynamic_parameters_mean = pd.DataFrame()
		self.hemodynamic_parameters_variance = pd.DataFrame()
		
		self.neural_parameter_key_list = ['Wxx','Wxxu', 'Wxu']
		self.hemo_parameter_key_list=['alpha','E0','k','gamma','tao','epsilon','V0','TE','r0','theta0']
		self.hemo_parameter_mean_list=[0.32, 0.34, 0.65, 0.41, 0.98, 0.4, 100., 0.03, 25, 40.3]
		self.hemo_parameter_variance_list=[0.0015, 0.0024, 0.015, 0.002, 0.0568, 0., 0., 0., 0., 0.]
		#self.hemo_parameter_variance_list=[num**2 for num in self.hemo_parameter_variance_list]

		for idx, key in enumerate(self.hemo_parameter_key_list):
			# add mean
			tmp = [self.hemo_parameter_mean_list[idx] for _ in range(n_region)]
			tmp = pd.Series(tmp, index=['region_'+str(i) for i in range(n_region)])
			self.hemodynamic_parameters_mean[key] = tmp
			# add variance
			tmp = [self.hemo_parameter_variance_list[idx] for _ in range(n_region)]
			tmp = pd.Series(tmp, index=['region_'+str(i) for i in range(n_region)])
			self.hemodynamic_parameters_variance[key] = tmp

		if self.flags.random_hemodynamic_parameter:
			self.hemodynamic_parameters, self.hemodynamic_parameters_deviation_normalized = self.sample_hemodynamic_parameters()
		else:
			self.hemodynamic_parameters = self.hemodynamic_parameters_mean.copy(True)
			h_shape = self.hemodynamic_parameters.shape
			self.hemodynamic_parameters_deviation_normalized = pd.DataFrame(np.zeros(h_shape[0]*h_shape[1]).reshape(h_shape[0],h_shape[1]),\
																index=['region_'+str(i) for i in range(n_region)],\
																columns=self.hemo_parameter_key_list)
		# create hemodynamic matrices 
		self.create_hemodynamic_matrices()

		# state variables
		if self.flags.random_x_state_initial:
			self.x_state_initial = np.random.normal(loc=0.2, scale=0.2, size=(n_region))
			#self.x_state_initial=np.zeros((n_region))
		else:
			self.x_state_initial=np.zeros((n_region))

		if self.flags.random_h_state_initial:
			self.h_state_initial=np.ones((n_region,4))
			self.h_state_initial[:,0] = np.random.normal(loc=0, scale=0.3, size=(n_region))	
			self.h_state_initial[:,1] = np.random.normal(loc=1.5, scale=0.5, size=(n_region))	
			self.h_state_initial[:,2] = np.random.normal(loc=1.15, scale=0.15, size=(n_region))	
			self.h_state_initial[:,3] = np.random.normal(loc=0.85, scale=0.15, size=(n_region))	
		else:
			self.h_state_initial=np.ones((n_region,4))
			self.h_state_initial[:,0]=0

		self.u=np.zeros((n_stimuli,n_time_point))
		self.x_state=np.zeros((n_region,1,n_time_point))
		self.h_state=np.zeros((n_region,4,n_time_point))
		self.f_output=np.zeros((n_region,1,n_time_point))	


	def add_default_flags(self,flags):
		random_hemodynamic_parameter = False
		random_x_state_initial = False
		random_h_state_initial = False
		if flags == None:
			flags = lambda:None
			flags.random_hemodynamic_parameter = random_hemodynamic_parameter
			flags.random_x_state_initial = random_x_state_initial
			flags.random_h_state_initial = random_h_state_initial
		else:
			if not hasattr(flags, 'random_hemodynamic_parameter'):
				flags.random_hemodynamic_parameter = random_hemodynamic_parameter
			if not hasattr(flags, 'random_x_state_initial'):
				flags.random_x_state_initial = random_x_state_initial
			if not hasattr(flags, 'random_h_state_initial'):
				flags.random_h_state_initial = random_h_state_initial
		return flags


	def phi_h(self, h_state_current,alpha,E0):
		# used to map hemodynamic states into higher dimension
		# for hemodynamic states evolvement
		#alpha = alpha or self.alpha
		#E0 = E0 or self.E0
		h_state_augmented=np.zeros(7)
		h_state_augmented[0:4]=h_state_current
		h_state_augmented[4]=h_state_current[2]**(1/alpha)
		h_state_augmented[5]=h_state_current[3]/(h_state_current[2])*h_state_augmented[4]
		h_state_augmented[6]=(1-(1-E0)**(1/h_state_current[1]))/(E0)*h_state_current[1]
		return h_state_augmented

	def phi_o(self, h_state_current):
		# used to map hemodynamic states into higher dimension
		# for fMRI output
	    o_state_augmented=np.zeros(3)
	    o_state_augmented[0:2]=h_state_current[2:4]
	    o_state_augmented[2]=o_state_augmented[1]/o_state_augmented[0]
	    return o_state_augmented

	def set_parameters(self,t_delta=None,n_region=None,n_time_point=None):
		t_delta = None or self.t_delta
		n_region = None or self.n_region
		n_time_point = None or self.n_time_point

	def sample_hemodynamic_parameters(self, hemodynamic_parameters_mean=None, hemodynamic_parameters_variance=None, deviation_constraint=0.4):
		# sample a subject from hemodynamic parameter distribution
		h_mean = hemodynamic_parameters_mean or self.hemodynamic_parameters_mean
		h_vari = hemodynamic_parameters_variance or self.hemodynamic_parameters_variance
		h_para = h_mean.copy(True)
		h_devi = h_mean.copy(True)

		p_shape = h_mean.shape
		for r in range(p_shape[0]):
			for c in range(p_shape[1]):
				if h_vari.iloc[r,c] > 0:
					h_para.iloc[r,c] = np.random.normal(loc=h_mean.iloc[r,c], scale=mth.sqrt(h_vari.iloc[r,c]))
					#h_devi.iloc[r,c] =  sp.stats.norm(h_mean.iloc[r,c], mth.sqrt(h_vari.iloc[r,c])).pdf(h_para.iloc[r,c])
					h_devi.iloc[r,c] = (h_para.iloc[r,c]-h_mean.iloc[r,c])/mth.sqrt(h_vari.iloc[r,c])
					while abs(h_devi.iloc[r,c]) > deviation_constraint:
						h_para.iloc[r,c] = np.random.normal(loc=h_mean.iloc[r,c], scale=mth.sqrt(h_vari.iloc[r,c]))
						h_devi.iloc[r,c] = (h_para.iloc[r,c]-h_mean.iloc[r,c])/mth.sqrt(h_vari.iloc[r,c])
				else:
					h_devi.iloc[r,c] =  0.0
		return [h_para, h_devi]

	def create_hemodynamic_matrices(self):
		t_delta = self.t_delta

		self.Whh = []
		self.Whx = []
		self.Wo = []
		self.bh = []
		self.bo = []

		for n in range(self.n_region):
			['alpha','E0','k','gamma','tao','epsilon','V0','TE','r0','theta0']
			alpha = self.hemodynamic_parameters.loc['region_'+str(n),'alpha']
			E0 = self.hemodynamic_parameters.loc['region_'+str(n),'E0']
			k = self.hemodynamic_parameters.loc['region_'+str(n),'k']
			gamma = self.hemodynamic_parameters.loc['region_'+str(n),'gamma']
			tao = self.hemodynamic_parameters.loc['region_'+str(n),'tao']
			epsilon = self.hemodynamic_parameters.loc['region_'+str(n),'epsilon']
			V0 = self.hemodynamic_parameters.loc['region_'+str(n),'V0']
			TE = self.hemodynamic_parameters.loc['region_'+str(n),'TE']
			r0 = self.hemodynamic_parameters.loc['region_'+str(n),'r0']
			theta0 = self.hemodynamic_parameters.loc['region_'+str(n),'theta0']

			Whh = np.zeros((4,7))
			Whh[0,0]=-(t_delta*k-1)
			Whh[0,1]=-t_delta*gamma
			Whh[1,0]=t_delta
			Whh[1,1]=1
			Whh[2,1]=t_delta/tao
			Whh[2,2]=1
			Whh[2,4]=-t_delta/tao
			Whh[3,3]=1
			Whh[3,5]=-t_delta/tao
			Whh[3,6]=t_delta/tao
			self.Whh.append(Whh)

			self.Whx.append(np.array([t_delta,0,0,0]).reshape(4,1))

			Wo=np.array([-(1-epsilon)*V0, -4.3*theta0*E0*V0*TE, -epsilon*r0*E0*V0*TE])
			self.Wo.append(Wo)

			bh=np.array(np.asarray([t_delta*gamma,0,0,0]).reshape(4,1))
			self.bh.append(bh)

			bo=V0*(4.3*theta0*E0*TE+epsilon*r0*E0*TE+(1-epsilon))
			self.bo.append(bo)
	
	def show_all_variable_value(self, visFlag=False):
		output=[]
		output_buff = pd.DataFrame()
		for key in self.neural_parameter_key_list:
			if key == 'Wxx':
				tmp = pd.DataFrame(self.Wxx,index=['To_r'+str(i) for i in range(self.n_region)],\
                   columns=['From_r'+str(i) for i in range(self.n_region)])
				tmp.name=key
				output.append(tmp)
			elif key == 'Wxxu':
				for idx,item in enumerate(self.Wxxu) :
					tmp = pd.DataFrame(self.Wxxu[idx],index=['To_r'+str(i) for i in range(self.n_region)],\
                   		columns=['From_r'+str(i) for i in range(self.n_region)])
					tmp.name=key+'_s'+str(idx)
					output.append(tmp) 
			elif key == 'Wxu':
				tmp = pd.DataFrame(self.Wxu,index=['To_r'+str(i) for i in range(self.n_region)],\
                   columns=['stimuli_'+str(i) for i in range(self.n_stimuli)])
				tmp.name=key
				output.append(tmp)
		output.append(self.hemodynamic_parameters)
		output[-1].name='hemodynamic_parameters'
		if visFlag:
			for item in output:
				print(item.name)
				display(item)
		return output 

