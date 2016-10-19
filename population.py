import numpy as np
#import tensorflow as tf
import math as mth

class get_a_standard_subject:

	# global parameters, need to be changed before scanning
	t_delta=0.25
	n_region=3
	n_time_point=128
	n_stimuli = 1


	def phi_h(self, h_state_current,alpha=None,E0=None):
		# used to map hemodynamic states into higher dimension
		# for hemodynamic states evolvement
		alpha = alpha or self.alpha
		E0 = E0 or self.E0
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



	def __init__(self,t_delta=None,n_region=None,n_time_point=None,n_stimuli=None):
		t_delta = t_delta or self.t_delta
		n_region = n_region or self.n_region
		n_time_point = n_time_point or self.n_time_point
		n_stimuli = n_stimuli or self.n_stimuli
		#if n_time_point == None:

		self.t_delta=t_delta
		self.n_region=n_region
		self.n_time_point=n_time_point

		#print(t_delta)
		#print(n_region)
		#print(n_time_point)

		# neural parameters
		self.Wxx=np.array([[-0.65,-0.2,0],
		              [0.4,-0.4,-0.3],
		              [0,0.2,-0.3]])*t_delta+np.eye(n_region,n_region,0)
		self.Wxxu=np.array([[0.5,0,0.25],
		              [0,0,0],
		              [0,0,0.3]])*t_delta
		self.Wxu=np.array([0.2,0,0])*t_delta


		# Hemodynamic parameters
		self.alpha=0.32
		self.E0=0.34
		self.k=0.65
		self.gamma=0.41
		self.tao=0.98
		self.epsilon=0.4
		self.V0=1.
		self.TE=0.03
		self.r0=25
		self.theta0=40.3

		self.Whh=np.zeros((4,7))
		self.Whh[0,0]=-(t_delta*self.k-1)
		self.Whh[0,1]=-t_delta*self.gamma
		self.Whh[1,0]=t_delta
		self.Whh[1,1]=1
		self.Whh[2,1]=t_delta/self.tao
		self.Whh[2,2]=1
		self.Whh[2,4]=-t_delta/self.tao
		self.Whh[3,3]=1
		self.Whh[3,5]=-t_delta/self.tao
		self.Whh[3,6]=t_delta/self.tao

		self.Whx=np.array([t_delta,0,0,0])

		self.Wo=np.array([-(1-self.epsilon)*self.V0, -4.3*self.theta0*self.E0*self.V0*self.TE, \
			-self.epsilon*self.r0*self.E0*self.V0*self.TE])

		self.bh=np.array([t_delta*self.gamma,0,0,0])
		self.bo=self.V0*(4.3*self.theta0*self.E0*self.TE+self.epsilon*self.r0*self.E0*self.TE+(1-self.epsilon))


		# state variables
		self.x_state_initial=np.zeros((n_region))
		self.h_state_initial=np.ones((n_region,4))
		self.h_state_initial[:,0]=0
		self.f_output_initial=np.zeros((n_region))

		self.u=np.zeros((n_stimuli,n_time_point))
		self.x_state=np.zeros((n_region,1,n_time_point))
		self.h_state=np.zeros((n_region,4,n_time_point))
		self.f_output=np.zeros((n_region,1,n_time_point))	


