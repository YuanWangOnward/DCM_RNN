# Introduction
<div style="text-align: justify; text-indent: 30px;">
<p>Neural Network (NN) is a powerful modelling tool and one can find its exciting successes in various fields recently. In neurology, it can be used to detecting lesions, predicting treatment outcomes, and assisting diagnosis. However, its ability of discovering the working mechanism of the brain remains limited, because almost all the NN lack clear biophysical interpretability, although NN is originally inspired by biological studies. In this project, we make our efforts to address the problem. Instead of other related works which build NNs with standard NN models and then try to load the NNs with biophysical meaning, we propose and customize new Generalized Recurrent Neural Network (GRNN) deeply from an advanced biophysical model, Dynamic Causal Modeling (DCM). The resulting DCM_RNN links the power/flexibility of NN and the biophysical interpretability of DCM.
</p>
</div>

# Dynamic Causal Modelling
<center>
<img width="172" height="295"  src="assets/images/diagram_of_DCM.png?raw=true" >
</center>
<div style="text-align: justify; text-align:center;" >
<p>Fig. 1 High level overview of Dynamic Causal Modelling.</p>
</div>

<div style="text-align: justify; text-indent: 30px; ">
<p>
Dynamic Causal Modelling (DCM) [1] is a highly nonlinear generative
model used to infer the causal architecture of coupled dynamical
systems in the brain from biophysical measurements,
such as functional MRI (fMRI) data. The causal architecture indicates
how the neural activities interact with each other between distributed
brain regions and how input stimuli may alter the pattern. Fig. 1 shows
a high level overview of DCM. It is the only one  that
explicitly models the complete processes from inputs, neural
activities, hemodynamics, to MRI imaging, and thus considered to be the
most biologically plausible as well as the most technically advanced
fMRI modeling method[2][3].
</p>
</div>

<div style="text-align: justify; text-indent: 30px; ">
<p>Neural activity is an abstract description of the activity
(neuron firing)  in brain regions, one scalar per region. DCM assumes
the neural activities are couple between regions in a bilinear way:
</p>
</div>

<center>
<img width="295" height="106"  src="assets/images/x_evolving.png?raw=true">
</center>
<div style="text-align: justify;">
<p>
where x is the neural activity. Dot above a variable means its temporal
derivative.  u is the experimental or exogenous stimulation,
taken as the input of DCM.  m indicates the number of experimental
stimuli. Multiple stimuli, such as visual, acoustic, and tactile,
may present in a single experiment.  Subscript j means the jth entry
in a vector or the th matrix of in a matrix set. A, B, C
are parameters to be estimated, loosely referred to as the effective
connectivity. Fig. 2 visualizes how the matrices encode the causal
architecture in the brain.
</p>
</div>


<center><img width="400" height="200" align="center" src="assets/images/causal_architecture.png?raw=true" alt="causal architecture in the brain">
</center>
<div style="text-aligh: justify; text-align:center;" >
<p>Fig. 2 Connectivity matrices and the causal architecture.
Ni's are brain regions. u is input stimulus,
</p>
</div>


<div style="text-align: justify; text-indent: 30px; ">
<p>
Neural activity consumes oxygen and causes changes in blood
supply and blood oxygen content. MRI scanner can detect these
changes and record them as fMRI images. These are described in
the hemodynamic module and MRI imaging module which are highly
nonlinear. One overview of DCM with details are shown in Fig. 3.
</p>
</div>

<center>
<img align="center" src="assets/images/whole_DCM.png?raw=true">
</center>
<p style="text-align:center" >
Fig. 3 Overview of DCM with details. Although it seems very complex,
one can read the flow by noticing u is experimental input, x is neural
activity, s is a transitional signal, f is blood flow, v is blood
Volume, q is the deoxyhemoglobin content, and y is fMRI signal.
</p>


# DCM_RNN
<div style="text-align: justify; text-indent: 30px; "><p>
At the first glance, it is hard to see any correspondence between DCM
and any generic neural network, such as CNN or LSTM.
And it is the case :) However, we propose a generalization of
vanilla RNN (GRNN) and made it to cast DCM as
a special configuration of GRNN.
</p></div>

## A generalization of vanilla RNN
<div style="text-align: justify; text-indent: 30px; "><p>
The vanilla RNN models the relationship between its input and its
output as
</p></div>

<center>
<img width="375" height="89"  src="assets/images/classic_rnn.png?raw=true">
</center>

<div style="text-align: justify;"><p>
where x is the input, h is the hidden state, and y is the output.
W and b are weighting matrix and bias, the tunable parameters.
Subscript t indicates time and superscript are used in content to
differentiate parameters. f are nonlinear functions.
</p></div>

<div style="text-align: justify; text-indent: 30px; "><p>
The vanilla RNN is too simple to accommodating the complexity of DCM.
We generalize it by adding more nonlinearity:
</p></div>

<center>
<img width="375" height="82" src="assets/images/GRNN.png?raw=true">
</center>

<div style="text-align: justify;  "><p>
The φ's are the extra nonlinear functions added which may be
parameterized by ξ's. It greatly extends the flexibility of RNN.
</p></div>

## Convert DCM to DCM_RNN
<div style="text-align: justify; text-indent: 30px;"><p>
The key idea of the conversion turns out to be very simple.
The only tricks are a simple approximation and equation terms
rearrangement. The approximation is
</p></div>

<center>
<img width="183" height="74" src="assets/images/approximation.png?raw=true">
</center>

<div style="text-align: justify; "><p>
where ∆t is the time interval between two adjacent fMRI samples.
This approximation is valid as long as ∆t is small.
Substitute it into the original equation. One obtains
</p></div>

<center>
<img width="754" height="132" src="assets/images/x_equation.png?raw=true">
</center>

<div style="text-align: justify; "><p>
It can be visualized as a piece of neural network
</p></div>

<center>
<img width="397" height="232" src="assets/images/x_nn_piece.png?raw=true">
</center>

<div style="text-align: justify; "><p>
Similar trick can be applied for other parts of DCM and one can obtain
DCM_RNN as shown in Fig. 4.
</p></div>

<center>
<img align="center" src="assets/images/DCM_RNN.png?raw=true">
</center>
<div style="text-align: justify; text-align:center;" >
<p>Fig. 4 Overview of DCM_RNN.</p>
</div>

# Advantages of DCM_RNN over DCM
It is important to see that DCM_RNN is much more than a simple reformatting of DCM. Its advantages over the classical DCM includes:

* It provides a much more flexible framework which makes very simple to add biophysically inspired plausible modification, as long as the added modification operators are partially differentiable. In DCM, it’s like to induce a major change of its estimation procedure or even prohibited.
* DCM_RNN can leverage backpropagation for training which is significantly different from any existing methods for DCM. One can pursue model simplicity and data fidelity simultaneously by specifying an appropriate loss function for network training, while it has to be done separately in traditional DCM.  We show in experiments that DCM_RNN can achieve more accurate and sparser estimation of the effective connectivity matrices ABC.
* DCM_RNN is biophysical meaningful and compatible with other neural networks, which means it can be embedded into other NNs for advanced bio-signal modelling.
* DCM_RNN circumvent the bilinear approximation used in SPM, the main implementation of DCM. We have been able to show our implementation error is controllable at the cost of computation while the bilinear approximation of SPM causes significant loss of nonlinearity in the DCM theory.


# Publications

* Yuan Wang, Yao Wang, and Yvonne W Lui,
[Generalized Recurrent Neural Network accommodating Dynamic Causal Modelling for functional MRI analysis](http://dev.ismrm.org/2017/0953.html)
, ISMRM, 2017
* Yuan Wang, Yao Wang, and Yvonne W Lui,
Generalized Recurrent Neural Network accommodating Dynamic Causal Modelling for functional MRI analysis
, submitting to NeuroImage, under in-group review

#### <a href="https://yuanwangonward.github.io/">Back to Yuan's homepage</a>

