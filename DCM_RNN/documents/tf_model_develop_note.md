

# Initializer Model
-
- It works on 

- Hemodynamic_layer has two possible initials.
One is the true initial value for the whole signal sequence, which should be trainable.
One is the initial value for one particular signal segment, which is preferred to stay un-changed.
It's switched at optimization ops. 


### Hyper parameters:
- Gradient mask: control gradients back-propagated to each variable
- Sparse mask: if True, a corresponding variable has sparsity cost
- Prior mask: if True, a corresponding variable has prior distribution based cost
- 

### Variable:
- Connection variables
- Hemodynamic variables


### PlaceHolders:
- Training data
- Variable masks
- Learning rate
- 

