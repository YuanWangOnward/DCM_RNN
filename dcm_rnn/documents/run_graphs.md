Run forward Initializer Graph
---
- Assign x state value 
sess.run(dr.assign_x_state_stacked, feed_dict={dr.x_state_stacked_placeholder: data['x_hat'][idx]})
- Assign h initial value
