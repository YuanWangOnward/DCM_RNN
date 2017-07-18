def train(self, dr, data_package):
    """"""
    print('Training starts!')
    dp = data_package
    data = dp.data
    data['loss_x'] = []
    data['loss_y'] = []
    data['loss_smooth'] = []
    data['loss_total'] = []
    x_hat_previous = data['x_hat_merged'].data.copy()  # for stop criterion checking
    isess = tf.Session()  # used for calculate log SPM_data
    data['total_iteration_count'] = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        self.calculate_log_data(dr, data_package, isess)
        # self.add_image_log(data_package)
        # self.add_data_log(data_package)

        for epoch in range(dp.MAX_EPOCHS):
            h_initial_segment = data['H_STATE_INITIAL']
            sess.run(tf.assign(dr.x_state_stacked_previous, data['x_hat_merged'].get(0)))
            for i_segment in range(dp.N_SEGMENTS):

                for epoch_inner in range(dp.MAX_EPOCHS_INNER):
                    # assign proper SPM_data
                    if dp.IF_NODE_MODE is True:
                        sess.run([tf.assign(dr.x_state_stacked,
                                            data['x_hat_merged'].get(i_segment).reshape(dr.n_recurrent_step, 1)),
                                  tf.assign(dr.h_state_initial, h_initial_segment)])
                    else:
                        sess.run([tf.assign(dr.x_state_stacked, data['x_hat_merged'].get(i_segment)),
                                  tf.assign(dr.h_state_initial, h_initial_segment)])

                    # training
                    sess.run(dr.train, feed_dict={dr.y_true: data['y_train'][i_segment]})

                    # collect results
                    data['x_hat_merged'].set(i_segment, sess.run(dr.x_state_stacked))

                    # add counting
                    data['total_iteration_count'] += 1

                    # Display logs per CHECK_STEPS step
                    if data['total_iteration_count'] % dp.CHECK_STEPS == 0:
                        self.calculate_log_data(dr, data_package, isess)
                        self.add_image_log(data_package)
                        self.add_data_log(data_package)

                        # saved summary = sess.run(dr.merged_summary)
                        # saved dr.summary_writer.add_summary(summary, count_total)

                        '''
                        print("Total iteration:", '%04d' % count_total, "loss_y=",
                              "{:.9f}".format(SPM_data['loss_y'][-1]))
                        print("Total iteration:", '%04d' % count_total, "loss_x=",
                              "{:.9f}".format(SPM_data['loss_x'][-1]))

                        if IF_IMAGE_LOG:
                            add_image_log(extra_prefix=LOG_EXTRA_PREFIX)

                        if IF_DATA_LOG:
                            add_data_log(extra_prefix=LOG_EXTRA_PREFIX)
                        '''
                        '''
                        # check stop criterion
                        relative_change = tb.rmse(x_hat_previous, SPM_data['x_hat_merged'].get())
                        if relative_change < dr.stop_threshold:
                            print('Relative change: ' + str(relative_change))
                            print('Stop criterion met, stop training')
                        else:
                            # x_hat_previous = copy.deepcopy(SPM_data['x_hat_merged'])
                            x_hat_previous = SPM_data['x_hat_merged'].get().copy()
                        '''

                # prepare for next segment
                # update hemodynamic state initial
                h_initial_segment = sess.run(dr.h_connector)
                # update previous neural state
                sess.run(tf.assign(dr.x_state_stacked_previous, data['x_hat_merged'].get(i_segment)))


    isess.close()

    print("Optimization Finished!")

    # SPM_data['y_hat_log'] = y_hat_log

    return data_package