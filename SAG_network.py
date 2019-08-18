from network import Network
from data_helper import *
import tensorflow as tf


class SAGNet(Network):
    def __init__(self, data, data_helper, for_training):
        self.inputs = []
        self.data = data

        self.data_helper = data_helper
        self.for_training = for_training

        self.rnn_cell_depth = data['config_dict']['RNN_CELL_DEPTH']  # 2D
        self.rnn_state_dim = data['config_dict']['RNN_STATE_DIM']  # 512D

        self.leaky_value = data['config_dict']['LEAK_VALUE']
        self.max_part_size = data['config_dict']['MAX_PART_SIZE']
        self.embedding_size = data['config_dict']['EMBEDDING_VECOTR_SIZE']  # 320D
        self.bbox_size = data['config_dict']['BOUNDING_BOX_SIZE']  # 6D

        self.vert_rnn_max_time_step = self.max_part_size  # k
        self.edge_rnn_max_time_step = self.max_part_size * (self.max_part_size - 1) / 2  # K = (k - 1) x k / 2

        self.layers = {}

        if self.for_training:
            self.gpu_num = data['config_dict']['TRAIN']['NUM_GPUS']
            self.batch_size = data['config_dict']['TRAIN']['BATCH_SIZE']

            self.part_voxels = data['part_voxels']
            self.part_bboxs = data['part_bbox']
            self.part_visible_masks = data['part_visible_masks']
            self.gaussian_noise = data['gaussian_noise']

            # number of refine iterations
            self.n_iter = data['config_dict']['TRAIN']['EXCHANGE_NUM']

            # The same as variables mentioned above, two dimensional vector
            self.edge_pair_mask_inds = data['rel_pair_mask_inds']

            self.vert_lr = data['vert_lr']
            self.edge_lr = data['edge_lr']
            self.graph_gen_lr = data['graph_gen_lr']

            self.recon_gen_loss_ratio = data['recon_gen_loss_ratio']
            self.voxel_bbox_ratio = data['voxel_bbox_ratio']
            self.g_rec_kl_loss_ratio = data['g_rec_kl_loss_ratio']

            self.max_gradient_norm = data['max_gradient_norm']

            self.voxel_loss_weights = data['part_voxel_loss_weights']

            self.part_bbox_loss_masks = data['part_bbox_loss_masks']

            self.optimizer = data['config_dict']['TRAIN']['OPTIMIZER_TYPE']  # Optimizer type

            self.keep_prob = tf.placeholder(tf.float32)
        else:
            self.gpu_num = 1
            self.batch_size = data['config_dict']['TRAIN']['BATCH_SIZE']

            self.latent_z = data['latent_codes']
            self.part_visible_masks = data['part_visible_masks']

    ##############################################################################
    # Functions to setup network
    ##############################################################################
    def setup(self):
        if self.for_training:
            self.setup_for_training()
        else:
            self.setup_for_testing()

    def setup_for_training(self):
        self._setup_rnns_for_training()
        self._setup_optimizer()

        global voxel_output, bbox_output, g_loss
        global graph_gen_grads_and_vars, g_part_mse_loss

        vert_grad_var_list = []
        edge_grad_var_list = []
        graph_gen_grad_var_list = []

        vert_loss_list = []
        edge_loss_list = []

        graph_rec_loss_list = []
        graph_kl_loss_list = []

        total_loss_list = []

        for gpu_id in range(int(self.gpu_num)):
            cur_dev_str = '/gpu:%d' % gpu_id

            self.vert_multi_cell_state = self.vert_multi_cell.zero_state(self.batch_size, tf.float32)
            self.edge_multi_cell_state = self.edge_multi_cell.zero_state(self.batch_size, tf.float32)

            with tf.device(cur_dev_str):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                    voxels_vector = self._voxel_encoder(self.part_voxels[gpu_id])
                    bboxs_vector = self._bbox_encoder(self.part_bboxs[gpu_id],
                                                      edge_pair_mask=self.edge_pair_mask_inds[gpu_id])

                    vert_factor, edge_factor = self._iterate(voxels_vector, bboxs_vector,
                                                             edge_pair_mask=self.edge_pair_mask_inds[gpu_id])

                    vert_out, edge_out, mu, log_sigma, g_part_mse_loss = self._learn_representation_for_graph(
                        vert_factor, edge_factor, part_visible_masks=self.part_visible_masks[gpu_id],
                        gaussian_noise=self.gaussian_noise[gpu_id], reuse=(gpu_id > 0))

                    voxel_output, bbox_output = self._pred_output(vert_out, edge_out)

                    # KL loss
                    g_kl_loss = self._final_graph_kl_loss(mu=mu, log_sigma=log_sigma)
                    g_kl_loss = tf.expand_dims(g_kl_loss, 0)
                    graph_kl_loss_list.append(g_kl_loss)

                    # The reconstruction loss for the graph representation
                    g_mse_loss = self._final_graph_reconstruction_loss(g_vert_in=vert_factor,
                                                                       g_edge_in=edge_factor,
                                                                       g_vert_out=vert_out,
                                                                       g_edge_out=edge_out)
                    g_mse_loss = g_mse_loss + g_part_mse_loss
                    g_mse_loss = tf.expand_dims(g_mse_loss, 0)
                    graph_rec_loss_list.append(g_mse_loss)

                    # Total loss for graph representation
                    g_loss = [g_mse_loss * self.g_rec_kl_loss_ratio,
                              g_kl_loss * (1 - self.g_rec_kl_loss_ratio)]
                    g_loss = tf.add_n(g_loss, name='graph_loss')

                    # Reconstruction loss for the voxel maps and bounding boxes
                    cur_voxel_loss = self._final_voxels_loss(self.part_voxels[gpu_id], voxel_output,
                                                             voxel_loss_weight=self.voxel_loss_weights,
                                                             voxel_loss_mask=self.part_visible_masks[gpu_id])

                    cur_bbox_loss = self._final_bboxs_loss(self.bbox_input, bbox_output,
                                                               bbox_loss_mask=self.part_bbox_loss_masks[gpu_id])

                    cur_voxel_loss = tf.expand_dims(cur_voxel_loss, 0)
                    cur_bbox_loss = tf.expand_dims(cur_bbox_loss, 0)

                    vert_loss_list.append(cur_voxel_loss)
                    edge_loss_list.append(cur_bbox_loss)

                    rec_losses = [cur_voxel_loss * self.voxel_bbox_ratio, cur_bbox_loss * (1 - self.voxel_bbox_ratio)]
                    rec_losses = tf.add_n(rec_losses, name='reconstruction_loss')

                    # Total losses for the whole framework
                    cur_total_loss = rec_losses * self.recon_gen_loss_ratio + \
                                     g_loss * (1 - self.recon_gen_loss_ratio)

                    # Compute the gradients for the three modules in our framework
                    vert_var_list, edge_var_list, graph_gen_list = self.merge_variable_list()

                    grads = tf.gradients(cur_total_loss, vert_var_list + edge_var_list + graph_gen_list)
                    grads, _ = tf.clip_by_global_norm(grads, self.max_gradient_norm)
                    vert_grad = grads[:len(vert_var_list)]
                    edge_grad = grads[len(vert_var_list): len(vert_var_list) + len(edge_var_list)]
                    graph_gen_grad = grads[len(vert_var_list) + len(edge_var_list):]

                    graph_gen_grad_var_list.append((graph_gen_grad, graph_gen_list))
                    vert_grad_var_list.append((vert_grad, vert_var_list))
                    edge_grad_var_list.append((edge_grad, edge_var_list))

                    total_loss_list.append(cur_total_loss)

        vert_loss = tf.concat(values=vert_loss_list, axis=0)
        self.voxel_loss = tf.reduce_mean(vert_loss, name='voxel_loss')
        edge_loss = tf.concat(values=edge_loss_list, axis=0)
        self.bbox_loss = tf.reduce_mean(edge_loss, name='bbox_loss')

        graph_rec_loss = tf.concat(values=graph_rec_loss_list, axis=0)
        self.g_mse_loss = tf.reduce_mean(graph_rec_loss, name='graph_mse_loss')
        graph_kl_loss = tf.concat(values=graph_kl_loss_list, axis=0)
        self.g_kl_loss = tf.reduce_mean(graph_kl_loss, name='graph_kl_loss')

        # Using three different optimizers to process three modules in our framework
        graph_gen_grads_and_vars = self._average_gradients(graph_gen_grad_var_list)

        tf.contrib.training.add_gradients_summaries(graph_gen_grads_and_vars)

        tf.summary.scalar('graph_kl_loss', self.g_kl_loss)
        tf.summary.scalar('graph_rec_loss', self.g_mse_loss)

        total_loss = tf.concat(values=total_loss_list, axis=0)
        self.total_losses = tf.reduce_mean(total_loss, name='total_loss')

        # Average the gradients for the geometry and structure information
        vert_grads_and_vars = self._average_gradients(vert_grad_var_list)
        edge_grads_and_vars = self._average_gradients(edge_grad_var_list)

        tf.summary.scalar('bbox_loss_', self.bbox_loss)
        tf.summary.scalar('voxel_loss_', self.voxel_loss)

        tf.contrib.training.add_gradients_summaries(vert_grads_and_vars)
        tf.contrib.training.add_gradients_summaries(edge_grads_and_vars)

        # Apply the gradients for the three optimizers
        vert_op = self.vert_opt.apply_gradients(vert_grads_and_vars)
        edge_op = self.edge_opt.apply_gradients(edge_grads_and_vars)
        graph_gen_op = self.graph_gen_opt.apply_gradients(graph_gen_grads_and_vars)

        self.train_op = tf.group(vert_op, edge_op, graph_gen_op, name='train_op')

        self.summary_op = tf.summary.merge_all()

    def setup_for_testing(self):
        self._setup_rnns_for_testing()

        cur_dev_str = '/gpu:0'
        with tf.device(cur_dev_str):
            with tf.variable_scope(tf.get_variable_scope()):
                layer_name = 'graph_embedding_layer'
                with tf.variable_scope(layer_name) as scope:
                    p_masks = tf.cast(self.part_visible_masks[self.gpu_num - 1], tf.float32)
                    obj_model_embedding = tf.concat(values=[self.latent_z, p_masks], axis=1)

                    part_representations = self._obj_gen_decoder_rnn_forward(obj_model_embedding)

                    vert_out = self._vert_gen_decoder_rnn_forward(part_representations[0])
                    edge_out = self._edge_gen_decoder_rnn_forward(part_representations[1])

                _ = self._pred_output(vert_out, edge_out)

    def merge_variable_list(self):
        """Merge all the trainable variables into different lists"""
        vert_list = []
        edge_list = []
        graph_gen_list = []

        for cur_var in tf.trainable_variables():
            var_name = cur_var.name
            if var_name.find('Graph') != -1 or var_name.find('graph') != -1:
                graph_gen_list.append(cur_var)
            elif var_name.find('voxel') != -1 or var_name.find('vert') != -1 or var_name.find('Voxel') != -1:
                vert_list.append(cur_var)
            elif var_name.find('bbox') != -1 or var_name.find('edge') != -1 or var_name.find('BBox') != -1:
                edge_list.append(cur_var)
        return vert_list, edge_list, graph_gen_list

    def _setup_optimizer(self):
        """Setup the optimizers for different modules in our framework. The vert_opt is
            for the geometry information, and the edge_opt is for the structure information,
            and the graph_gen_opt is for the 2-way VAE."""
        momentum_value = self.data['config_dict']['TRAIN']['MOMENTUM_VALUE']

        if self.optimizer.lower() == 'adadelta':
            self.vert_opt = tf.train.AdadeltaOptimizer(learning_rate=self.vert_lr)
            self.edge_opt = tf.train.AdadeltaOptimizer(learning_rate=self.edge_lr)
            self.graph_gen_opt = tf.train.AdadeltaOptimizer(learning_rate=self.graph_gen_lr)
        elif self.optimizer.lower() == 'adam':
            self.vert_opt = tf.train.AdamOptimizer(learning_rate=self.vert_lr)
            self.edge_opt = tf.train.AdamOptimizer(learning_rate=self.edge_lr)
            self.graph_gen_opt = tf.train.AdamOptimizer(learning_rate=self.graph_gen_lr)
        elif self.optimizer.lower() == 'rmsprop':
            self.vert_opt = tf.train.RMSPropOptimizer(learning_rate=self.vert_lr)
            self.edge_opt = tf.train.RMSPropOptimizer(learning_rate=self.edge_lr)
            self.graph_gen_opt = tf.train.RMSPropOptimizer(learning_rate=self.graph_gen_lr)
        elif self.optimizer.lower() == 'momentum':
            self.vert_opt = tf.train.MomentumOptimizer(learning_rate=self.vert_lr, momentum=momentum_value, use_nesterov=False)
            self.edge_opt = tf.train.MomentumOptimizer(learning_rate=self.edge_lr, momentum=momentum_value, use_nesterov=False)
            self.graph_gen_opt = tf.train.MomentumOptimizer(learning_rate=self.graph_gen_lr, momentum=momentum_value, use_nesterov=False)
        elif self.optimizer.lower() == 'nesterov':
            self.vert_opt = tf.train.MomentumOptimizer(learning_rate=self.vert_lr, momentum=momentum_value, use_nesterov=True)
            self.edge_opt = tf.train.MomentumOptimizer(learning_rate=self.edge_lr, momentum=momentum_value, use_nesterov=True)
            self.graph_gen_opt = tf.train.MomentumOptimizer(learning_rate=self.graph_gen_lr, momentum=momentum_value, use_nesterov=True)
        elif self.optimizer.lower() == 'adagrad':
            self.vert_opt = tf.train.AdagradOptimizer(learning_rate=self.vert_lr)
            self.edge_opt = tf.train.AdagradOptimizer(learning_rate=self.edge_lr)
            self.graph_gen_opt = tf.train.AdagradOptimizer(learning_rate=self.graph_gen_lr)
        elif self.optimizer.lower() == 'adagradda':
            self.vert_opt = tf.train.AdagradDAOptimizer(learning_rate=self.vert_lr)
            self.edge_opt = tf.train.AdagradDAOptimizer(learning_rate=self.edge_lr)
            self.graph_gen_opt = tf.train.AdagradDAOptimizer(learning_rate=self.graph_gen_lr)
        else:
            self.vert_opt = tf.train.GradientDescentOptimizer(learning_rate=self.vert_lr)
            self.edge_opt = tf.train.GradientDescentOptimizer(learning_rate=self.edge_lr)
            self.graph_gen_opt = tf.train.GradientDescentOptimizer(learning_rate=self.graph_gen_lr)

        tf.summary.scalar('vert_learning_rate_', self.vert_lr)
        tf.summary.scalar('edge_learning_rate_', self.edge_lr)
        tf.summary.scalar('graph_gen_learning_rate', self.graph_gen_lr)

    def _setup_rnns_for_testing(self):
        #   build rnn for decode the whole object representation
        obj_decode_gru_cell = tf.contrib.rnn.GRUCell(self.rnn_state_dim, activation=tf.tanh)
        self.obj_decode_multi_cell = tf.contrib.rnn.MultiRNNCell([obj_decode_gru_cell] * self.rnn_cell_depth)
        self.obj_decode_multi_cell_state = self.obj_decode_multi_cell.zero_state(self.batch_size, tf.float32)

        #   build the rnn for decode the vert and edge information
        vert_decode_gru_cell = tf.contrib.rnn.GRUCell(self.rnn_state_dim, activation=tf.tanh)
        edge_decode_gru_cell = tf.contrib.rnn.GRUCell(self.rnn_state_dim, activation=tf.tanh)

        self.vert_decode_multi_cell = tf.contrib.rnn.MultiRNNCell([vert_decode_gru_cell] * self.rnn_cell_depth)
        self.edge_decode_multi_cell = tf.contrib.rnn.MultiRNNCell([edge_decode_gru_cell] * self.rnn_cell_depth)

        self.vert_decode_multi_cell_state = self.vert_decode_multi_cell.zero_state(self.batch_size, tf.float32)
        self.edge_decode_multi_cell_state = self.edge_decode_multi_cell.zero_state(self.batch_size, tf.float32)

    def _setup_rnns_for_training(self):
        """Construct RNN cells and states. And build and initialize RNNs for message passing"""
        vert_gru_cell = tf.contrib.rnn.GRUCell(self.rnn_state_dim, activation=tf.tanh)
        edge_gru_cell = tf.contrib.rnn.GRUCell(self.rnn_state_dim, activation=tf.tanh)

        self.vert_multi_cell = tf.contrib.rnn.MultiRNNCell([vert_gru_cell] * self.rnn_cell_depth)
        self.edge_multi_cell = tf.contrib.rnn.MultiRNNCell([edge_gru_cell] * self.rnn_cell_depth)

        self.vert_multi_cell_state = self.vert_multi_cell.zero_state(self.batch_size, tf.float32)
        self.edge_multi_cell_state = self.edge_multi_cell.zero_state(self.batch_size, tf.float32)

        #  build the rnns for encode the vert and edge information
        vert_encode_gru_cell = tf.contrib.rnn.GRUCell(self.rnn_state_dim, activation=tf.tanh)
        edge_encode_gru_cell = tf.contrib.rnn.GRUCell(self.rnn_state_dim, activation=tf.tanh)

        self.vert_encode_multi_cell = tf.contrib.rnn.MultiRNNCell([vert_encode_gru_cell] * self.rnn_cell_depth)
        self.edge_encode_multi_cell = tf.contrib.rnn.MultiRNNCell([edge_encode_gru_cell] * self.rnn_cell_depth)

        self.vert_encode_multi_cell_state = self.vert_encode_multi_cell.zero_state(self.batch_size, tf.float32)
        self.edge_encode_multi_cell_state = self.edge_encode_multi_cell.zero_state(self.batch_size, tf.float32)

        #   build the rnn for decode the vert and edge information
        vert_decode_gru_cell = tf.contrib.rnn.GRUCell(self.rnn_state_dim, activation=tf.tanh)
        edge_decode_gru_cell = tf.contrib.rnn.GRUCell(self.rnn_state_dim, activation=tf.tanh)

        self.vert_decode_multi_cell = tf.contrib.rnn.MultiRNNCell([vert_decode_gru_cell] * self.rnn_cell_depth)
        self.edge_decode_multi_cell = tf.contrib.rnn.MultiRNNCell([edge_decode_gru_cell] * self.rnn_cell_depth)

        self.vert_decode_multi_cell_state = self.vert_decode_multi_cell.zero_state(self.batch_size, tf.float32)
        self.edge_decode_multi_cell_state = self.edge_decode_multi_cell.zero_state(self.batch_size, tf.float32)

        #   build rnn for encode/decode the whole object representation
        obj_encode_gru_cell = tf.contrib.rnn.GRUCell(self.rnn_state_dim, activation=tf.tanh)
        obj_decode_gru_cell = tf.contrib.rnn.GRUCell(self.rnn_state_dim, activation=tf.tanh)

        self.obj_encode_multi_cell = tf.contrib.rnn.MultiRNNCell([obj_encode_gru_cell] * self.rnn_cell_depth)
        self.obj_decode_multi_cell = tf.contrib.rnn.MultiRNNCell([obj_decode_gru_cell] * self.rnn_cell_depth)

        self.obj_encode_multi_cell_state = self.obj_encode_multi_cell.zero_state(self.batch_size, tf.float32)
        self.obj_decode_multi_cell_state = self.obj_decode_multi_cell.zero_state(self.batch_size, tf.float32)

    def _average_gradients(self, tower_grads):
        """Average the gradients for multi-GPU training"""
        total_grads = []
        grad_len = len(tower_grads[0][0])
        for i in range(grad_len):
            total_grads.append([])
        for t_ind in range(len(total_grads)):
            for g_ind in range(len(tower_grads)):
                total_grads[t_ind].append(tower_grads[g_ind][0][t_ind])

        grad_and_var = []
        for (grads, vars) in zip(total_grads, tower_grads[0][1]):
            has_none = False
            for grad in grads:
                if grad is None:
                    has_none = True
            if has_none:
                continue
            cur_grad = grads
            cur_grad = tf.reduce_mean(cur_grad, axis=0)

            grad_and_var.append((cur_grad, vars))
        return grad_and_var

    ##############################################################################
    # Functions in the encoder module
    ##############################################################################
    def _iterate(self, voxel_vector, bbox_vector, edge_pair_mask=None):
        """
        iterate for passing global relationship between parts and merge the geometry & structure features iteratively.
        """
        with tf.variable_scope('voxel_unary') as scope:
            (self.feed(voxel_vector)
                 .fc(self.rnn_state_dim, leaky_value=self.leaky_value, relu=False, name='vert_unary_fc')
                 .batch_norm(name='vert_unary', relu=False))
        with tf.variable_scope('bbox_unary') as scope:
            (self.feed(bbox_vector)
                 .fc(self.rnn_state_dim, leaky_value=self.leaky_value, relu=False, name='edge_unary_fc')
                 .batch_norm(name='edge_unary', relu=False))

        vert_unary = self.get_output('vert_unary')
        edge_unary = self.get_output('edge_unary')

        global vert_factor, edge_factor
        # we obtain the new states of the gru cells
        vert_factor = self._vert_rnn_forward(vert_unary, reuse=False)
        edge_factor = self._edge_rnn_forward(edge_unary, reuse=False)

        for i in xrange(self.n_iter):
            reuse = i > 0
            # compute vert states
            vert_ctx = self._compute_vert_context(edge_factor, vert_factor, reuse=reuse, edge_pair_mask=edge_pair_mask)
            vert_ctx = tf.reshape(vert_ctx, [-1, self.vert_rnn_max_time_step, self.rnn_state_dim])

            vert_factor = self._vert_rnn_forward(vert_ctx, reuse=True)

            # compute edge states
            edge_ctx = self._compute_edge_context(vert_factor, edge_factor, reuse=reuse, edge_pair_mask=edge_pair_mask)
            edge_ctx = tf.reshape(edge_ctx, [-1, self.edge_rnn_max_time_step, self.rnn_state_dim])

            edge_factor = self._edge_rnn_forward(edge_ctx, reuse=True)

        # These two features are used to compute the reconstruction loss
        self.vert_gen_encoder_input = vert_factor
        self.edge_gen_encoder_input = edge_factor

        return vert_factor, edge_factor

    def _voxel_encoder(self, input_voxels, reuse=False):
        """Encoder for voxel maps"""
        layer_name = 'voxel_encoder'

        voxel_size = self.data['config_dict']['CUBE_LEN']

        with tf.variable_scope(layer_name) as scope:
            if reuse: tf.get_variable_scope().reuse_variables()

            input_voxels = tf.reshape(input_voxels, (-1, voxel_size, voxel_size, voxel_size, 1))

            init_weights = tf.contrib.layers.xavier_initializer()
            init_biases = tf.zeros_initializer()

            valid_strides = [1, 2, 2, 2, 1]
            same_strides = [1, 1, 1, 1, 1]

            ve_conv_1_out = self.conv3d(input_voxels, 3, 64, 've_conv_1_1', init_weights, strides=valid_strides,init_biases=init_biases,
                                        leaky_value=self.leaky_value, relu=True,  batch_norm=True, padding='SAME')

            # first resnet blocks
            ve_conv_block_2_1_out = self.conv_residual_block(ve_conv_1_out, 3, 64, init_weights,
                                                             've_conv_block_2_1', leaky_value=self.leaky_value, padding='SAME', bottle_neck=True)
            ve_conv_block_2_2_out = self.conv_residual_block(ve_conv_block_2_1_out, 3, 64, init_weights,
                                                             've_conv_block_2_2', leaky_value=self.leaky_value, padding='SAME', bottle_neck=False)

            # second resnet blocks
            ve_conv_block_3_1_out = self.conv_residual_block(ve_conv_block_2_2_out, 3, 128, init_weights,
                                                             've_conv_block_3_1', self.leaky_value, padding='SAME', bottle_neck=True)
            ve_conv_block_3_2_out = self.conv_residual_block(ve_conv_block_3_1_out, 3, 128, init_weights,
                                                             've_conv_block_3_2', self.leaky_value, padding='SAME', bottle_neck=False)

            # third resnet blocks
            ve_conv_block_4_1_out = self.conv_residual_block(ve_conv_block_3_2_out, 3, 256, init_weights,
                                                             've_conv_block_4_1', self.leaky_value, padding='SAME', bottle_neck=True)
            ve_conv_block_4_2_out = self.conv_residual_block(ve_conv_block_4_1_out, 3, 256, init_weights,
                                                             've_conv_block_4_2', self.leaky_value, padding='SAME', bottle_neck=False)
            ve_conv_block_4_3_out = self.conv_residual_block(ve_conv_block_4_2_out, 3, 256, init_weights,
                                                             've_conv_block_4_3', self.leaky_value, padding='SAME', bottle_neck=False)

            ve_conv_out = self.conv3d(ve_conv_block_4_3_out, 2, 512, 've_conv_out', init_weights, strides=same_strides, init_biases=init_biases,
                                      leaky_value=self.leaky_value, relu=True, batch_norm=True, padding='SAME')
            return ve_conv_out

    def _voxel_decoder(self, input_voxel_features, reuse=False):
        """decode and predict the value of each voxel"""
        layer_name = 'voxel_decoder'

        with tf.variable_scope(layer_name) as scope:
            if reuse: tf.get_variable_scope().reuse_variables()

            init_weights = tf.contrib.layers.xavier_initializer()
            init_biases = tf.zeros_initializer()

            valid_strides = [1, 2, 2, 2, 1]
            same_strides = [1, 1, 1, 1, 1]
            batch_size = tf.shape(input_voxel_features)[0]
            z = tf.reshape(input_voxel_features, (-1, 1, 1, 1, self.embedding_size))

            vd_deconv_1_1_out = self.deconv3d(z, 2, 512, 'vd_deconv_1_1', (batch_size, 2, 2, 2, 512), init_weights,
                                              same_strides, init_biases, leaky_value=self.leaky_value, relu=True,
                                              batch_norm=True, padding='VALID')

            # first deconv block
            vd_deconv_2_1_out = self.deconv_residual_block(vd_deconv_1_1_out, 3, 256, (batch_size, 4, 4, 4, 256),
                                                           init_weights, 'vd_deconv_2_1_block', self.leaky_value,
                                                           padding='SAME', bottle_neck=True)
            vd_deconv_2_2_out = self.deconv_residual_block(vd_deconv_2_1_out, 3, 256, (batch_size, 4, 4, 4, 256),
                                                           init_weights, 'vd_deconv_2_2_block', self.leaky_value,
                                                           padding='SAME', bottle_neck=False)
            vd_deconv_2_3_out = self.deconv_residual_block(vd_deconv_2_2_out, 3, 256, (batch_size, 4, 4, 4, 256),
                                                           init_weights, 'vd_deconv_2_3_block', self.leaky_value,
                                                           padding='SAME', bottle_neck=False)

            # second deconv block
            vd_deconv_3_1_out = self.deconv_residual_block(vd_deconv_2_3_out, 3, 128, (batch_size, 8, 8, 8, 128),
                                                           init_weights, 'vd_deconv_3_1_block', self.leaky_value,
                                                           padding='SAME', bottle_neck=True)
            vd_deconv_3_2_out = self.deconv_residual_block(vd_deconv_3_1_out, 3, 128, (batch_size, 8, 8, 8, 128),
                                                           init_weights, 'vd_deconv_3_2_block', self.leaky_value,
                                                           padding='SAME', bottle_neck=False)

            # third deconv block
            vd_deconv_4_1_out = self.deconv_residual_block(vd_deconv_3_2_out, 3, 64, (batch_size, 16, 16, 16, 64),
                                                           init_weights, 'vd_deconv_4_1_block', self.leaky_value,
                                                           padding='SAME', bottle_neck=True)
            vd_deconv_4_2_out = self.deconv_residual_block(vd_deconv_4_1_out, 3, 64, (batch_size, 16, 16, 16, 64),
                                                           init_weights, 'vd_deconv_4_2_block', self.leaky_value,
                                                           padding='SAME', bottle_neck=False)

            vd_deconv_out = self.deconv3d(vd_deconv_4_2_out, 3, 1, 'vd_deconv_5_1', (batch_size, 32, 32, 32, 1),
                                          init_weights, valid_strides, init_biases, leaky_value=self.leaky_value, relu=False,
                                          batch_norm=False, padding='SAME')

            deconv_feature = tf.nn.sigmoid(vd_deconv_out, name='voxels_out')
            self.wrap(deconv_feature, layer_name)

            return deconv_feature

    def _bbox_encoder(self, input_bb_feature, reuse=False, edge_pair_mask=None):
        """Encoder for bounding boxes"""
        layer_name = 'bbox_encoder'

        with tf.variable_scope(layer_name) as scope:
            bbox_tensor = tf.reshape(input_bb_feature, [-1, self.bbox_size])

            # we get the context relationship of pairwise vertex of the bounding box of a graph
            bbox_f_factor = tf.gather(bbox_tensor, edge_pair_mask[:, 0])
            bbox_s_factor = tf.gather(bbox_tensor, edge_pair_mask[:, 1])
            bbox_input = tf.concat(values=[bbox_f_factor, bbox_s_factor], axis=1, name="bbox_concat_input")
            self.bbox_input = bbox_input

            self.bbox_origin_input = bbox_tensor

            (self.feed(bbox_input)
                 .fc(self.embedding_size * 2, leaky_value=self.leaky_value, relu=False, name='bbox_encoder_layer_fc', reuse=reuse)
                 .batch_norm(name='bbox_encoder_layer_bn', relu=False)
                 .lrelu(leaky_value=self.leaky_value, name='bbox_encoder_layer_out'))
            bbox_encoder_feature = self.get_output('bbox_encoder_layer_out')

            return bbox_encoder_feature

    def _decode_bbox(self, input_bbox_feature, reuse=False):
        layer_name = 'bbox_decoder'
        pred_name = 'bbox_pred'

        with tf.variable_scope(layer_name) as scope:
            (self.feed(input_bbox_feature)
                 .fc(self.embedding_size * 2, leaky_value=self.leaky_value, relu=False, name='bbox_decoder_layer_1_fc', reuse=reuse)
                 .batch_norm(name='bbox_decoder_layer_1_bn', relu=False, reuse=reuse)
                 .lrelu(leaky_value=self.leaky_value, name='bbox_decoder_layer_1_out')
                 .fc(self.bbox_size * 2, leaky_value=self.leaky_value, relu=False, name='bbox_decoder_layer_2_fc', reuse=reuse)
                 .batch_norm(name=pred_name, relu=False, reuse=reuse))
            self.bbox_pred = self.get_output(pred_name)

            return self.bbox_pred

    def _decode_part_voxels(self, input_layer, reuse=False):
        layer_name = 'part_voxel_output'
        print(layer_name)

        # Transform the dimension through a fully-connected layer
        with tf.variable_scope('part_voxel_decoder') as scope:
            (self.feed(input_layer)
                 .fc(self.embedding_size, relu=False, leaky_value=self.leaky_value, name='part_voxel_decoder_fc', reuse=reuse)
                 .batch_norm(name='part_voxel_decoder_bn', relu=False, reuse=reuse)
                 .lrelu(leaky_value=self.leaky_value, name=layer_name))
        vert_feature = self.get_output(layer_name)

        self.voxel_pred = self._voxel_decoder(vert_feature, reuse=reuse)

        return self.voxel_pred

    def _pred_output(self, vert_factor, edge_factor, reuse=False):
        """Predict the outputs for geometry(voxel maps) and structure(bounding boxes)"""
        voxel_decodings = self._decode_part_voxels(vert_factor, reuse=reuse)
        bbox_decodings = self._decode_bbox(edge_factor, reuse=reuse)

        return voxel_decodings, bbox_decodings

    ##############################################################################
    # Functions to compute context and learn the latent representation through RNNs
    ##############################################################################
    def _compute_edge_context(self, vert_factor, edge_factor, reuse=False, edge_pair_mask=None):
        """
        attention-based edge message pooling
        """
        vert_factor = tf.reshape(vert_factor, [-1, self.rnn_state_dim])
        edge_factor = tf.reshape(edge_factor, [-1, self.rnn_state_dim])

        vert_in_factor = tf.gather(vert_factor, edge_pair_mask[:, 0])
        vert_out_factor = tf.gather(vert_factor, edge_pair_mask[:, 1])

        vert_w_input_first = tf.concat(values=[vert_in_factor, edge_factor], axis=1)
        vert_w_input_second = tf.concat(values=[vert_out_factor, edge_factor], axis=1)

        # compute compatibility scores
        (self.feed(vert_w_input_first)
             .fc(1, relu=False, leaky_value=self.leaky_value, reuse=reuse, name='vert_first_w_fc')
             .sigmoid(name='edge_vert_first_score'))
        (self.feed(vert_w_input_second)
             .fc(1, relu=False, leaky_value=self.leaky_value, reuse=True, name='vert_first_w_fc')
             .sigmoid(name='edge_vert_second_score'))

        vert_first_w = self.get_output('edge_vert_first_score')
        vert_second_w = self.get_output('edge_vert_second_score')

        weighted_first_vert = tf.multiply(vert_in_factor, vert_first_w)
        weighted_second_vert = tf.multiply(vert_out_factor, vert_second_w)

        return weighted_first_vert + weighted_second_vert

    def _compute_vert_context(self, edge_factor, vert_factor, reuse=False, edge_pair_mask=None):
        """
        attention-based vertex(node) message pooling
        """
        """the edge_pair_mask_inds[:, 0] store the index of in-bound vertex of an edge 
        and the edge_pair_mask_inds[:, 1] store the index of out-bound vertex of an edge"""
        edge_factor = tf.reshape(edge_factor, [-1, self.rnn_state_dim])
        vert_factor = tf.reshape(vert_factor, [-1, self.rnn_state_dim])

        vert_in_factor = tf.gather(vert_factor, edge_pair_mask[:, 0])
        vert_out_factor = tf.gather(vert_factor, edge_pair_mask[:, 1])

        # concat outgoing edges and ingoing edges with gathered vert_factors
        in_edge_w_input = tf.concat(values=[vert_in_factor, edge_factor], axis=1)
        out_edge_w_input = tf.concat(values=[vert_out_factor, edge_factor], axis=1)

        # compute compatibility scores
        (self.feed(out_edge_w_input)
             .fc(1, relu=False, leaky_value=self.leaky_value, reuse=reuse, name='edge_w_fc')
             .sigmoid(name='out_edge_score'))
        (self.feed(in_edge_w_input)
             .fc(1, relu=False, leaky_value=self.leaky_value, reuse=True, name='edge_w_fc')
             .sigmoid(name='in_edge_score'))

        out_edge_w = self.get_output('out_edge_score')
        in_edge_w = self.get_output('in_edge_score')

        # weigh the edge factors with computed weigths
        out_edge_weighted = tf.multiply(edge_factor, out_edge_w)
        in_edge_weighted = tf.multiply(edge_factor, in_edge_w)

        out_edge_weighted = tf.reshape(out_edge_weighted, [-1, self.edge_rnn_max_time_step, self.rnn_state_dim])
        in_edge_weighted = tf.reshape(in_edge_weighted, [-1, self.edge_rnn_max_time_step, self.rnn_state_dim])

        out_edge_weighted_list = tf.split(out_edge_weighted, num_or_size_splits=self.edge_rnn_max_time_step, axis=1)
        in_edge_weighted_list = tf.split(in_edge_weighted, num_or_size_splits=self.edge_rnn_max_time_step, axis=1)

        first_index_list = []
        second_index_list = []
        first_tens_list = []
        second_tens_list = []

        cur_index = 0
        for ind in range(self.max_part_size - 1):
            first_index_list.append(cur_index)
            first_tens_list.append(tf.identity(in_edge_weighted_list[cur_index]))
            cur_index = cur_index + self.max_part_size - 1 - ind
        for ind in range(self.max_part_size - 1):
            second_index_list.append(ind)
            second_tens_list.append(tf.identity(out_edge_weighted_list[ind]))

        cur_part_index = 0
        for f_offset in range(self.max_part_size):
            for s_offset in range(f_offset + 1, self.max_part_size):
                if not (cur_part_index in first_index_list):
                    first_tens_list[f_offset] = tf.add(first_tens_list[f_offset], in_edge_weighted_list[cur_part_index])
                if not (cur_part_index in second_index_list):
                    second_tens_list[s_offset - 1] = tf.add(second_tens_list[s_offset - 1], out_edge_weighted_list[cur_part_index])
                cur_part_index = cur_part_index + 1

        self.first_tens_list = first_tens_list
        self.second_tens_list = second_tens_list
        self.out_edge_weighted_list = out_edge_weighted_list
        self.in_edge_weighted_list = in_edge_weighted_list

        final_list = []
        for ind in range(self.max_part_size):
            if ind == 0:
                final_list.append(first_tens_list[ind])
            elif ind == self.max_part_size - 1:
                final_list.append(second_tens_list[ind - 1])
            else:
                final_list.append(first_tens_list[ind] + second_tens_list[ind - 1])
        vert_ctx = tf.concat(values=final_list, axis=0)
        vert_ctx = tf.reshape(vert_ctx, [-1, self.rnn_state_dim])
        return vert_ctx

    def _vert_rnn_forward(self, vert_in, reuse=False):
        with tf.variable_scope('vert_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()

            vert_in = tf.reshape(vert_in, [-1, self.vert_rnn_max_time_step, self.rnn_state_dim])

            (vert_out, self.vert_multi_cell_state) = \
                tf.nn.dynamic_rnn(self.vert_multi_cell, vert_in, initial_state=self.vert_multi_cell_state, time_major=False)
        return vert_out

    def _edge_rnn_forward(self, edge_in, reuse=False):
        with tf.variable_scope('edge_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()

            edge_in = tf.reshape(edge_in, [-1, self.edge_rnn_max_time_step, self.rnn_state_dim])

            (edge_out, self.edge_multi_cell_state) = \
                tf.nn.dynamic_rnn(self.edge_multi_cell, inputs=edge_in, initial_state=self.edge_multi_cell_state, time_major=False)
        return edge_out

    def _vert_gen_encoder_rnn_forward(self, vert_in, reuse=False):
        with tf.variable_scope('vert_gen_encoder_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()

            vert_in = tf.reshape(vert_in, [-1, self.vert_rnn_max_time_step, self.rnn_state_dim])

            vert_encoder_initial_state = self.vert_encode_multi_cell.zero_state(self.batch_size, tf.float32)
            (vert_out, self.vert_encode_multi_cell_state) = \
                tf.nn.dynamic_rnn(self.vert_encode_multi_cell, vert_in, initial_state=vert_encoder_initial_state, time_major=False)
            vert_state_out = self.vert_encode_multi_cell_state[-1]
        return vert_state_out

    def _edge_gen_encoder_rnn_forward(self, edge_in, reuse=False):
        with tf.variable_scope('edge_gen_encoder_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()

            edge_in = tf.reshape(edge_in, [-1, self.edge_rnn_max_time_step, self.rnn_state_dim])

            edge_encoder_initial_state = self.edge_encode_multi_cell.zero_state(self.batch_size, tf.float32)
            (edge_out, self.edge_encode_multi_cell_state) = \
                tf.nn.dynamic_rnn(self.edge_encode_multi_cell, edge_in, initial_state=edge_encoder_initial_state, time_major=False)
            edge_state_out = self.edge_encode_multi_cell_state[-1]
        return edge_state_out

    def _obj_gen_encoder_rnn_forward(self, obj_vert_in, obj_edge_in, reuse=False):
        with tf.variable_scope('obj_gen_encoder_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()

            obj_in = tf.concat(values=[obj_vert_in, obj_edge_in], axis=1)
            obj_in = tf.reshape(obj_in, [self.batch_size, -1, self.rnn_state_dim])

            obj_encoder_initial_state = self.obj_encode_multi_cell.zero_state(self.batch_size, tf.float32)
            (obj_out, self.obj_encode_multi_cell_state) = \
                    tf.nn.dynamic_rnn(self.obj_encode_multi_cell, obj_in, initial_state=obj_encoder_initial_state, time_major=False)
            obj_state_out = self.obj_encode_multi_cell_state[-1]
        return obj_state_out

    def _obj_gen_decoder_rnn_forward(self, latent_input, reuse=False):
        with tf.variable_scope('obj_gen_decoder_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()

            part_out_list = []

            # transform the dimension for initial state
            (self.feed(latent_input)
                 .fc(self.rnn_state_dim, relu=False, name='obj_initial_state_dense', reuse=reuse)
                 .batch_norm(name='obj_initial_state_bn', relu=False)
                 .lrelu(leaky_value=self.leaky_value, name='obj_initial_state_dense_out'))
            obj_decoder_initial_state = self.get_output('obj_initial_state_dense_out')

            obj_zero_input = tf.zeros_like(obj_decoder_initial_state)

            obj_initial_input = tf.concat(values=[obj_zero_input, obj_decoder_initial_state], axis=1)
            # transform the dimension for initial input
            (self.feed(obj_initial_input)
                 .fc(self.rnn_state_dim, relu=False, name='obj_decoder_input_embedding', reuse=reuse)
                 .batch_norm(name='obj_decoder_input_bn', relu=False)
                 .lrelu(leaky_value=self.leaky_value, name='obj_decoder_input_out'))
            obj_input = self.get_output('obj_decoder_input_out')

            obj_state = [obj_decoder_initial_state] * self.rnn_cell_depth
            (part_out, obj_state) = self.obj_decode_multi_cell(obj_input, obj_state)
            part_out_list.append(obj_state[-1])

            part_out = tf.concat(values=[part_out, obj_decoder_initial_state], axis=1)
            (self.feed(part_out)
                 .fc(self.rnn_state_dim, relu=False, name='obj_decoder_input_embedding', reuse=True)
                 .batch_norm(name='obj_decoder_input_bn', relu=False, reuse=True)
                 .lrelu(leaky_value=self.leaky_value, name='obj_decoder_input_out'))
            part_out = self.get_output('obj_decoder_input_out')

            (part_out, obj_state) = self.obj_decode_multi_cell(part_out, obj_state)
            part_out_list.append(obj_state[-1])

            return part_out_list

    def _vert_gen_decoder_rnn_forward(self, vert_initial_state, reuse=False):
        with tf.variable_scope('vert_gen_decoder_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()

            vert_out_list = []

            vert_zero_input = tf.zeros_like(vert_initial_state)
            vert_initial_input = tf.concat(values=[vert_zero_input, vert_initial_state], axis=1)

            (self.feed(vert_initial_input)
                 .fc(self.rnn_state_dim, relu=False, name='vert_input_embedding', reuse=reuse)
                 .batch_norm(name='vert_decoder_input_bn', relu=False)
                 .lrelu(leaky_value=self.leaky_value, name='vert_decoder_input_out'))
            vert_input = self.get_output('vert_decoder_input_out')

            vert_state = [vert_initial_state] * self.rnn_cell_depth
            (vert_out, vert_state) = self.vert_decode_multi_cell(vert_input, vert_state)
            vert_out_list.append(vert_out)

            for ind in range(1, self.max_part_size):
                if ind > 0:
                    should_reuse = True

                vert_out = tf.concat(values=[vert_out, vert_initial_state], axis=1)
                (self.feed(vert_out)
                     .fc(self.rnn_state_dim, relu=False, name='vert_input_embedding', reuse=should_reuse)
                     .batch_norm(name='vert_decoder_input_bn', relu=False, reuse=should_reuse)
                     .lrelu(leaky_value=self.leaky_value, name='vert_decoder_input_out'))
                vert_input = self.get_output('vert_decoder_input_out')
                (vert_out, vert_state) = self.vert_decode_multi_cell(vert_input, vert_state)
                vert_out_list.append(vert_out)

            vert_decoder_out = tf.stack(vert_out_list, axis=1)
            return vert_decoder_out

    def _edge_gen_decoder_rnn_forward(self, edge_initial_state, reuse=False):
        with tf.variable_scope('edge_gen_decoder_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()

            edge_out_list = []

            edge_zero_input = tf.zeros_like(edge_initial_state)
            edge_initial_input = tf.concat(values=[edge_zero_input, edge_initial_state], axis=1)

            (self.feed(edge_initial_input)
                 .fc(self.rnn_state_dim, relu=False, name='edge_input_embedding', reuse=reuse)
                 .batch_norm(name='edge_decoder_input_bn', relu=False)
                 .lrelu(leaky_value=self.leaky_value, name='edge_decoder_input_out'))
            edge_input = self.get_output('edge_decoder_input_out')

            edge_state = [edge_initial_state] * self.rnn_cell_depth
            (edge_out, edge_state) = self.edge_decode_multi_cell(edge_input, edge_state)
            edge_out_list.append(edge_out)

            for ind in range(1, self.edge_rnn_max_time_step):
                if ind > 0:
                    should_reuse = True

                edge_out = tf.concat(values=[edge_out, edge_initial_state], axis=1)
                (self.feed(edge_out)
                     .fc(self.rnn_state_dim, relu=False, name='edge_input_embedding', reuse=should_reuse)
                     .batch_norm(name='edge_decoder_input_bn', relu=False, reuse=should_reuse)
                    .lrelu(leaky_value=self.leaky_value, name='edge_decoder_input_out'))
                edge_input = self.get_output('edge_decoder_input_out')

                (edge_out, edge_state) = self.edge_decode_multi_cell(edge_input, edge_state)
                edge_out_list.append(edge_out)

            edge_decoder_out = tf.stack(edge_out_list, axis=1)
            return edge_decoder_out

    def _learn_representation_for_graph(self, input_vert_feature, input_edge_feature, part_visible_masks, gaussian_noise,
                                        layer_suffix='', phase_train=True, reuse=False):
        """Learn a latent space in the 2-way VAE"""
        layer_name = 'graph_embedding_layer_' + layer_suffix if layer_suffix != '' else 'graph_embedding_layer'

        with tf.variable_scope(layer_name) as scope:
            if reuse: tf.get_variable_scope().reuse_variables()

            p_masks = tf.cast(part_visible_masks, tf.float32)

            # Expand the part masks and then merge them with vert features
            mask_dims = tf.expand_dims(p_masks, axis=1)
            vert_expanded_mask = tf.tile(mask_dims, [1, self.max_part_size, 1])
            vert_expanded_mask = tf.reshape(vert_expanded_mask, [-1, self.max_part_size])

            input_vert_feature = tf.reshape(input_vert_feature, [-1, self.rnn_state_dim])
            input_edge_feature = tf.reshape(input_edge_feature, [-1, self.rnn_state_dim])

            vert_feature = tf.concat(values=[input_vert_feature, vert_expanded_mask], axis=1)
            (self.feed(vert_feature)
                 .fc(self.rnn_state_dim, relu=False, name='input_embedding_feature', reuse=reuse, trainable=phase_train)
                 .batch_norm(is_training=phase_train, name='vert_feature_batch_norm', relu=False)
                 .lrelu(leaky_value=self.leaky_value, name='vert_feature_out'))
            vert_feature_out = self.get_output('vert_feature_out')

            # Expand the part masks and then merge them with edge features
            edge_expanded_mask = tf.tile(mask_dims, [1, self.edge_rnn_max_time_step, 1])
            edge_expanded_mask = tf.reshape(edge_expanded_mask, [-1, self.max_part_size])

            edge_feature = tf.concat(values=[input_edge_feature, edge_expanded_mask], axis=1)
            (self.feed(edge_feature)
                 .fc(self.rnn_state_dim, relu=False, name='input_embedding_feature', reuse=True, trainable=phase_train)
                 .batch_norm(is_training=phase_train, name='edge_feature_batch_norm', relu=False)
                 .lrelu(leaky_value=self.leaky_value, name='edge_feature_out'))
            edge_feature_out = self.get_output('edge_feature_out')

            parts_vert_out = self._vert_gen_encoder_rnn_forward(vert_feature_out, reuse=reuse)
            parts_edge_out = self._edge_gen_encoder_rnn_forward(edge_feature_out, reuse=reuse)

            obj_embedding = self._obj_gen_encoder_rnn_forward(parts_vert_out, parts_edge_out, reuse=reuse)

            # obj_embedding = tf.concat(values=[obj_embedding, p_masks], axis=1)
            (self.feed(obj_embedding)
                 .fc(256, relu=False, name='graph_vector_fc', reuse=reuse, trainable=phase_train)
                 .batch_norm(is_training=phase_train, name='graph_vector_batch_norm', relu=False)
                 .lrelu(leaky_value=self.leaky_value, name='graph_vector_out'))
            graph_embedding_vector = self.get_output('graph_vector_out')

            (self.feed(graph_embedding_vector)
                 .fc(self.embedding_size, relu=False, name='graph_mu', reuse=reuse, trainable=phase_train)
                 .batch_norm(is_training=phase_train, name='graph_mu_out', relu=False))
            (self.feed(graph_embedding_vector)
                 .fc(self.embedding_size, relu=False, name='graph_sigma', reuse=reuse, trainable=phase_train)
                 .batch_norm(is_training=phase_train, name='graph_sigma_out', relu=False))

            mu = self.get_output('graph_mu_out')
            log_sigma = self.get_output('graph_sigma_out')

            # we make the sigma positive in this way by an exponential operation
            sigma = tf.exp(0.5 * log_sigma)
            self.latent_z = mu + tf.multiply(sigma, gaussian_noise)

            obj_model_embedding = tf.concat(values=[self.latent_z, p_masks], axis=1)

            part_representations = self._obj_gen_decoder_rnn_forward(obj_model_embedding, reuse=reuse)

            vert_decoder_out = self._vert_gen_decoder_rnn_forward(part_representations[0], reuse=reuse)
            edge_decoder_out = self._edge_gen_decoder_rnn_forward(part_representations[1], reuse=reuse)

            self.vert_decoder_out = vert_decoder_out
            self.edge_decoder_out = edge_decoder_out

            graph_part_loss = self._final_graph_reconstruction_loss(g_vert_in=parts_vert_out,
                                                                    g_edge_in=parts_edge_out,
                                                                    g_vert_out=part_representations[0],
                                                                    g_edge_out=part_representations[1])
            return vert_decoder_out, edge_decoder_out, mu, log_sigma, graph_part_loss

    ##############################################################################
    # Functions to compute losses
    ##############################################################################
    def _final_bboxs_loss(self, bbox_input, bbox_output, bbox_loss_mask=None):
        """calculate losses about bounding boxes"""
        bbox_in = tf.reshape(bbox_input, [-1])
        bbox_pred = tf.reshape(bbox_output, [-1])

        l1_loss = tf.abs(tf.subtract(bbox_pred, bbox_in))

        l1_loss = tf.reshape(l1_loss, [-1, 2 * self.bbox_size])
        l1_loss = tf.reduce_mean(l1_loss, axis=1)

        if bbox_loss_mask is not None:
            bbox_loss_mask = tf.reshape(bbox_loss_mask, [-1])
            l1_loss = tf.multiply(bbox_loss_mask, l1_loss)

        bbox_loss = tf.reduce_mean(l1_loss)

        return bbox_loss

    def _final_voxels_loss(self, voxel_input, voxel_output, voxel_loss_weight=None, voxel_loss_mask=None):
        """calculate the losses for the voxels of parts"""
        voxel_in = tf.reshape(voxel_input, [-1])
        voxel_pred = tf.reshape(voxel_output, [-1])

        cube_len = self.data['config_dict']['CUBE_LEN']
        mse_loss = tf.pow(voxel_in - voxel_pred, 2)
        mse_loss = tf.reshape(mse_loss, [-1, cube_len * cube_len * cube_len])
        mse_loss = tf.reduce_mean(mse_loss, axis=1)

        if voxel_loss_weight is not None:
            voxel_loss_weight = tf.reshape(voxel_loss_weight, [-1])
            mse_loss = tf.multiply(mse_loss, voxel_loss_weight)

        if voxel_loss_mask is not None:
            voxel_loss_mask = tf.reshape(voxel_loss_mask, [-1])
            voxel_loss_mask = tf.cast(voxel_loss_mask, tf.float32)
            mse_loss = tf.multiply(voxel_loss_mask, mse_loss)
        mse_loss = tf.reduce_mean(mse_loss)

        return mse_loss

    def _final_graph_reconstruction_loss(self, g_vert_in, g_edge_in, g_vert_out, g_edge_out,
                                         vert_loss_weight=1.0, edge_loss_weight=1.0):
        """calculate the reconstruction loss for the graph representation"""
        g_vert_in = tf.reshape(g_vert_in, [-1])
        g_vert_out = tf.reshape(g_vert_out, [-1])

        g_edge_in = tf.reshape(g_edge_in, [-1])
        g_edge_out = tf.reshape(g_edge_out, [-1])

        g_vert_mse_loss = tf.reduce_mean(tf.pow(g_vert_in - g_vert_out, 2))
        g_edge_mse_loss = tf.reduce_mean(tf.pow(g_edge_in - g_edge_out, 2))

        g_total_mse_loss = g_vert_mse_loss * vert_loss_weight + g_edge_mse_loss * edge_loss_weight
        return g_total_mse_loss

    def _final_graph_kl_loss(self, mu, log_sigma):
        """calculate the kl loss for the graph representation"""
        kl_loss = -0.5 * tf.reduce_sum(1 + log_sigma - tf.pow(mu, 2) - tf.exp(log_sigma), reduction_indices=1)
        kl_loss = tf.reduce_mean(kl_loss)
        return kl_loss

    ##############################################################################
    # Functions to output geometry and structure features
    ##############################################################################
    def pred_voxel_and_bbox(self):
        """output results of voxel map and bounding box"""
        voxel_pred = self.get_output('voxel_decoder')
        bbox_pred = self.get_output('bbox_pred')

        return voxel_pred, bbox_pred

    ##############################################################################
    # Functions for checking input data
    ##############################################################################
    def check_feeds(self, inputs_data):
        part_voxels = inputs_data['part_voxels']
        part_bbox = inputs_data['part_bbox']
        rel_pair_mask_inds = inputs_data['rel_pair_mask_inds']
        part_visible_masks = inputs_data['part_visible_masks']

        if part_voxels[0].shape[0] != part_bbox[0].shape[0]:
            raise KeyError("voxel array and bbox array must have the same size")
        if part_visible_masks[0].shape[0] != part_voxels[0].shape[0] / self.max_part_size:
            raise KeyError("part visible masks and voxel array mush have the same size")

        input_feed = {}
        input_feed[self.part_voxels] = part_voxels
        input_feed[self.part_bboxs] = part_bbox
        input_feed[self.gaussian_noise] = inputs_data['gaussian_noise']
        input_feed[self.edge_pair_mask_inds] = rel_pair_mask_inds
        input_feed[self.part_visible_masks] = inputs_data['part_visible_masks']
        input_feed[self.vert_lr] = float(inputs_data['vert_lr'])
        input_feed[self.edge_lr] = float(inputs_data['edge_lr'])
        input_feed[self.graph_gen_lr] = float(inputs_data['graph_gen_lr'])
        input_feed[self.recon_gen_loss_ratio] = float(inputs_data['recon_gen_loss_ratio'])
        input_feed[self.voxel_bbox_ratio] = float(inputs_data['voxel_bbox_ratio'])
        input_feed[self.g_rec_kl_loss_ratio] = float(inputs_data['g_rec_kl_loss_ratio'])
        input_feed[self.max_gradient_norm] = float(inputs_data['max_gradient_norm'])
        input_feed[self.voxel_loss_weights] = inputs_data['part_voxel_loss_weights']
        input_feed[self.part_bbox_loss_masks] = inputs_data['part_bbox_loss_masks']

        return input_feed

    ##############################################################################
    # Additional function to compute the final output voxel maps and bounding boxes
    ##############################################################################
    def get_batch_info(self, sess, inputs_data):
        input_feed = self.check_feeds(inputs_data)
        part_visible_masks = sess.run(self.part_visible_masks[self.gpu_num - 1], feed_dict=input_feed)

        d_voxels = sess.run(self.voxel_pred, feed_dict=input_feed)
        d_bboxs = sess.run(self.bbox_pred, feed_dict=input_feed)

        # part_visible_masks = sess.run(self.part_visible_masks[self.gpu_num - 1], feed_dict=input_feed)
        voxels_list = self.data_helper.process_voxel_data(d_voxels, part_visible_masks)
        bboxs_list = self.data_helper.process_bbox_data(d_bboxs, part_visible_masks)

        return voxels_list, bboxs_list, part_visible_masks

    ##############################################################################
    # Functions to train and test
    ##############################################################################
    def train(self, sess, inputs_data, iter_n, is_summary=False):
        input_feed = self.check_feeds(inputs_data)

        keep_prob = self.data['config_dict']['TRAIN']['DROPOUT_KEEP_PROB']
        input_feed[self.keep_prob] = keep_prob

        total_iter_n = int(self.data['config_dict']['TRAIN']['ITER_NUM'])

        if is_summary:
            output_feed = [self.train_op, self.total_losses, self.voxel_loss, self.bbox_loss, self.summary_op]
        else:
            output_feed = [self.train_op, self.total_losses, self.voxel_loss, self.bbox_loss]

        outputs = sess.run(output_feed, input_feed)

        print ("[%6d/%6d], total loss: %.8f, voxel loss: %.8f, bbox loss: %.8f" %
               (int(iter_n), total_iter_n, outputs[1], outputs[2], outputs[3]))

        if is_summary:
            return outputs[1], outputs[4]
        else:
            return outputs[1]

    def test(self, sess, inputs_data):
        input_feed = {}
        input_feed[self.latent_z] = inputs_data['gaussian_noise'][self.gpu_num - 1]
        input_feed[self.part_visible_masks] = inputs_data['visible_part_index']

        d_voxels = sess.run(self.voxel_pred, feed_dict=input_feed)
        d_bboxs = sess.run(self.bbox_pred, feed_dict=input_feed)

        part_visible_masks = inputs_data['visible_part_index'][self.gpu_num - 1]

        voxels_list = self.data_helper.process_voxel_data(d_voxels, part_visible_masks)
        bboxs_list = self.data_helper.process_bbox_data(d_bboxs, part_visible_masks)

        return voxels_list, bboxs_list, part_visible_masks