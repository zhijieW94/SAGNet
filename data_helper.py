#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ['QT_API'] = 'pyqt5'

import scipy.ndimage as nd
import scipy.io as io
import scipy.stats as stats
import numpy as np
from utils import *

try:
    import trimesh
    from stl import mesh
except:
    print 'All dependencies not loaded, some functionality may not work'

class data_runner(object):
    ##############################################################################
    # Functions to initialize the data helper
    ##############################################################################
    def __init__(self, config_dict, for_training):
        self.for_training = for_training

        self.load_config_info(config_dict=config_dict)

        if for_training:
            self.load_dataset(config_dict)  # load the dataset for a certain kind

    def load_config_info(self, config_dict):
        self.config_dict = config_dict

        self.cube_len = self.config_dict['CUBE_LEN']
        self.bbox_size = self.config_dict['BOUNDING_BOX_SIZE']
        self.embedding_size = self.config_dict['EMBEDDING_VECOTR_SIZE']
        self.max_part_size = self.config_dict['MAX_PART_SIZE']

        self.input_pls = {}
        self.obj_num = 0

        if self.for_training:
            self.gpu_nums = self.config_dict['TRAIN']['NUM_GPUS']
            self.batch_size = self.config_dict['TRAIN']['BATCH_SIZE']

            self.shape_name = self.config_dict['TRAIN']['SHAPE_NAME']

            self.voxel_array = np.array([])  # used to store the voxel information in a list
            self.bbox_array = np.array([])  # used to store the baounding box information in a list
            self.part_obj_index_list = []  # the index of part_obj_index_list stands for the index of a certain object and the value is the index of a part
            self.obj_part_index_list = []  # the index of this list is the index of an object and the value stands for the index of a certain part

            self.global_steps = tf.placeholder(tf.float32, shape=(), name="global_steps")

            initial_vert_lr = config_dict['TRAIN']['VERT_LEARNING_RATE']
            initial_edge_lr = config_dict['TRAIN']['EDGE_LEARNING_RATE']
            initial_graph_gen_lr = config_dict['TRAIN']['GRAPH_GEN_LEARNING_RATE']

            vert_gamma = config_dict['TRAIN']['VERT_GAMMA']
            edge_gamma = config_dict['TRAIN']['EDGE_GAMMA']
            graph_gen_gamma = config_dict['TRAIN']['GRAPH_GEN_GAMMA']

            vert_lr_decay_step = config_dict['TRAIN']['VERT_LR_DECAY_STEP']
            edge_lr_decay_step = config_dict['TRAIN']['EDGE_LR_DECAY_STEP']
            graph_gen_lr_decay_step = config_dict['TRAIN']['GRAPH_GEN_LR_DECAY_STEP']

            self.vert_lr = tf.train.exponential_decay(initial_vert_lr,
                                                      global_step=self.global_steps,
                                                      decay_steps=vert_lr_decay_step,
                                                      decay_rate=vert_gamma,
                                                      staircase=True)
            self.edge_lr = tf.train.exponential_decay(initial_edge_lr,
                                                      global_step=self.global_steps,
                                                      decay_steps=edge_lr_decay_step,
                                                      decay_rate=edge_gamma,
                                                      staircase=True)
            self.graph_gen_lr = tf.train.exponential_decay(initial_graph_gen_lr,
                                                           global_step=self.global_steps,
                                                           decay_steps=graph_gen_lr_decay_step,
                                                           decay_rate=graph_gen_gamma,
                                                           staircase=True)

            initial_kl_loss_ratio = config_dict['TRAIN']['RECON_GEN_INITIAL_LOSS_RATIO']
            kl_loss_gamma = config_dict['TRAIN']['RECON_GEN_RATIO_GAMMA']
            kl_loss_decay_step = config_dict['TRAIN']['RECON_GEN_DECAY_STEP']
            self.kl_loss_ratio = tf.train.exponential_decay(initial_kl_loss_ratio,
                                                            global_step=self.global_steps,
                                                            decay_steps=kl_loss_decay_step,
                                                            decay_rate=kl_loss_gamma,
                                                            staircase=False)
        else:
            self.gpu_nums = 1
            self.batch_size = self.config_dict['TRAIN']['BATCH_SIZE']

            self.shape_name = config_dict['model_info']['model_class']

            self.visible_part_indexes_array = config_dict['model_info']['part_masks']
            self.obj_num = len(config_dict['model_info']['part_masks'])

            self.bbox_means_array = config_dict['model_info']['bbox_info']['bbox_mean']
            self.bbox_stds_array = config_dict['model_info']['bbox_info']['bbox_variance']
            self.bbox_benchmark_array = config_dict['model_info']['bbox_info']['bbox_benchmark']


    ##############################################################################
    # Functions to load the dataset
    ##############################################################################
    def get_voxels_from_file(self, path, cube_len=32):
        voxels = io.loadmat(path)['voxels3D']
        if cube_len != 32 and cube_len == 64:
            voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
        return voxels

    def get_bboxs_from_file(self, obj_path):
        found_bbox = False

        bbox_file = open(obj_path, 'r')
        all_lines = bbox_file.readlines()
        for c_line in all_lines:
            if c_line.find('Min_Max') != -1:
                found_bbox = True
                continue

            if found_bbox:
                cur_line = c_line.strip()
                bbox_list = map(float, cur_line.split())
                bbox_list = np.array(bbox_list)
                return bbox_list

    def get_part_label(self, file_path):
        if file_path.find('.mat') == len(file_path) - 4:
            label_num = int(file_path[len(file_path) - 5 : len(file_path) - 4])
            return label_num

    def get_part_info(self, obj_path, cube_len=32):
        voxel_file_list = [f for f in os.listdir(obj_path) if f.endswith('.mat')]
        bbox_file_list = [f for f in os.listdir(obj_path) if f.endswith('.txt')]

        voxel_file_list.sort()
        bbox_file_list.sort()

        voxel_default_value = self.config_dict['TRAIN']['VOXEL_DEFAULT_VALUE']

        if len(voxel_file_list) != len(bbox_file_list):
            error_str = "the file number for voxel maps should correspond to the number of bounding boxes."
            raise KeyError(error_str)

        visible_part_index_list = np.asarray([self.get_part_label(f_name) for f_name in voxel_file_list], dtype=np.int).tolist()
        voxels_list = np.asarray([self.get_voxels_from_file(obj_path + f_name, cube_len) for f_name in voxel_file_list], dtype=np.float32)
        bboxs_list = np.array([self.get_bboxs_from_file(os.path.join(obj_path, f_name)) for f_name in bbox_file_list])

        final_part_mask_list = [[]] * self.max_part_size
        final_part_voxel_list = [[]] * self.max_part_size
        final_part_bbox_list = [[]] * self.max_part_size

        for index, label in enumerate(visible_part_index_list):
            final_part_voxel_list[label] = voxels_list[index]
            final_part_bbox_list[label] = bboxs_list[index]
            final_part_mask_list[label] = 1

        for part_index, mask in enumerate(final_part_mask_list):
            if not mask:
                # final_part_voxel_list[part_index] = np.zeros_like(voxels_list[0])
                final_part_voxel_list[part_index] = np.full(voxels_list[0].shape, float(voxel_default_value))
                final_part_bbox_list[part_index] = np.zeros_like(bboxs_list[0])
                final_part_mask_list[part_index] = 0

        voxels_array = np.asarray(final_part_voxel_list, dtype=np.float32)
        bboxs_array = np.asarray(final_part_bbox_list)
        part_mask_array = np.asarray(final_part_mask_list)

        return voxels_array, bboxs_array, part_mask_array

    def load_dataset(self, config_dict):
        data_path = config_dict['TRAIN']['DIR_PATH']
        data_path = os.path.join(data_path, self.shape_name)

        # fn_list: a list for file names
        fn_list = [f_path for f_path in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f_path))]
        # fp_list: a list for file paths
        fp_list = [os.path.join(data_path, f_path) for f_path in fn_list if os.path.isdir(os.path.join(data_path, f_path))]

        obj_num = len(fp_list)  # Total number of 3D shapes
        all_voxels_list = []    # a list for all the voxel maps
        all_bboxs_list = []     # a list for all the bounding boxes
        all_visible_part_indexes_list = []    # a list for the visible part indexes. Two-dimensional list: (shape_index, [visible_part_indexes])

        """part_obj_index_list is used to store the index of certain object the index of relation_list indicates a part
        in all_voxels_list or all_bboxs_list and the value is the index of certain object"""
        part_index = 0
        obj_part_index_list = []    # two-dimensional list
        part_obj_index_list = []    # one-dimensional list

        for obj_index, f_path in enumerate(fp_list):
            cur_voxels, cur_bboxs, cur_visible_part_index_list = self.get_part_info(f_path + "/")
            if cur_voxels.shape[0] != cur_bboxs.shape[0]:
                raise KeyError("Wrong sizes between the current voxel map arrays and bounding box arrays!!!")

            for c_voxel in cur_voxels:
                all_voxels_list.append(c_voxel)
                part_obj_index_list.append(obj_index)

            cur_part_index_list = []
            for c_bbox in cur_bboxs:
                all_bboxs_list.append(c_bbox)
                cur_part_index_list.append(part_index)

                part_index = part_index + 1
            obj_part_index_list.append(cur_part_index_list)

            all_visible_part_indexes_list.append(cur_visible_part_index_list)

        obj_part_index_array = np.array(obj_part_index_list)
        file_name_array = np.array(fn_list)

        all_voxels_array = np.array(all_voxels_list, dtype=np.float32)
        all_bboxs_array = np.array(all_bboxs_list)
        visible_part_index_array = np.array(all_visible_part_indexes_list)

        all_bboxs_array = self.preprocess_bbox_data(all_bboxs_array, visible_part_indexes=visible_part_index_array)

        self.obj_num, self.voxel_array, self.bbox_array, self.visible_part_indexes_array,\
        self.part_obj_index_list, self.obj_part_index_array, self.fn_list = \
            (obj_num, all_voxels_array, all_bboxs_array, visible_part_index_array, part_obj_index_list, obj_part_index_array, file_name_array)

        return obj_num, all_voxels_array, all_bboxs_array, all_visible_part_indexes_list, part_obj_index_list, obj_part_index_array


    ##############################################################################
    # Functions to preprocess the original input data
    ##############################################################################
    def preprocess_voxel_data(self, voxel_input):
        voxel_data = np.reshape(voxel_input, [-1, self.max_part_size, self.cube_len*self.cube_len*self.cube_len])
        voxel_data = voxel_data.astype(float)

        final_voxel_mean_list = []
        final_voxel_std_list = []
        final_voxel_list = []

        for part_index in range(self.max_part_size):
            cur_part_data = voxel_data[:, part_index, :]
            cur_part_data = np.reshape(cur_part_data, [-1])

            cur_mean = np.mean(cur_part_data)
            final_voxel_mean_list.append(cur_mean)

            cur_part_data -= cur_mean

            cur_std = np.std(cur_part_data)
            final_voxel_std_list.append(cur_std)

            cur_part_data /= cur_std

            cur_part_data = np.reshape(cur_part_data, [-1, self.cube_len*self.cube_len*self.cube_len])
            final_voxel_list.append(cur_part_data)

        self.voxel_means_array = np.array(final_voxel_mean_list)
        self.voxel_stds_array = np.array(final_voxel_std_list)

        if len(self.voxel_means_array) != len(self.voxel_stds_array):
            raise KeyError("The voxel means array and stds array don't have the same shape")

        final_data_array = np.stack(final_voxel_list, axis=1)
        final_data_array = np.reshape(final_data_array, [-1, self.cube_len, self.cube_len, self.cube_len])
        return final_data_array

    def preprocess_bbox_data(self, bbox_input, visible_part_indexes=None):
        """There are two steps in preprocessing bbox information. One is to transform the bounding boxes into offset forms. And the second step
            is to scale the processed bounding boxes with extracted means and variances."""
        bbox_data = np.reshape(bbox_input, [-1, self.max_part_size, self.bbox_size])

        # calculate average values for bounding boxes
        average_bboxs = self.average_bboxs(bbox_data, remove_zero=True)

        # complete the missing data
        bbox_data = self.compute_missing_data(bbox_data, visible_part_indexes)

        transformed_bbox_list = []
        # transform the bounding boxes into the offset form
        for part_index in range(self.max_part_size):
            cur_part_data = bbox_data[:, part_index, :]

            target_dx = (cur_part_data[:, 0] - average_bboxs[part_index][0]) / average_bboxs[part_index][3]
            target_dy = (cur_part_data[:, 1] - average_bboxs[part_index][1]) / average_bboxs[part_index][4]
            target_dz = (cur_part_data[:, 2] - average_bboxs[part_index][2]) / average_bboxs[part_index][5]
            target_dl = np.log(cur_part_data[:, 3] / average_bboxs[part_index][3])
            target_dw = np.log(cur_part_data[:, 4] / average_bboxs[part_index][4])
            target_dh = np.log(cur_part_data[:, 5] / average_bboxs[part_index][5])

            normalized_part_array = np.stack([target_dx, target_dy, target_dz, target_dl, target_dw, target_dh], axis=1)
            transformed_bbox_list.append(normalized_part_array)

        bboxs_array = np.stack(transformed_bbox_list, axis=1)
        bboxs_array = np.reshape(bboxs_array, [-1, self.bbox_size])

        # scale the bounding boxes with computed means and variances
        final_data_array = self.scale_bbox_data(bboxs_array)

        return final_data_array

    def compute_missing_data(self, bbox_input, visible_part_indexes=None):
        bbox_data = np.reshape(bbox_input, [-1, self.max_part_size, self.bbox_size])

        for obj_ind, part_masks in enumerate(visible_part_indexes):
            for p_ind, p_mask in enumerate(part_masks):
                if not p_mask:
                    bbox_data[obj_ind][p_ind] = self.bbox_benchmark_array[p_ind]

        return bbox_data

    def average_bboxs(self, bbox_input, remove_zero=True):
        """The benchmark array indicates the mean value for each part"""
        bboxs_mean_list = []
        for part_index in range(self.max_part_size):
            cur_part_data = bbox_input[:, part_index, :]

            if remove_zero:
                cur_mean_list = np.ma.masked_equal(cur_part_data, 0.0).mean(axis=0)
            else:
                cur_mean_list = np.mean(cur_part_data, axis=0)
            bboxs_mean_list.append(cur_mean_list)

        # The benchmark array indicates the mean value for each part
        self.bbox_benchmark_array = np.array(bboxs_mean_list)

        return self.bbox_benchmark_array

    def bboxs_mode(self, bbox_input):
        bbox_data = np.reshape(bbox_input, [-1, self.max_part_size, self.bbox_size])

        bboxs_mode_list = []
        for part_index in range(self.max_part_size):
            cur_part_data = bbox_data[:, part_index, :]

            cur_mode_list = stats.mode(cur_part_data, axis=0)
            bboxs_mode_list.append(cur_mode_list)

        self.bbox_benchmark_array = np.array(bboxs_mode_list)

        return self.bbox_benchmark_array

    def bboxs_median(self, bbox_input):
        bbox_data = np.reshape(bbox_input, [-1, self.max_part_size, self.bbox_size])

        bboxs_median_list = []
        for part_index in range(self.max_part_size):
            cur_part_data = bbox_data[:, part_index, :]

            mask_part_data = np.ma.masked_where(cur_part_data == 0, cur_part_data)
            cur_median_list = np.ma.median(mask_part_data, axis=0).filled(0)

            # cur_median_list = np.median(cur_part_data, axis=0)
            bboxs_median_list.append(cur_median_list)

        self.bbox_benchmark_array = np.array(bboxs_median_list)

        return self.bbox_benchmark_array

    def scale_bbox_data(self, bbox_input):
        bbox_data = np.reshape(bbox_input, [-1, self.max_part_size, self.bbox_size])

        final_bbox_mean_list = []
        final_bbox_std_list = []
        final_data_list = []

        for part_index in range(self.max_part_size):
            cur_part_data = bbox_data[:, part_index, :]

            cur_mean_list = np.mean(cur_part_data, axis=0)
            final_bbox_mean_list.append(cur_mean_list)

            cur_part_data -= cur_mean_list

            cur_std_list = np.std(cur_part_data, axis=0)
            final_bbox_std_list.append(cur_std_list)

            cur_std_array = np.array(cur_std_list)
            cur_part_data /= cur_std_array

            final_data_list.append(cur_part_data)

        self.bbox_means_array = np.array(final_bbox_mean_list)
        self.bbox_stds_array = np.array(final_bbox_std_list)

        if self.bbox_means_array.shape[0] != self.bbox_stds_array.shape[0] or\
                self.bbox_means_array.shape[1] != self.bbox_stds_array.shape[1]:
            raise KeyError("The bbox means array and bbox stds array don't have the same shape")

        final_data_array = np.stack(final_data_list, axis=1)
        final_data_array = np.reshape(final_data_array, [-1, self.bbox_size])

        return final_data_array


    ##############################################################################
    # Functions to prepare the data for next batch
    ##############################################################################
    def get_placeholders(self):
        if self.for_training:
            input_pls = self.get_placeholders_for_training()
        else:
            input_pls = self.get_placeholders_for_testing()

        return input_pls

    def get_placeholders_for_testing(self):
        embedding_size = self.config_dict['EMBEDDING_VECOTR_SIZE']

        input_pls = {
            'part_visible_masks': tf.placeholder(dtype=tf.int32, shape=[None, self.batch_size, self.max_part_size], name='part_visible_masks'),  # the mask array for visible parts
            'latent_codes': tf.placeholder(dtype=tf.float32, shape=[self.batch_size, embedding_size], name='latent_codes'),  # the mask array for visible parts
        }
        return input_pls

    def get_placeholders_for_training(self):
        cube_len = self.cube_len
        bbox_size = self.bbox_size

        input_pls = {
            'part_voxels': tf.placeholder(dtype=tf.float32, shape=[None, None, cube_len, cube_len, cube_len, 1], name='cur_part_voxels'),
            'part_bbox': tf.placeholder(shape=[None, None, bbox_size], dtype=tf.float32, name='part_bbox'),
            'gaussian_noise': tf.placeholder(shape=[None, None, self.embedding_size], dtype=tf.float32, name='gaussian_noise'),
            'labels': tf.placeholder(dtype=tf.int32, shape=[None], name='labels'),  # label of each part per batch
            'rel_pair_mask_inds': tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name='rel_pair_mask_inds'), # the indexes of two nodes for a bounding box pair
            'part_visible_masks': tf.placeholder(dtype=tf.int32, shape=[None, self.batch_size, self.max_part_size], name='part_visible_masks'),  # the mask array for visible parts
            # the weight vector for a part's loss
            'part_voxel_loss_weights': tf.placeholder(dtype=tf.float32, shape=[None, self.max_part_size], name='part_voxel_loss_weights'),
            'part_bbox_loss_masks': tf.placeholder(dtype=tf.float32, shape=[None, None, self.max_part_size * (self.max_part_size - 1) / 2], name='part_bbox_loss_masks'),  # the loss mask for a part's loss
            'vert_lr': tf.placeholder(dtype=tf.float32, shape=(), name='vert_learning_rate'),
            'edge_lr': tf.placeholder(dtype=tf.float32, shape=(), name='edge_learning_rate'),
            'graph_gen_lr': tf.placeholder(dtype=tf.float32, shape=(), name='graph_gen_learning_rate'),
            'recon_gen_loss_ratio': tf.placeholder(dtype=tf.float32, shape=(), name='recon_generation_loss_ratio'),
            'voxel_bbox_ratio': tf.placeholder(dtype=tf.float32, shape=(), name='voxel_bbox_ratio'),
            'g_rec_kl_loss_ratio': tf.placeholder(dtype=tf.float32, shape=(), name='graph_recon_kl_loss_ratio'),
            'max_gradient_norm': tf.placeholder(dtype=tf.float32, shape=(), name='max_gradient_norm')
        }

        self.input_pls = input_pls

        return dict(input_pls)

    def get_inputs_for_testing(self, cur_iter):
        cur_selected_obj_idx = np.random.randint(self.obj_num, size=[self.gpu_nums, self.batch_size])

        current_visible_part_index = self.visible_part_indexes_array[cur_selected_obj_idx]
        gaussian_noise = np.random.normal(0, 1, [self.gpu_nums, self.batch_size, self.embedding_size])
        # gaussian_noise = np.full((self.gpu_nums, self.batch_size, self.embedding_size), 1) * (cur_iter + 1)

        print "Gaussian noise:\n"
        print gaussian_noise

        input_pls = {}
        input_pls['gaussian_noise'] = gaussian_noise
        input_pls['visible_part_index'] = current_visible_part_index
        input_pls['selected_model_idx'] = cur_selected_obj_idx

        return input_pls

    def get_next_minibatch(self, object_index=None, cur_iter=-1):
        """We can set the object_index to be not None to retrieve certain objects"""
        if object_index is not None:
            cur_selected_obj_idx = object_index
        else:
            cur_selected_obj_idx = np.random.randint(self.obj_num, size=[self.gpu_nums, self.batch_size])

        selected_part_index = self.obj_part_index_array[cur_selected_obj_idx]   # two dimensional array
        selected_visible_part_index = self.visible_part_indexes_array[cur_selected_obj_idx]

        final_selected_part_index_list = np.reshape(selected_part_index, [len(selected_part_index), -1])

        # get the voxel map/bounding boxes/part label for current mini batch
        selected_part_voxels_list, selected_part_bboxs_list = self.prepare_train_batch(final_selected_part_index_list)

        # Compute the edge indexes for each bounding box pair
        total_edge_mask_inds = []
        for gpu_iter in range(self.gpu_nums):
            current_begin_index = 0
            cur_edge_pair_mask_inds = []
            for s_index in range(self.batch_size):
                for f_offset in range(self.max_part_size):
                    for s_offset in range(f_offset + 1, self.max_part_size):
                        tmp_list = [f_offset + current_begin_index, s_offset + current_begin_index]
                        cur_edge_pair_mask_inds.append(tmp_list)
                current_begin_index = current_begin_index + self.max_part_size
            total_edge_mask_inds.append(cur_edge_pair_mask_inds)

        part_voxel_arr = np.array(selected_part_voxels_list)
        part_voxel_arr = part_voxel_arr[:, :, :, :, :, np.newaxis]
        part_bbox_arr = np.array(selected_part_bboxs_list)

        # Compute the loss mask for each bounding box pair
        cur_bbox_loss_masks = []
        for g_part_visible_masks in selected_visible_part_index:    # per GPU
            cur_g_loss_masks = []
            for b_part_visible_masks in g_part_visible_masks:   # per batch
                cur_batch_loss_masks = []
                for f_p_index in range(len(b_part_visible_masks)):
                    for s_p_index in range(f_p_index + 1, len(b_part_visible_masks)):
                        # Only compute the bounding box pair losses when at least one part is visible
                        if b_part_visible_masks[f_p_index] + b_part_visible_masks[s_p_index] > 0:
                            cur_batch_loss_masks.append(1)
                        else:
                            cur_batch_loss_masks.append(0)
                cur_g_loss_masks.append(cur_batch_loss_masks)
            cur_bbox_loss_masks.append(cur_g_loss_masks)

        bbox_loss_mask_array = np.array(cur_bbox_loss_masks)

        voxel_loss_weights = None
        shape_name = self.config_dict['TRAIN']['SHAPE_NAME']
        if shape_name == 'motorbike':
            voxel_loss_weights = np.array([[1, 1, 1, 1, 1], ] * self.batch_size)  # motorbike
        elif shape_name == 'chair':
            voxel_loss_weights = np.array([[40, 5, 5, 1, 4], ] * self.batch_size)  # chair
        elif shape_name == 'airplane':
            voxel_loss_weights = np.array([[1, 1, 1, 1, 1, 1], ] * self.batch_size)  # airplane
        elif shape_name == 'guitar':
            voxel_loss_weights = np.array([[1, 1, 1], ] * self.batch_size)  # guitar
        elif shape_name == 'lamp':
            voxel_loss_weights = np.array([[5, 20, 1, 1], ] * self.batch_size)  # lamp
        elif shape_name == 'toy_examples':
            voxel_loss_weights = np.array([[1, 20], ] * self.batch_size)  # toy_examples

        data_pls = self.get_batch_config_dict(config_dict=self.config_dict, cur_iter=cur_iter)
        data_pls['part_voxels'] = part_voxel_arr
        data_pls['part_bbox'] = part_bbox_arr
        data_pls['rel_pair_mask_inds'] = total_edge_mask_inds
        data_pls['selected_model_idx'] = cur_selected_obj_idx
        data_pls['part_visible_masks'] = selected_visible_part_index
        data_pls['part_voxel_loss_weights'] = voxel_loss_weights
        data_pls['part_bbox_loss_masks'] = bbox_loss_mask_array

        return dict(data_pls)

    def get_batch_config_dict(self, config_dict, cur_iter=-1):
        # Compute the current learning rate for different components in the network
        f_dict = {}
        f_dict['vert_lr'] = self.vert_lr.eval(feed_dict={self.global_steps: cur_iter})
        f_dict['edge_lr'] = self.edge_lr.eval(feed_dict={self.global_steps: cur_iter})
        f_dict['graph_gen_lr'] = self.graph_gen_lr.eval(feed_dict={self.global_steps: cur_iter})

        kl_anneal_iter = config_dict['TRAIN']['KL_ANNEAL_ITER']
        if cur_iter > kl_anneal_iter:
            final_kl_ratio = config_dict['TRAIN']['FINAL_KL_RATIO']

            kl_loss_ratio = self.kl_loss_ratio.eval(feed_dict={self.global_steps: (cur_iter - kl_anneal_iter)})
            kl_loss_rate = max(kl_loss_ratio, final_kl_ratio)
        else:
            kl_loss_rate = 1.0

        print "vert_lr: %6f, edge_lr: %6f, graph_gen_lr: %6f, kl_loss_ratio: %6f" % \
              (f_dict['vert_lr'], f_dict['edge_lr'], f_dict['graph_gen_lr'], kl_loss_rate)

        f_dict['recon_gen_loss_ratio'] = kl_loss_rate

        f_dict['voxel_bbox_ratio'] = config_dict['TRAIN']['VOXEL_BBOX_LOSS_RATIO']

        f_dict['g_rec_kl_loss_ratio'] = config_dict['TRAIN']['GRAPH_REC_KL_LOSS_RATIO']

        embedding_size = config_dict['EMBEDDING_VECOTR_SIZE']
        f_dict['gaussian_noise'] = np.random.normal(0, 1, [self.gpu_nums, self.batch_size, embedding_size])

        f_dict['max_gradient_norm'] = float(config_dict['TRAIN']['MAX_GRADIENT_NORM'])

        return f_dict

    def prepare_train_batch(self, selected_part_index_list):
        total_selected_voxels_list = []
        total_selected_bboxs_list = []

        for s_part_indexs in selected_part_index_list:
            selected_part_voxels_list = []
            selected_part_bboxs_list = []

            for part_index in s_part_indexs:
                if part_index != -1:
                    selected_part_voxels_list.append(self.voxel_array[part_index])
                    selected_part_bboxs_list.append(self.bbox_array[part_index])
                else:
                    zero_voxel_array = np.zeros([self.cube_len, self.cube_len, self.cube_len], dtype=np.float32)
                    zero_bbox_array = np.zeros(self.bbox_size)

                    selected_part_voxels_list.append(zero_voxel_array)
                    selected_part_bboxs_list.append(zero_bbox_array)

            total_selected_voxels_list.append(selected_part_voxels_list)
            total_selected_bboxs_list.append(selected_part_bboxs_list)

        return total_selected_voxels_list, total_selected_bboxs_list


    ##############################################################################
    # Other functions for data processing
    ##############################################################################
    def clip_voxels_data(self, voxel_data):
        """Clip the voxel value to zero or one by evaluating the threshold 0.5"""
        result_voxels = np.copy(voxel_data)

        for x_ind in range(len(voxel_data)):
            for y_ind in range(len(voxel_data[0])):
                for z_ind in range(len(voxel_data[0][0])):
                    cur_value = voxel_data[x_ind][y_ind][z_ind]
                    if cur_value >= 0 and cur_value <= 0.5:
                        result_voxels[x_ind][y_ind][z_ind] = 0
                    elif cur_value > 0.5 and cur_value <= 1:
                        result_voxels[x_ind][y_ind][z_ind] = 1

        return result_voxels

    def process_voxel_data(self, voxel_data, part_visible_masks):
        """This method is used to turn the voxel value to zero or one"""
        voxel_data = np.reshape(voxel_data, [-1, self.cube_len, self.cube_len, self.cube_len])

        voxels_list = []
        start_index = 0

        for part_masks in part_visible_masks:
            cur_object_voxel_list = []
            for p_ind, p_mask in enumerate(part_masks):  # per part
                if int(p_mask) == 1:
                    voxel_index = start_index + p_ind
                    cur_voxel_map = voxel_data[voxel_index]
                    result_voxels = self.clip_voxels_data(cur_voxel_map)
                    cur_object_voxel_list.append(result_voxels)
            voxels_list.append(cur_object_voxel_list)
            start_index = start_index + self.max_part_size

        return voxels_list

    def process_bbox_data(self, bboxs_data, part_visible_masks):
        """"Split a 12D feature vector into two 6D vectors and then compute the final bounding box output by averaging
        the (k-1) corresponding bounding boxes for a part."""
        max_ctx_size = self.max_part_size * (self.max_part_size - 1) / 2

        bboxs_list = []
        start_index = 0
        for part_masks in part_visible_masks:
            object_ctx_arr = bboxs_data[start_index:start_index + max_ctx_size, :]
            object_ctx_arr = np.split(object_ctx_arr, 2, axis=1)

            start_index = start_index + max_ctx_size

            # indicates the first bounding box in a box pair
            f_bbox_arr = np.zeros([self.max_part_size, self.bbox_size], dtype=np.float32)
            # indicates the second bounding box in a box pair
            s_bbox_arr = np.zeros([self.max_part_size, self.bbox_size], dtype=np.float32)

            cur_part_index = 0
            for f_offset in range(self.max_part_size):
                for s_offset in range(f_offset + 1, self.max_part_size):
                    cur_first_bbox_value = object_ctx_arr[0][cur_part_index]
                    cur_second_bbox_value = object_ctx_arr[1][cur_part_index]

                    f_bbox_arr[f_offset] = f_bbox_arr[f_offset] + cur_first_bbox_value
                    s_bbox_arr[s_offset] = s_bbox_arr[s_offset] + cur_second_bbox_value
                    cur_part_index = cur_part_index + 1

            if self.max_part_size > 1:
                total_bbox_arr = (f_bbox_arr + s_bbox_arr) / (self.max_part_size - 1)
            else:
                total_bbox_arr = f_bbox_arr + s_bbox_arr

            cur_object_ctx_list = []  # store all the part bbox information for one certain object
            for p_ind, p_mask in enumerate(part_masks):
                if int(p_mask) == 1:
                    cur_object_ctx_list.append(total_bbox_arr[p_ind])
            bboxs_list.append(cur_object_ctx_list)

        return bboxs_list


    ##############################################################################
    # Functions to output results to files
    ##############################################################################
    def write_output_to_file(self, voxels_list=None, bboxs_list=None, part_visible_masks=None, input_info_dict=None,
                             output_dir=None, iter_n=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        selected_model_idx = input_info_dict['selected_model_idx']

        if len(voxels_list[0]) != len(bboxs_list[0]):
            raise KeyError("The voxels list and the bboxs list should share the same size when outputting")
        if len(voxels_list) != len(selected_model_idx[self.gpu_nums-1]):
            raise KeyError("The voxels list and the selected-model-idx list should share the same size when outputting")

        for obj_index, (iter_model_voxels, iter_model_bboxs, cur_model_idx, p_masks) in \
                enumerate(zip(voxels_list, bboxs_list, selected_model_idx[self.gpu_nums-1], part_visible_masks)):
            if len(iter_model_voxels) != len(iter_model_bboxs):
                raise KeyError("The iter_model_voxels and the iter_model_bboxs don't have the same size when outputting")

            if self.for_training:
                cur_model_name = self.fn_list[cur_model_idx]

                dir_name = "%s_iter_%d" % (cur_model_name, iter_n)
                output_model_dir = os.path.join(output_dir, dir_name)
                if not os.path.exists(output_model_dir):
                    os.makedirs(output_model_dir)

                file_name = dir_name
            else:
                cur_output_index = iter_n * self.batch_size + obj_index

                output_model_dir = os.path.join(output_dir, str(cur_output_index))
                if not os.path.exists(output_model_dir):
                    os.makedirs(output_model_dir)

                file_name = str(cur_output_index)

            start_index = 0
            for m_idx, p_mask in enumerate(p_masks):
                if p_mask:
                    self.write_voxels_to_file(iter_model_voxels[start_index], file_name, m_idx, output_model_dir)

                    cur_model_bbox = iter_model_bboxs[start_index] * self.bbox_stds_array[m_idx] + self.bbox_means_array[m_idx]
                    cur_model_bbox[0] = self.bbox_benchmark_array[m_idx][3] * cur_model_bbox[0] + self.bbox_benchmark_array[m_idx][0]
                    cur_model_bbox[1] = self.bbox_benchmark_array[m_idx][4] * cur_model_bbox[1] + self.bbox_benchmark_array[m_idx][1]
                    cur_model_bbox[2] = self.bbox_benchmark_array[m_idx][5] * cur_model_bbox[2] + self.bbox_benchmark_array[m_idx][2]
                    cur_model_bbox[3] = np.exp(cur_model_bbox[3]) * self.bbox_benchmark_array[m_idx][3]
                    cur_model_bbox[4] = np.exp(cur_model_bbox[4]) * self.bbox_benchmark_array[m_idx][4]
                    cur_model_bbox[5] = np.exp(cur_model_bbox[5]) * self.bbox_benchmark_array[m_idx][5]
                    self.write_bboxs_to_file(cur_model_bbox, file_name, m_idx, output_model_dir)
                    start_index = start_index + 1

                    print file_name + "_" + str(m_idx)

    def write_voxels_to_file(self, voxels_info, file_name, seq_num, output_dir):
        """The parameter dual_iter_index stands for the dual iteration index while the iter_index is the
           index of certain epoch and the seq_num stands for the index of a part inside certain object"""

        output_file_name = "%s_%d.mat" % (file_name, seq_num)
        output_file_path = os.path.join(output_dir, output_file_name)

        io.savemat(output_file_path, {'voxels3D': voxels_info})

    def write_bboxs_to_file(self, bboxs_info, file_name, seq_num, output_dir):
        """the parameter dual_iter_index stands for the dual iteration index while the iter_index is the
        index of certain epoch and the seq_num stands for the index of a part inside certain object"""

        output_file_name = "%s_%d_transform_info.txt" % (file_name, seq_num)
        output_file_path = os.path.join(output_dir, output_file_name)

        with open(output_file_path, "w") as out_f:
            out_f.write("6D vector for representation\n")
            for value in bboxs_info:
                output_content = "%f " % (value)
                out_f.write(output_content)
            out_f.write("\n")


    ##############################################################################
    # Functions to output model config info
    ##############################################################################
    def write_model_info_to_file(self, output_dir=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file_name = "%s_model_info.txt" % self.shape_name
        output_file_path = os.path.join(output_dir, output_file_name)

        with open(output_file_path, "w") as out_f:
            out_f.write("model_class\n")
            out_f.write("%s\n" % self.shape_name)

            out_f.write("\nbbox_mean\n")
            for p_means in self.bbox_means_array:
                for value in p_means:
                    output_content = "%f " % (value)
                    out_f.write(output_content)
                out_f.write("\n")

            out_f.write("\nbbox_variance\n")
            for p_stds in self.bbox_stds_array:
                for value in p_stds:
                    output_content = "%f " % (value)
                    out_f.write(output_content)
                out_f.write("\n")

            out_f.write("\nbbox_benchmark\n")
            for p_benchmark in self.bbox_benchmark_array:
                for value in p_benchmark:
                    output_content = "%f " % (value)
                    out_f.write(output_content)
                out_f.write("\n")

        mask_file_name = "%s_mask.mat" % self.shape_name
        mask_file_path = os.path.join(output_dir, mask_file_name)

        io.savemat(mask_file_path, {'masks': self.visible_part_indexes_array})
