#!/usr/bin/env python
from SAG_network import *
from utils import *
import os
import time
import math

c_dict = load_from_yml("config.yml")
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, c_dict['TEST']['GPU_ID']))

def load_model_info(model_dir):
    info_dict = {}

    for f in os.listdir(model_dir):
        if f.find("model_info.txt") != -1:
            found_modelclass = False
            found_bbox_mean = False
            found_bbox_variance = False
            found_bbox_bemchmark = False

            bbox_dict = {}
            bbox_mean_list = []
            bbox_stds_list = []
            bbox_benchmark_list = []

            bbox_info_file = os.path.join(model_dir, f)
            model_info_file = open(bbox_info_file, 'r')
            all_lines = model_info_file.readlines()
            for c_line in all_lines:
                if c_line.find('model_class') != -1:
                    found_modelclass = True
                    continue
                if c_line.find('bbox_mean') != -1:
                    found_bbox_mean = True
                    continue
                if c_line.find('bbox_variance') != -1:
                    found_bbox_variance = True
                    continue
                if c_line.find('bbox_benchmark') != -1:
                    found_bbox_bemchmark = True
                    continue
                if c_line == "\n":
                    found_modelclass = False
                    found_bbox_mean = False
                    found_bbox_variance = False
                    found_bbox_bemchmark = False
                    continue

                if found_modelclass:
                    info_dict['model_class'] = str(c_line[:-1])
                    continue
                elif found_bbox_mean:
                    cur_line = c_line.strip()
                    cur_mean_list = map(float, cur_line.split())
                    cur_mean_array = np.array(cur_mean_list)
                    bbox_mean_list.append(cur_mean_array)
                    continue
                elif found_bbox_variance:
                    cur_line = c_line.strip()
                    cur_mean_list = map(float, cur_line.split())
                    cur_stds_array = np.array(cur_mean_list)
                    bbox_stds_list.append(cur_stds_array)
                    continue
                elif found_bbox_bemchmark:
                    cur_line = c_line.strip()
                    cur_benchmark_list = map(float, cur_line.split())
                    cur_benchmark_array = np.array(cur_benchmark_list)
                    bbox_benchmark_list.append(cur_benchmark_array)
                    continue

            bbox_dict['bbox_mean'] = np.array(bbox_mean_list)
            bbox_dict['bbox_variance'] = np.array(bbox_stds_list)
            bbox_dict['bbox_benchmark'] = np.array(bbox_benchmark_list)

            info_dict['bbox_info'] = bbox_dict

        if f.find(".mat") != -1:
            mask_info_file = os.path.join(model_dir, f)
            part_masks = io.loadmat(mask_info_file)['masks']

            info_dict['part_masks'] = part_masks

    if info_dict['bbox_info'] is not None and info_dict['part_masks'] is not None:
        return info_dict

    return None


def test_net(sess, config_dict):
    output_dir = config_dict['TEST']['RESULTS_DIRECTORY']
    checkpoint_path = config_dict['TEST']['PRETRAINED_MODEL_PATH']

    checkpoint_dir = os.path.abspath(os.path.dirname(checkpoint_path))
    info_dict = load_model_info(checkpoint_dir)
    if info_dict is not None:
        shape_name = info_dict['model_class']
        config_dict['model_info'] = info_dict
    else:
        raise KeyError("Can not find model info or part mask files in pretrained model directory!!!")

    max_part_size = 0
    if shape_name == 'motorbike':
        max_part_size = 5  # motorbike
    elif shape_name == 'chair':
        max_part_size = 5  # chair
    elif shape_name == 'airplane':
        max_part_size = 6  # airplane
    elif shape_name == 'guitar':
        max_part_size = 3  # guitar
    elif shape_name == 'lamp':
        max_part_size = 4  # lamp
    elif shape_name == 'toy_examples':
        max_part_size = 2  # toy_examples
    config_dict['MAX_PART_SIZE'] = max_part_size

    # Construct and initialize a data_runner
    data_helper = data_runner(config_dict, for_training=False)
    inputs = data_helper.get_placeholders()

    inputs['config_dict'] = config_dict

    test_net = SAGNet(inputs, data_helper=data_helper, for_training=False)
    print "setup SAGNet....................................................."
    test_net.setup()

    cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    cur_dir_name = "%s_%s" % (shape_name, cur_time)

    results_directory = os.path.join(output_dir, cur_dir_name)

    saver = tf.train.Saver(tf.trainable_variables())

    sess.run(tf.global_variables_initializer())

    # Restore the trained model
    if checkpoint_path != '' and checkpoint_path.endswith('.ckpt'):
        print "Restore model from checkpoints..."
        saver.restore(sess, checkpoint_path)
        print "Restore done."

    n_epochs = config_dict['TEST']['SAMPLE_SIZE']
    n_epochs = int(math.ceil(n_epochs / config_dict['TRAIN']['BATCH_SIZE']))

    print "Starting to synthesize 3D shapes."

    for cur_iter in range(n_epochs):
        f_dict = data_helper.get_inputs_for_testing(cur_iter = cur_iter)
        voxels_list, bboxs_list, part_visible_masks = test_net.test(sess, f_dict)
        data_helper.write_output_to_file(voxels_list=voxels_list, bboxs_list=bboxs_list,
                                         part_visible_masks=part_visible_masks, input_info_dict=f_dict,
                                         output_dir=results_directory, iter_n=cur_iter)

        print ("[%6d/%6d] %s Generation Done" % (int(cur_iter), int(n_epochs), shape_name))


if __name__ == "__main__":
    with tf.Session() as sess:
        test_net(sess=sess, config_dict=c_dict)