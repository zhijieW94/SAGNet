#!/usr/bin/env python
from SAG_network import *
from utils import *
import os
from datetime import datetime

c_dict = load_from_yml("config.yml")
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, c_dict['TEST']['GPU_ID']))

def test_net(sess, config_dict):
    shape_name = config_dict['TEST']['SHAPE_NAME']
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

    output_dir = config_dict['TEST']['RESULTS_DIRECTORY']
    checkpoint_path = config_dict['TEST']['PRETRAINED_MODEL_PATH']

    cur_time = str(datetime.now())
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

    print "Starting to synthesize 3D shapes."

    for cur_iter in range(n_epochs):
        f_dict = data_helper.get_inputs_for_testing()
        voxels_list, bboxs_list, part_visible_masks = test_net.test(sess, f_dict)
        data_helper.write_output_to_file(voxels_list=voxels_list, bboxs_list=bboxs_list,
                                         part_visible_masks=part_visible_masks, input_info_dict=f_dict,
                                         output_dir=results_directory, iter_n=cur_iter)
        print ("[%6d/%6d] Generation Done" % (int(cur_iter), int(n_epochs)))


if __name__ == "__main__":
    with tf.Session() as sess:
        test_net(sess=sess, config_dict=c_dict)