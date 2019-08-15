#!/usr/bin/env python
from SAG_network import *
from timer import Timer
from utils import *
import os
import time
import numpy as np

c_dict = load_from_yml("config.yml")
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, c_dict['TRAIN']['GPU_ID']))

def snapshot(sess, saver, output_dir, cur_iter, cur_shape, cur_time):
    """Take a snapshot of the network and save the trainable parameters."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cur_dir_name = "%s_%s" % (cur_shape, cur_time)
    if cur_dir_name is not None:
        current_output_dir = output_dir + cur_dir_name + "/"
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        filename = os.path.join(current_output_dir, '%s_%i.ckpt' % (cur_shape, cur_iter))
    else:
        filename = os.path.join(output_dir, '%s_%i.ckpt' % (cur_shape, cur_iter))

    saver.save(sess, filename)
    print 'Wrote snapshot to: {:s}'.format(filename)


def train_net(sess, config_dict):
    config_dict['TRAIN']['NUM_GPUS'] = len(config_dict['TRAIN']['GPU_ID'])

    shape_name = config_dict['TRAIN']['SHAPE_NAME']
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
    data_helper = data_runner(config_dict, for_training=True)
    inputs = data_helper.get_placeholders()

    # set the random seed
    random_seed = config_dict['TRAIN']['RANDOM_SEED']
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)

    inputs['config_dict'] = config_dict

    total_timer = Timer()
    total_timer.tic()

    cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    cur_dir_name = "%s_%s" % (shape_name, cur_time)

    train_net = SAGNet(inputs, data_helper=data_helper, for_training=True)
    print "setup GCDNet....................................................."
    train_net.setup()
    total_time_diff = total_timer.toc()

    print "------------------------------------------------------------------------------"
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        print "shape:" + str(shape)
        print "name:" + variable.name

    print "Initial time: step 1: build the network: %g" % (total_time_diff)

    results_dir = config_dict['TRAIN']['RESULTS_DIRECTORY']
    results_dir = os.path.join(results_dir, cur_dir_name)

    model_dir = config_dict['TRAIN']['MODEL_DIRECTORY']
    log_dir = config_dict['TRAIN']['LOG_DIRECTORY']
    checkpoint_path = config_dict['TRAIN']['PRETRAINED_MODEL_PATH']

    snapshot_freq = config_dict['TRAIN']['SNAPSHOT_FREQ']
    summary_freq = config_dict['TRAIN']['SUMMARY_FREQ']

    current_log_dir = os.path.join(log_dir, cur_dir_name)
    train_writer = tf.summary.FileWriter(current_log_dir, sess.graph)

    iter_timer = Timer()

    saver = tf.train.Saver(tf.trainable_variables())

    sess.run(tf.global_variables_initializer())

    if checkpoint_path != '' and checkpoint_path.endswith('.ckpt'):
        print "Restore model from checkpoints..."
        saver.restore(sess, checkpoint_path)
        print "Restore done."

        restore_epoch = os.path.splitext(os.path.basename(checkpoint_path))[0]
        restore_epoch = int(restore_epoch.split('_')[-1])

        print "Restore done. Current starting iteration: %d" % restore_epoch
    else:
        restore_epoch = 0

    current_iter = -1

    last_snapshot_iter = -1
    n_epochs = config_dict['TRAIN']['ITER_NUM']

    for cur_iter in range(restore_epoch, n_epochs):
        current_iter = cur_iter
        iter_timer.tic()

        f_dict = data_helper.get_next_minibatch(cur_iter=cur_iter)  # Load the input data for next mini-batch

        if cur_iter == 0:
            print "train neural network---------------------------------------------------"

        if (cur_iter + 1) % summary_freq == 0 or cur_iter < 20:
            _, summaries = train_net.train(sess, f_dict, cur_iter, is_summary=True)
            train_writer.add_summary(summaries, cur_iter)
        else:
            _ = train_net.train(sess, f_dict, cur_iter, is_summary=False)

        iter_diff = iter_timer.toc()
        print "Current iter: %d, time diff: %g" % (cur_iter, iter_diff)

        if (cur_iter + 1) % snapshot_freq == 0:
            last_snapshot_iter = cur_iter
            snapshot(sess, saver, model_dir, cur_iter, cur_shape=shape_name, cur_time=cur_time)

            voxels_list, bboxs_list, part_visible_masks = train_net.get_batch_info(sess, f_dict)
            data_helper.write_output_to_file(voxels_list=voxels_list, bboxs_list=bboxs_list,
                                             part_visible_masks=part_visible_masks,
                                             input_info_dict=f_dict, output_dir=results_dir,
                                             iter_n=current_iter)

    if last_snapshot_iter != current_iter:
        snapshot(sess, saver, model_dir, current_iter, cur_shape=shape_name, cur_time=cur_time)

        f_dict = data_helper.get_next_minibatch(cur_iter=current_iter)

        voxels_list, bboxs_list, part_visible_masks = train_net.get_batch_info(sess, f_dict)
        data_helper.write_output_to_file(voxels_list=voxels_list, bboxs_list=bboxs_list,
                                         part_visible_masks=part_visible_masks,
                                         input_info_dict=f_dict, output_dir=results_dir, iter_n=current_iter)


if __name__ == "__main__":
    with tf.Session() as sess:
        train_net(sess=sess, config_dict=c_dict)
