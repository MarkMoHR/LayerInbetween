import json
import os
import six
from configs.example_configs import test_data_base, test_img_id, do_inv, gen_time, example_info_map, gpu_id
from configs.model_configs import model_config_map

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
os.environ["KMP_WARNINGS"] = "0"

import model2_local_transform as sketch_endpoint_model
from utils2_local_transform import load_dataset, load_real_dataset


def trainer(model_params):
    print('Hyperparams:')
    for key, val in six.iteritems(model_params.values()):
        print('%s = %s' % (key, str(val)))
    print('-' * 100)

    datasets = load_dataset(model_params)

    sub_log_root = os.path.join(model_params.log_root, model_params.workspace)
    sub_log_img_root = os.path.join(model_params.log_img_root, model_params.workspace)
    sub_snapshot_root = os.path.join(model_params.snapshot_root, model_params.workspace)
    os.makedirs(sub_log_root, exist_ok=True)
    os.makedirs(sub_log_img_root, exist_ok=True)
    os.makedirs(sub_snapshot_root, exist_ok=True)

    train_set = datasets[0]
    val_set = datasets[1]
    train_model_params = datasets[2]
    val_model_params = datasets[3]

    # Write config file to json file.
    with open(os.path.join(sub_snapshot_root, 'model_config.json'), 'w') as f:
        json.dump(train_model_params.values(), f, indent=True)

    model = sketch_endpoint_model.FullModel(model_params, train_set, val_set,
                                            sub_log_root, sub_snapshot_root, sub_log_img_root)
    model.train()
    model.evaluate()


def tester(model_params, mode):
    print('Hyperparams:')
    for key, val in six.iteritems(model_params.values()):
        print('%s = %s' % (key, str(val)))
    print('-' * 100)

    datasets = load_dataset(model_params, test_only=True)
    do_inv = False
    train_set = datasets[0]
    val_set = datasets[1]

    sub_snapshot_root = os.path.join(model_params.snapshot_root, model_params.workspace)

    model = sketch_endpoint_model.FullModel(model_params, train_set, val_set,
                                            None, sub_snapshot_root, None)
    if mode == 'inference':
        sub_inference_root = os.path.join(model_params.inference_root, model_params.workspace)
        if model_params.use_optical_flow:
            sub_inference_root += '-[optical]'
        os.makedirs(sub_inference_root, exist_ok=True)
        model.inference(sub_inference_root, show_data='all', do_inv=do_inv)  # ['selected', 'all', 'occluded']
    else:
        model.evaluate(load_trained_weights=True)
        # model.evaluate(load_trained_weights=True, occluded_only=True)


def tester_real(model_params, mode):
    # print('Hyperparams:')
    # for key, val in six.iteritems(model_params.values()):
    #     print('%s = %s' % (key, str(val)))
    # print('-' * 100)

    data_base = test_data_base
    if gen_time > 0:
        data_base += '-Gen' + str(gen_time)
    datasets = load_real_dataset(model_params, data_base=data_base, generation_time=gen_time)
    val_set = datasets[0]

    sub_snapshot_root = os.path.join(model_params.snapshot_root, model_params.workspace)

    model = sketch_endpoint_model.FullModel(model_params, None, val_set,
                                            None, sub_snapshot_root, None)
    if mode == 'inference_real':
        sub_inference_root = os.path.join(model_params.inference_root_real, model_params.workspace)
        if model_params.use_optical_flow:
            sub_inference_root += '-[optical]'
        model.inference_real(sub_inference_root, data_base)
    else:
        raise Exception('Unknown mode:', mode)


def tester_real_inv(model_params, mode):
    # print('Hyperparams:')
    # for key, val in six.iteritems(model_params.values()):
    #     print('%s = %s' % (key, str(val)))
    # print('-' * 100)

    data_base = test_data_base
    data_base_extra = "outputs/stroke_correspondence_results"
    if gen_time > 0:
        data_base += '-Gen' + str(gen_time)
        data_base_extra += '-[Gen%d]' % gen_time
    data_base = os.path.join(data_base, '[0inv]')
    datasets = load_real_dataset(model_params, data_base=data_base, data_base_extra=data_base_extra,
                                 generation_time=gen_time)
    val_set = datasets[0]

    sub_snapshot_root = os.path.join(model_params.snapshot_root, model_params.workspace)

    model = sketch_endpoint_model.FullModel(model_params, None, val_set,
                                            None, sub_snapshot_root, None)
    sub_inference_root = os.path.join(model_params.inference_root_real, model_params.workspace)
    if model_params.use_optical_flow:
        sub_inference_root += '-[optical]'
    model.inference_real(sub_inference_root, data_base, do_inv=True)


if __name__ == '__main__':
    mode = 'inference_real'  # ['train', 'test', 'inference', 'inference_real']
    if do_inv:
        mode += '_inv'

    model_params = sketch_endpoint_model.get_default_hparams()

    if 'real' in mode:
        # Add params for real inference
        model_params.add_hparam('use_optical_flow', example_info_map[str(test_img_id)]['use_optical_flow'])
        if not do_inv:
            model_params.add_hparam('use_target_layer', example_info_map[str(test_img_id)]['use_target_layer'])
            model_params.add_hparam('use_target_layer_mask', example_info_map[str(test_img_id)]['use_target_layer_mask'])
        else:
            model_params.add_hparam('use_target_layer', False)
            model_params.add_hparam('use_target_layer_mask', 'stroke')
        model_params.add_hparam('target_layer_method', example_info_map[str(test_img_id)]['target_layer_method'])
    else:
        model_params.add_hparam('use_optical_flow', True)
        model_params.add_hparam('use_target_layer_mask', 'stroke')
        model_params.add_hparam('target_layer_method', 'both')

    model_params.add_hparam('raster_size', model_config_map['raster_size'])
    model_params.add_hparam('window_size_scaling_ref_comp', model_config_map['window_size_scaling_ref_comp'])
    model_params.add_hparam('window_size_min_comp', model_config_map['window_size_min_comp'])

    model_params.add_hparam('window_size_scaling_ref', model_config_map['window_size_scaling_ref_comp_local'])
    model_params.add_hparam('window_size_min', model_config_map['window_size_min_comp_local'])
    model_params.add_hparam('window_size_scaling_times_tar', model_config_map['window_size_scaling_times_tar_comp_local'])

    if mode == 'train':
        trainer(model_params)
    elif mode == 'test' or mode == 'inference' or mode == 'inference_full':
        tester(model_params, mode)
    elif mode == 'inference_real':
        tester_real(model_params, mode)
    elif mode == 'inference_real_inv':
        tester_real_inv(model_params, mode)
    else:
        raise Exception('Unknown mode:', mode)
