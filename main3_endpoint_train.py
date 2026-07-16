import json
import os
import six

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["KMP_WARNINGS"] = "0"

import model3_endpoint_train as sketch_endpoint_model
from utils3_endpoint_train import load_dataset


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

    datasets = load_dataset(model_params)
    train_set = datasets[0]
    val_set = datasets[1]

    sub_snapshot_root = os.path.join(model_params.snapshot_root, model_params.workspace)

    model = sketch_endpoint_model.FullModel(model_params, train_set, val_set,
                                            None, sub_snapshot_root, None)
    if mode == 'inference':
        sub_inference_root = os.path.join(model_params.inference_root, model_params.workspace)
        os.makedirs(sub_inference_root, exist_ok=True)
        model.inference(sub_inference_root, show_data='selected')  # ['selected', 'all', 'occluded']
    elif mode == 'inference_full':
        sub_inference_root = os.path.join(model_params.inference_full_root, model_params.workspace)
        os.makedirs(sub_inference_root, exist_ok=True)
        model.inference_full(sub_inference_root, show_data='selected')  # ['selected', 'all']
    else:
        model.evaluate(load_trained_weights=True)
        # model.evaluate(load_trained_weights=True, occluded_only=True)


if __name__ == '__main__':
    mode = 'inference'  # ['train', 'test', 'inference', 'inference_full']

    model_params = sketch_endpoint_model.get_default_hparams()

    if mode == 'train':
        trainer(model_params)
    elif mode == 'test' or mode == 'inference' or mode == 'inference_full':
        tester(model_params, mode)
    else:
        raise Exception('Unknown mode:', mode)
