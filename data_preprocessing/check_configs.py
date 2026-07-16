import json
import shutil
from copy import deepcopy
from configs.example_configs import test_img_id, example_info_map, example_configs_path


def ensure_test_img_config(test_img_id, example_info_map, example_configs_path):
    test_img_key = str(test_img_id)
    if test_img_key in example_info_map:
        return False

    if not example_info_map:
        raise ValueError("example_info_map is empty; cannot create default configuration.")

    default_example = {
        "use_optical_flow": True,
        "optical_flow_method": "raft",
        "use_distance_transform": True,

        "use_target_layer": False,
        "use_target_layer_mask": "stroke",
        "target_layer_method": "box_depth_ol+mask_line",

        "target_layer_gen_configs": {
            "mask_dilate_size": 5
        },

        "inbetweening_configs": {
            "layering_method": "region_mask",
            "mask_dilate_size": -8,
            "order_type": "area",
            "manual_order": None,
        }
    }
    example_info_map[test_img_key] = default_example

    temp_path = example_configs_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(example_info_map, f, ensure_ascii=False, indent=4)
        f.write("\n")
    shutil.move(temp_path, example_configs_path)
    return True


if __name__ == "__main__":
    updated = ensure_test_img_config(test_img_id, example_info_map, example_configs_path)
    if updated:
        print(f"Added configuration for test_img_id={test_img_id} to {example_configs_path}.")
    else:
        print(f"Configuration for test_img_id={test_img_id} already exists in {example_configs_path}.")


