import json

test_data_base = "test_examples"  # the base directory for test data
test_img_id = 7  # the test image ID to be used for testing; this should correspond to a subdirectory in test_data_base
do_inv = False  # whether to perform backward prediction; if False, we will perform forward prediction
gen_time = 0  # 0 for first gen (2 keyframes); 1 for second gen (3 keyframes); 2 for third gen (4 keyframes)

gpu_id = "1"  # the GPU ID to be used

if gen_time == 0:
    example_configs_path = "configs/example_configs.json"
else:
    example_configs_path = "configs/example_configs_gen%d.json" % gen_time

with open(example_configs_path, "r") as load_f:
    example_info_map = json.load(load_f)
# Keys:
#  - optical_flow_method: [gma, raft, flowdiffuser]
#  - use_distance_transform: [true, false], whether use distance transform for optical flow estimation
#  - use_target_layer: [true, false], whether use predicted target layer image for each reference group
#  - use_target_layer_mask: [none, stroke], whether use predicted target layer mask for reference stroke fixing
#  - target_layer_method: [box_depth, box_depth_ol, mask_line, box_depth+mask_line, box_depth_ol+mask_line]
#  - inbetweening_configs:
#    - layering_method: [box, line_mask, region_mask, line_mask+region_mask]
