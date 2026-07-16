import json

test_data_base = "test_examples"
test_img_id = 7
do_inv = False
gen_time = 0  # 0-first gen; 1-second gen

gpu_id = "1"

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
