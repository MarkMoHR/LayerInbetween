model_config_map = {
    'raster_size': 256,

    # Global layer/component transform
    'window_size_scaling_ref_comp': 2.0,
    'window_size_min_comp': 64,
    'window_size_scaling_times_tar_comp': (0.2, 2.0),

    # Local layer/component transform
    'window_size_scaling_ref_comp_local': 2.0,
    'window_size_min_comp_local': 64,
    'window_size_scaling_times_tar_comp_local': (0.2, 2.0),

    # Endpoint matching
    'window_size_scaling_ref_ep': 1.5,
    'window_size_min_ep': 64,

    # Control point matching
    'window_size_scaling_ref_ctrl': 1.5,
    'window_size_min_ctrl': 64,
}

