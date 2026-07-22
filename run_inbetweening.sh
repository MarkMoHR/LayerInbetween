# run the stroke interpolation
echo "[run_inbetweening]: stroke interpolation"
PYTHONPATH=. python inbetweening_utils/stroke_interpolation.py

# run the occlusion mask generation
echo "[run_inbetweening]: occlusion mask generation"
conda run -n sam PYTHONPATH=. python video_tracking/sam2/occlusion_mask_gen.py

# run the occlusion resolving
echo "[run_inbetweening]: occlusion resolving"
PYTHONPATH=. python inbetweening_utils/occlusion_resolve.py

# run the inbetweening visualization
echo "[run_inbetweening]: inbetweening visualization"
PYTHONPATH=. python inbetweening_utils/vis_inbetween.py