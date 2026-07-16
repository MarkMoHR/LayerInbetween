# run the config checking
echo "[run_forward]: config checking"
PYTHONPATH=. python data_preprocessing/check_configs.py

# run the optical flow inference
echo "[run_forward]: optical flow estimation"
PYTHONPATH=. python optical_flow/AnimeRun/inference.py

# run the data preprocessing for forward prediction
echo "[run_forward]: data preprocessing"
PYTHONPATH=. python data_preprocessing/preprocess_forward.py

# run the depth estimation inference
echo "[run_forward]: depth estimation"
PYTHONPATH=. python depth_estimation/Depth-Anything-V2/run.py

# run the SAM layer generation inside the sam conda environment
echo "[run_forward]: layer generation"
conda run -n sam PYTHONPATH=. python video_tracking/sam2/layer_gen.py

# run the main scripts for global layer transformation, local layer transformation, endpoint matching, and control point matching
echo "[run_forward]: main processes"
python main1_global_transform.py
python main2_local_transform.py
python main3_endpoint_inference.py
python main4_ctrlpoint_inference.py