# run the optical flow inference
echo "[run_inverse]: optical flow estimation"
PYTHONPATH=. python optical_flow/AnimeRun/inference.py

# run the data preprocessing for inverse prediction
echo "[run_inverse]: data preprocessing"
PYTHONPATH=. python data_preprocessing/preprocess_inverse.py

# run the main scripts for global layer transformation, local layer transformation, endpoint matching, and control point matching
echo "[run_inverse]: main processes"
python main1_global_transform.py
python main2_local_transform.py
python main3_endpoint_inference.py
python main4_ctrlpoint_inference.py