# LayerInbetween: Occlusion-Aware Stroke Correspondence and Inbetweening with Automatic Layering - SIGGRAPH 2026 (TOG)

[[Paper]](https://cislab.hkust-gz.edu.cn/media/documents/_SIGGRAPH_2026__LayerInbetween.pdf) | [[Project Page]](https://markmohr.github.io/LayerInbetween/)

This project can be used to produce **automatic inbetweening/2D animations** from clean line drawings, rough sketches, multi-character line art, complex scenes, abstract drawings, or multi-keyframe sequences.
Our vector-based approach can facilitate convenient **inbetweening editing**.

<img src='docs/figures/teaser3.png'>

<img src='docs/figures/gif-editing/horizontal-demo1.gif' height=180> &nbsp;&nbsp;&nbsp; <img src='docs/figures/gif-editing/horizontal-demo2.gif' height=180>


## Environment

Create a Conda environment named `layerinbetween` and install the main dependencies from `requirements.txt`:

```bash
conda create -n layerinbetween python=3.10 -y
conda activate layerinbetween
pip install -r requirements.txt
```

If some packages are missing from `requirements.txt`, their exact versions can be found in `requirements-full.txt`.

To avoid environment conflicts with SAM2, create a separate Conda environment named `sam` and install SAM2 using its [instructions](https://github.com/facebookresearch/sam2#installation).


## Getting Started

### Download Checkpoints

Our project relies on external models, including [optical flow estimation](https://github.com/lisiyao21/AnimeRun/tree/main/flow#pretrained-weights), [depth estimation](https://github.com/DepthAnything/Depth-Anything-V2#pre-trained-models), and [SAM2 video tracking](https://github.com/facebookresearch/sam2#download-checkpoints). Please download the corresponding models and place them according to the following file structure.
Our models can be downloaded [here](https://drive.google.com/file/d/1QKdDZc5uaYRXFBYIThIEF4vU1SmGrsWU/view?usp=sharing).

```
optical_flow/
  AnimeRun/
    checkpoints/
      20000_gma-animerun-v2-ft.pth
      20000_raft-animerun-v2-ft_again.pth

depth_estimation/
  Depth-Anything-V2/
    checkpoints/
      depth_anything_v2_vitl.pth

video_tracking/
  sam2/
    models/
      SAM2.1/
        sam2.1_hiera_large.pt

outputs/
  ctrlpoint/
    snapshot/
      FAD3-CP46-sep-dist/
        sketch_ctrlpoint_30000.pkl
  endpoint/
    snapshot/
      FAD3-EPO35-1.5x-min=64/
        sketch_endpoint_30000.pkl
  transform/
    snapshot/
      FAD3-T12-2.0x-51-min=64/
        sketch_transform_30000.pkl
      FAD3-T13-2.0x-51/
        sketch_transform_local_30000.pkl

vgg_utils/
  quickdraw-perceptual.pth
```

### Input Preparation

Prepare raster keyframes and the vector image for the first keyframe (following the tutorial [here](https://github.com/MarkMoHR/JoSTC/blob/main/tutorials/Krita_vector_generation.md) to create SVGs). Then, place them to `test_examples/raster_black/` and `test_examples/svg/` folders.

Please name them `X_ref.png`, `X_tar.png`, and `X.svg` using the same index `X`. 

If the raster keyframes are **rough sketches**, please place them in the `test_examples/raster_black/rough_raw/` folder first, and then run the following command for binarization and image squaring. The resulting clean keyframes will be saved to the `test_examples/raster_black/` folder:

```bash
python data_preprocessing/image_preprocess.py --image_id X
```

If the raster keyframes are **not square**, use the script and the command above after commenting out the `binarize(img)` and `darken(img)` in the script.


### Step 1: Forward Prediction

Go to [configs/example_configs.py](configs/example_configs.py), and set the `test_img_id` to `X`. If you want to use the provided cases directly, please set the `test_img_id` to the corresponding image index. Then, run the following command:

```bash
sh run_forward.sh
```

This script includes several sub-processes. Refer to it for details. Finally, the output results of vector stroke correspondence are placed in the `outputs/stroke_correspondence_results` folder.


### Step 2: Inverse Prediction

1. Prepare data for the inverse prediction:

```bash
PYTHONPATH=. python data_preprocessing/prepare_inverse_data.py
```
This will create a `test_examples/[0inv]/` folder, which stores raster images (i.e., alternating the original reference and target keyframes) and the SVG (i.e., the predicted vector image for the original target keyframe).

2. Make a vector image `X.svg` for the missing strokes following the tutorial [here](tutorials/Krita_complement_strokes.md).

3. Place the `X.svg` to the `test_examples/[0inv]/svg/` folder.

4. Set `do_inv = True` in [configs/example_configs.py](configs/example_configs.py) to indicate the prediction direction. Then, run the following command:

```bash
sh run_inverse.sh
```

This script also includes several sub-processes. Finally, the output results of vector stroke correspondence for the complementary strokes are placed in the `outputs/stroke_correspondence_results/[0inv]` folder.


### Step 3: Combine Two Directions

> Note that this step should be done even if the [Step 2: Inverse Prediction](#step-2-inverse-prediction) is not done

Combine the results of the two directions using the following command:

```bash
PYTHONPATH=. python data_preprocessing/combine_two_directions.py
```

Afterward, the results are saved to the `outputs/stroke_correspondence_results/[1comb]` folder.


### Step 4: Inbetweening

This step includes several sub-processes: stroke interpolation, occlusion mask generation, occlusion resolving, and inbetweening visualization.

Run the following command:

```bash
sh run_inbetweening.sh
```

Then, you can see:
- Original inbetweening frames in `outputs/inbetweening_results/X/linearts/`
- Inbetweening frames with occlusion resolved in `outputs/inbetweening_results/X/linearts_occ_resolve/`
- Inbetweening gif in `outputs/inbetweening_results/X/gifs/`
- Inbetweening image visualization in `outputs/inbetweening_results/X/inbetween_img/`


## Multi-frame Prediction

### Input Preparation

1. We first define the `K`-th generation: `K=1` denotes the second generation (3 keyframes); `K=2` denotes the third generation (4 keyframes); etc.

2. Place raster keyframes in `test_examples-GenK/raster_black/` (replace `K` with 1, 2, ...). Note that the `X_ref.png` should be the `X_tar.png` in the `K-1`-th generation.

3. Copy the output vector parameters (e.g., `outputs/stroke_correspondence_results/[1comb]/vector-params/X_tar.jsonl`) to `test_examples-GenK/vector-params/` folder, and rename it to `X_ref.jsonl`. Note that if inverse prediction was not done in the `K-1`-th generation, the [Step 3: Combine Two Directions](#step-3-combine-two-directions) should also be executed.

### Run

Go to [configs/example_configs.py](configs/example_configs.py), and set the `gen_time` to `K`. Then, perform the Step 1 to Step 3 above again.




## Citation

If you use the code and models, please cite:

```
@article{mo2026layerinbetween,
  title={LayerInbetween: Occlusion-Aware Stroke Correspondence and Inbetweening with Automatic Layering},
  author={Mo, Haoran and Guan, Zhongyue and Hu, Yixin and Wang, Zeyu},
  journal={ACM Transactions on Graphics (TOG)},
  volume={45},
  number={4},
  pages={1--18},
  year={2026},
  publisher={ACM New York, NY, USA}
}
```

