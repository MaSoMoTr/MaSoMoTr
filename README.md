# Markerless Mice Tracking for Social Experiments

This is an implementation of the pipeline to track unmarked mice of similar appearance. The technique and characterization are described in the paper https://biorxiv.org/cgi/content/short/2021.10.20.464614v1

## Installation
The code has been run successfully on Windows 7 and 10 with an NVIDIA GPU (Titan XP and RTX2070)
1. Clone this repository
2. [Anaconda](https://www.anaconda.com/distribution/) is highly recommended to install Python 3
3. Install dependencies with our provided Anaconda environments
   ```bash
   conda env create -f conda-environments/markerless-mice-tracking-windowsGPU.yaml
   ```

4. Activate the environment 

   ```bash
   conda activate markerless_mice_tracking_windowsGPU
   ```
To run Mask-RCNN and DeepLabCut using GPU, CUDA and cuDNN  must be installed according to Tensorflow documentation [GPU support](https://www.tensorflow.org/install/source#gpu). 
## Applying the algorithm on your own data
To apply the algorithm to new videos which have significantly different settings compared with our settings described in the paper, we recommend you to
retrain Mask-RCNN and Deeplabcut models on your own data.

- The workflow to train Mask-RCNN model can be found in the Jupyter Notebook *pipelines/mrcnn_training.ipynb*

- The instruction for training DLC model can be found in [DeepLabCut repository](https://github.com/DeepLabCut/DeepLabCut)

   DeepLabCut can be trained via GUI by executing:  `python -m deeplabcut`



To track mice in new videos, you can launch the GUI by executing:

   ```bash
   streamlit run app_markerless_mice_tracking.py -- --video=/path/to/video_dir/  --background=/path/to/background_dir/--mrcnn_model=/path/to/model_dir/ --dlc_model=/path/to/dlc_project_dir/
   ```

Follow the GUI to specify the inputs for the pipeline including video .avi, background .jpg, Mask-RCNN model .h5, config.yaml of DeepLabCut project.
* Output of the workflow are two csv files storing coordinates of snout and tailbase corresponding to two mice: *mouse1_ensemble.csv* and *mouse2_ensemble.csv*, and folder *tracking* containing masks of mice. 

## Pretrained models and video samples 
1. Mask-RCNN model [Download](http://people.ucalgary.ca/~kmurari/masomotr/trainedModels/mask_rcnn_mouse_0025.h5)
The model should be saved in the path:  *mrcnn_models/mask_rcnn_mouse_0025.h5*
2. Deeplabcut model [Download](http://people.ucalgary.ca/~kmurari/masomotr/trainedModels/dlc_mice_model.rar)
The zip file must be extracted and saved in the path:  *dlc_models/dlc_mice_tracking*

3. Mouse Tracking dataset [Download](http://people.ucalgary.ca/~kmurari/masomotr/MTdataset)

The current configuration allows you to track 2 mice in the videos, but you can expand to track more mice with proper configuration.

