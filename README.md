# Markerless Mice Tracking for Social Experiments

This is an implementation of the pipeline to track unmarked mice of similar appearance. The technique and characterization are described in the paper https://biorxiv.org/cgi/content/short/2021.10.20.464614v1

## Installation
The code has been run successfully on Windows 7 and 10 with NVIDIA GPUs (Titan XP and RTX2070)
1. Clone this repository
2. [Anaconda](https://www.anaconda.com/distribution/) is highly recommended to install Python 3
3. Install dependencies with our provided Anaconda environments
   ```bash
   conda env create -f conda-environments/MaSoMoTr.yaml
   ```

4. Activate the environment 

   ```bash
   conda activate MaSoMoTr
   ```
To run Mask-RCNN and DeepLabCut using GPU, CUDA and cuDNN  must be installed according to Tensorflow documentation [GPU support](https://www.tensorflow.org/install/source#gpu). 

## Applying the algorithm on your own data
To apply the algorithm to new videos which have significantly different settings compared with our settings described in the paper, we recommend you to
retrain Mask RCNN and Deeplabcut models on your own data.

Run the command line below to launch Streamlit-based UI 
   ```bash
   streamlit run app_markerless_mice_tracking.py -- --video=path_to_video_dir/  --background=path_to_background_image_dir/ --mrcnn_model=path_to_mrcnn_model_dir/ --dlc_project=path_to_dlc_project_dir/
   ```
If paths are not specified in the above command, streamlit will use default paths located in the project directory for loading videos and saving models. 
- Select the tab "1.Train maDeepLabCut" and follow the instructions for training a DLC model which can be found in [DeepLabCut repository](https://github.com/DeepLabCut/DeepLabCut)
- Select the tab "2.Train Mask R-CNN" and follow the workflow to train Mask-RCNN. A tutorial video for training the Mash RCNN (assuming the installation steps above have been completed) is at https://youtu.be/slhlq_QKNO8


To track mice in new videos, select the tab "3.Tracking Pipeline". A video tutorial for the process is at https://youtu.be/sfZiiN_cCCw
Follow the GUI to specify the inputs for the pipeline including video .avi, background .jpg, Mask-RCNN model .h5, config.yaml of DeepLabCut project.
* Output of the workflow are two CSV files storing coordinates of snout and tailbase corresponding to two mice: *mouse1.csv* and *mouse2.csv*, and a file *masks.h5* containing masks of the two mice. 


## Pretrained models and video samples 
1. Mask-RCNN model [Download](http://people.ucalgary.ca/~kmurari/masomotr/trainedModels/mask_rcnn_mouse_0025.h5)
The model should be saved in the path:  *mrcnn_models/mask_rcnn_mouse_0025.h5*
2. Deeplabcut model [Download](http://people.ucalgary.ca/~kmurari/masomotr/trainedModels/dlc_mice_model.zip)
The zip file must be extracted and saved in the path:  *dlc_models/dlc_mice_tracking*
3. Mouse Tracking dataset [Download](http://people.ucalgary.ca/~kmurari/masomotr/MTdataset)
4. Sample training and validation data used in the tutorials above [Download](http://people.ucalgary.ca/~kmurari/masomotr/tutorialData)

The current configuration allows you to track 2 mice in the videos, but you can expand to track more mice with proper configuration.

