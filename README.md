# Markerless Mice Tracking for Social Experiments

This is an implementation of the pipeline to track unmarked mice of similar appearance. The technique and characterization are described in the paper https://www.eneuro.org/content/11/2/ENEURO.0154-22.2023.abstract

## Installation
The code has been run successfully on Windows 7 and 10 with NVIDIA GPUs (Titan XP and RTX2070)
1. Clone this repository using the [Git tool](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository). Alternatively you may simply download the repository as a [zip archive](https://github.com/MaSoMoTr/MaSoMoTr/archive/refs/heads/master.zip) and extract it.
2. [Anaconda](https://www.anaconda.com/distribution/) is highly recommended to install Python 3. Once Anaconda is installed, [open an Anaconda prompt](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda) and navigate to the MaSoMoTr-master folder created in step 1.
3. Use the Anaconda environment configuration file provided in this repository to install DLC and dependencies needed by our algorithm.
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
retrain Mask RCNN and Deeplabcut models on your own data. A video tutorial for both training the Mask RCNN as well as using trained DLC and Mask RCNN models to track mice is available [here](https://youtu.be/cfDd8oyILiY).

There are two ways to launch Streamlit-based GUI:

Run the bat file "LaunchGUI.bat" and browse the paths for models, videos and background image.

OR

Run the command line below 
   ```bash
   streamlit run app_markerless_mice_tracking.py -- --video=path_to_video_dir/  --background=path_to_background_image_dir/ --mrcnn_model=path_to_mrcnn_model_dir/ --dlc_project=path_to_dlc_project_dir/
   ```
If paths are not specified in the above command, streamlit will use default paths located in the project directory for loading videos and saving models. 
- Select the tab "1.Train maDeepLabCut" and follow the instructions for training a DLC model which can be found in [DeepLabCut repository](https://github.com/DeepLabCut/DeepLabCut)
- Select the tab "2.Train Mask R-CNN" and follow the workflow to train Mask-RCNN.


To track mice in new videos, select the tab "3.Tracking Pipeline". 
Follow the GUI to specify the inputs for the pipeline including video .avi, background .jpg, Mask-RCNN model .h5, config.yaml of DeepLabCut project.
* Output of the workflow are two CSV files storing coordinates of snout and tailbase corresponding to two mice: *mouse1.csv* and *mouse2.csv*, and a file *masks.h5* containing masks of the two mice. 


## Pretrained models and video samples
Please visit [this link](https://uofc-my.sharepoint.com/:f:/g/personal/kmurari_ucalgary_ca/EjqrWeirbeRKsp5mJgH_on4BuBQ0ooMnpUPXdpU62ACwFg?e=xPdrEU) to download the files as detailed below:
1. Mask-RCNN model **mask_rcnn_mouse_0025.h5**
The model should be saved in the path:  *mrcnn_models/mask_rcnn_mouse_0025.h5*
2. Deeplabcut model **dlc_mice_model.zip**
The zip file must be extracted and saved in the path:  *dlc_models/dlc_mice_tracking*
3. Sample training and validation data used in the video tutorial above is in the directory **/tutorialData/**
4. Mouse Tracking dataset: directory **MTdataset/** has the 12 videos used in the paper

The current configuration allows you to track 2 mice in the videos, but you can expand to track more mice with proper configuration.

