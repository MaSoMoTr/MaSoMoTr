# streamlit run app_markerless_mice_tracking.py -- --video=/path/to/video_dir/ --background=/path/to/background_dir/ --mrcnn_model=/path/to/model_dir/
import argparse
from skimage.color import gray2rgb
from skimage.util import img_as_float
import skimage.io
import skimage
from PIL import Image, ImageDraw
import multiprocessing
from mouse.utils import dlc_snout_tailbase, deeplabcut_detection_multi_without_refine
from mouse.utils import check_mrcnn_model_path, tracking_inference_h5, mask_based_detection_h5, ensemble_features_multi_h5
from mouse.utils import video2frames
from mouse.utils import mouse_mrcnn_segmentation_h5, background_subtraction_parallel_h5, foreground_detection
import deeplabcut as dlc
import shutil
import ntpath
import glob
import numpy as np
import pandas as pd
import streamlit as st
import deepdish as dd
import os
import sys
import random



ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from imgaug import augmenters as iaa
from mouse.utils import video2frames, split_train_val, create_dataset,  background_subtraction_parallel
from mouse.mouse import MouseConfig, MouseDataset
import multiprocessing



def background_file_selector(folder_path):
    filenames = glob.glob(folder_path+'/*.jpg')
    list_file = [ntpath.basename(file) for file in filenames]
    selected_filename = st.selectbox('Select a background file', list_file)
    file_index = list_file.index(selected_filename)
    return filenames[file_index]


# -----------select a video----------------------------


def video_file_selector(folder_path):
    filenames = glob.glob(folder_path+'/*.avi')
    list_file = [ntpath.basename(file) for file in filenames]
    selected_filename = st.selectbox(
        'Select a video in avi format for tracking', list_file)
    file_index = list_file.index(selected_filename)
    return filenames[file_index]


def file_selector_location(folder_path, location, title, format='avi'):
    filenames = glob.glob(folder_path + '/*.' + format)
    list_file = [ntpath.basename(file) for file in filenames]
    selected_filename = location.selectbox(
        title, list_file)
    file_index = list_file.index(selected_filename)
    return filenames[file_index]

# ---------select MaskRCNN model----------------------

def mrcnn_model_selector(folder_path, last_train=None):
    filenames = glob.glob(folder_path+'/*.h5')


    list_model = [ntpath.basename(file) for file in filenames]

    if last_train !=None:
        filenames.append(last_train)
        list_model.append('Last train')

    selected_filename = st.selectbox('STEP 2: Select an Mask R-CNN model', list_model)
    file_index = list_model.index(selected_filename)
    return filenames[file_index]

def mrcnn_folder_selector(location):
    list_folders = []
    for it in os.scandir(ROOT_DIR + '\mrcnn_models'):
        if it.is_dir():

            list_folders.append(it.path)

    list_folder_names = [os.path.basename(path) for path in list_folders]

    selected_folder = location.selectbox(
        'STEP 2: Select Mask R-CNN folder', list_folder_names)


    return selected_folder


def mrcnn_h5_selector(mrcnn_folder, location):
    filenames = glob.glob(mrcnn_folder+'\*.h5')

    list_file = [ntpath.basename(file) for file in filenames]
    selected_filename = location.selectbox(
        'Select Mask R-CNN model', list_file)

    file_index = list_file.index(selected_filename)
    return filenames[file_index]

# ----------select dlc result -----------------------
@st.cache()
def dlc_result_selector(folder_path):
    filenames = glob.glob(folder_path+'/*.h5')

    list_file = [ntpath.basename(file) for file in filenames]
    selected_filename = st.selectbox('Pick DLC result', list_file)

    file_index = list_file.index(selected_filename)
    return filenames[file_index]


def dlc_config_selector(dlc_project, location):
    filenames = glob.glob(dlc_project+'\*.yaml')

    list_file = [ntpath.basename(file) for file in filenames]
    selected_filename = location.selectbox(
        'Select DeepLabCut config file', list_file)

    file_index = list_file.index(selected_filename)
    return filenames[file_index]

def dlc_project_selector(location):

    list_folders = os.listdir(ROOT_DIR + '\dlc_models')


    selected_folder = location.selectbox(
        'STEP 3: Select DeepLabCut project', list_folders)


    return selected_folder

# -----------------------------------------------------------------------------


def draw_points_on_img(img, point_ver, point_hor, color='red', intensity=1, radius=5):

    rr, cc = skimage.draw.circle(point_ver, point_hor, radius, img.shape)
    if color == 'red':
        img[rr, cc, 0] = intensity
        img[rr, cc, 1] = 0
        img[rr, cc, 2] = 0
    elif color == 'green':
        img[rr, cc, 0] = 0
        img[rr, cc, 1] = intensity
        img[rr, cc, 2] = 0
    elif color == 'blue':
        img[rr, cc, 0] = 0
        img[rr, cc, 1] = 0
        img[rr, cc, 2] = intensity
    elif color == 'yellow':
        img[rr, cc, 0] = intensity
        img[rr, cc, 1] = intensity
        img[rr, cc, 2] = 0

    return img

# ----------------------------------------------------------


#@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_tracking_results(path):
    df_mouse1_ensemble = pd.read_csv(path + '/mouse1.csv')
    df_mouse2_ensemble = pd.read_csv(path + '/mouse2.csv')

    return df_mouse1_ensemble, df_mouse2_ensemble


# ---------------------------------------------------------
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def initial_swap_status(frames_dir):
    df_swap = pd.DataFrame(
        {'swap': np.zeros((len(os.listdir(frames_dir)),), dtype=bool)})
    return df_swap


def main():

    st.title('Markerless Mice Tracking')

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run tracking pipeline.')
    parser.add_argument("--video", required=False,
                        default='videos',
                        metavar="/path/to/video_dir/",
                        help="Path to videos in avi")

    parser.add_argument("--video_train", required=False,
                        default='videos',
                        metavar="/path/to/video_dir/",
                        help="Path to videos in avi")

    parser.add_argument("--background", required=False,
                        default='videos',
                        metavar="/path/to/bg_dir/",
                        help="Path to  background in jpg")

    parser.add_argument('--mrcnn_model', required=False,
                        metavar="/path/to/mrcnn_dir/",
                        default='mrcnn_models',
                        help='Path to mask rcnn model .h5')

    parser.add_argument('--dlc_project', required=False,
                        metavar="/path/to/dlc_project_dir/",
                        default='dlc_models/dlc_mice_model',
                        help='Path to deeplabcut project containing config.yaml')

 

    args = parser.parse_args()



    tab  = st.sidebar.selectbox('Select tab', ('1.Train maDeepLabCut','2.Train Mask R-CNN', '3.Tracking Pipeline', '4.Validate Identities', '5.Validate Keypoints'))

    #-------------------Mask RCNN -------------------

    if tab=="1.Train maDeepLabCut":
        st.subheader('Train maDeepLabCut Model')

        st.write('Create a new maDeepLabCut project in directory:  ', ROOT_DIR +'\dlc_models')
        click_dlc = st .button('Launch DeepLabCut GUI')
        if click_dlc:
            os.system('python -m deeplabcut')

    #-------------------DLC-----------------------------
    elif tab=="2.Train Mask R-CNN":
        st.subheader('Train Mask R-CNN Model')

        # select video for training
        video_column_1, video_column_2, video_column_3 = st.columns(3)  
        train_video_dir = file_selector_location(folder_path=args.video_train, location=video_column_1, title="Select video:", format='avi')
        background_filename = file_selector_location(folder_path=args.background, location=video_column_2, title="Select image:", format='jpg')

        # Extracting frames from the video
        train_frames_dir = os.path.join(os.path.splitext(train_video_dir)[0], 'images')
        click_extract = video_column_3.button('STEP 1: Extract to frames')
        if click_extract:
            video2frames(train_video_dir)

        # Select background image
        BG_column_1, BG_column_2 = st.columns(2) 
        

        click_image = BG_column_1.checkbox('Background substraction settings')
        if click_image:

            hybrid_loc1, hybrid_loc2, hybrid_loc3 = st.columns(3)

            morphology_disk_radius = hybrid_loc1.number_input('Morphology radius', min_value=1, value=7)
            min_blob = hybrid_loc2.number_input('Min blob', min_value=1, value=1000)
            cpus = hybrid_loc3.number_input('#CPUs', min_value=1, max_value=multiprocessing.cpu_count(
            ), value=max(1, int(multiprocessing.cpu_count()/2-2)))

            display_loc1, display_loc2 = st.columns(2)
                         
            
            frame_num = 0
            select_frame = os.path.join(train_frames_dir, str(frame_num)+'.jpg')
            example = img_as_float(skimage.io.imread(select_frame))

            
            if example.ndim == 2:
                example = gray2rgb(example)


            display_loc1.image(example, caption='Frame '+str(frame_num))

            mask = foreground_detection(select_frame,background_filename,morphology_disk_radius, min_blob)

            rr, cc = skimage.draw.disk((int(mask.shape[0]/2), int(mask.shape[1]/2)), np.sqrt(min_blob/np.pi))
            mask[rr, cc, 0:2] = 255

            display_loc2.image(mask, caption='Morphological image (Objects less than yellow min blob are ignored)')

        else:
            morphology_disk_radius = 7
            min_blob = 1000
            cpus=6


        # get image for annotation
        dataset_column_1, dataset_column_2 = st.columns(2)   
        manual_num = dataset_column_1.number_input('Manual dataset (Number of images)', value=20)
        automatic_num = dataset_column_2.number_input('Automatic dataset (Number of images)', value=0)

        label_column_1, label_column_2, label_column_3 = st.columns(3)
        select_images = label_column_2.button('STEP 2: Generate dataset')

        if select_images:
            print('Selecting images for annotation for Mask R-CNN')
            components = background_subtraction_parallel(train_frames_dir, background_filename,  min_blob, morphology_disk_radius, num_processors=cpus) 

            create_dataset(train_frames_dir,background_filename, morphology_disk_radius, min_blob, components, manual_num=manual_num, automatic_num=automatic_num)   # increase dataset by increasing num_annotations 

        dataset_dir = os.path.join(os.path.dirname(train_frames_dir), 'dataset')

        # Label images

        click_annotation = label_column_3.button('STEP 3: Annotate and Split')
        split_ratio = label_column_1.number_input('Train size', value=0.8)


        if click_annotation:
            os.system('labelme')

        #if click_split:
            split_train_val(dataset_dir, frac_split_train=0.8)

        # -----------train the model--------------
        train_column_1, train_column_2, train_column_3 = st.columns(3) 
        init_with  = train_column_1.selectbox('Initial weights?', ('coco', 'imagenet', 'last (most recent trained model)'))
        epochs = train_column_2.number_input('Epochs', value=25)

        click_train = train_column_3.button('STEP 4: Train Mask R-CNN')


        
        if click_train:
            
            # Directory to save logs and trained model
            MODEL_DIR = os.path.join(ROOT_DIR, "mrcnn_models")

            train_size = len([f for f in os.listdir(os.path.join(dataset_dir, 'train')) if f.endswith('.jpg')])
            val_size = len([f for f in os.listdir(os.path.join(dataset_dir, 'val')) if f.endswith('.jpg')])

            config = MouseConfig()
            config.STEPS_PER_EPOCH = int(train_size // config.IMAGES_PER_GPU)
            config.VALIDATION_STEPS = int(val_size // config.IMAGES_PER_GPU)
            config.display()

            # Create model in training mode
            model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=MODEL_DIR)


            # Training dataset.
            dataset_train = MouseDataset()
            dataset_train.load_mouse(dataset_dir, "train")
            dataset_train.prepare()

            # Validation dataset
            dataset_val = MouseDataset()
            dataset_val.load_mouse(dataset_dir, "val")
            dataset_val.prepare()


            # Local path to trained weights file
            COCO_MODEL_PATH = os.path.join(ROOT_DIR,"mrcnn_models", "mask_rcnn_coco.h5")
            # Download COCO trained weights from Releases if needed
            if not os.path.exists(COCO_MODEL_PATH):
                utils.download_trained_weights(COCO_MODEL_PATH)

            # Create model in training mode
            model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

            if init_with == "imagenet":
                model.load_weights(model.get_imagenet_weights(), by_name=True)
            elif init_with == "coco":
                # Load weights trained on MS COCO, but skip layers that
                # are different due to the different number of classes
                # See README for instructions to download the COCO weights
                model.load_weights(COCO_MODEL_PATH, by_name=True,
                                exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                            "mrcnn_bbox", "mrcnn_mask"])
            elif init_with == "last (most recent trained model)":
                # Load the last model you trained and continue training
                model.load_weights(model.find_last(), by_name=True)


            # Image augmentation
            augmentation = iaa.SomeOf((0, 2), [iaa.Fliplr(0.5),
                                            iaa.Flipud(0.5),
                                            iaa.OneOf([iaa.Affine(rotate=90),
                                                        iaa.Affine(rotate=180),
                                                        iaa.Affine(rotate=270)]),
                                            iaa.Multiply((0.8, 1.5)),
                                            iaa.GaussianBlur(sigma=(0.0, 5.0))])

            print("Train network heads")
            model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=epochs,
            augmentation=augmentation,
            layers='heads')


    

    elif tab=="3.Tracking Pipeline":
        # select video for training
        track_left_column, track_right_column = st.columns(2)  
        track_video_dir = file_selector_location(folder_path=args.video, location=track_left_column, title="Select video:", format='avi')

        click_extract = track_right_column.button('Extract to frames')
        if click_extract:
            video2frames(track_video_dir)

        # Select background image

        st.subheader('Tracking settings')

        image_left_column, image_right_column = st.columns(2) 
        background_filename = file_selector_location(folder_path=args.background, location=image_left_column, title="STEP 1: Select background image:", format='jpg')
        BG = img_as_float(skimage.io.imread(background_filename))

        click_image = image_right_column.checkbox('Display background')

        st.session_state.x_offset = 50
        st.session_state.y_offset = 50


        if click_image:
            
            if BG.ndim == 2:
                BG = gray2rgb(BG)

            left_column_floor, right_column_floor = st.columns(2)
            st.session_state.x_offset  = left_column_floor.number_input(
                'Floor_offset_x', min_value=0, max_value=BG.shape[1], value=50)
            st.session_state.y_offset  = right_column_floor.number_input(
                'Floor_offset_y', min_value=0, max_value=BG.shape[0], value=50)

            rr, cc = skimage.draw.rectangle(start=(st.session_state.y_offset, st.session_state.x_offset), end=(
                BG.shape[0]-st.session_state.y_offset, BG.shape[1]-st.session_state.x_offset), shape=(BG.shape[0], BG.shape[1]))
            BG[rr, cc, 0:2] = 1
            right_column_floor.image(BG, caption='Background')


       
        # --------------------------------------------------------------------
        mrcnn_loc1, mrcnn_loc2 = st.columns(2)
        mrcnn_folder = mrcnn_folder_selector(mrcnn_loc1)

        MRCNN_MODEL_PATH = mrcnn_h5_selector(ROOT_DIR + '\mrcnn_models\\' +mrcnn_folder, mrcnn_loc2)

        # --------------------------------------------------------------------
        dlc_loc1, dlc_loc2 = st.columns(2)
        
        dlc_project = dlc_project_selector(dlc_loc1)

        dlc_config = dlc_config_selector(ROOT_DIR + '\dlc_models\\' + dlc_project, dlc_loc2)

        loc1, loc2, loc3, loc4 = st.columns(4)
        shuffle = loc1.number_input('Shuffle', value=1)
        trainingsetindex = loc2.number_input('Training set index', value=0)
        track_method = loc3.selectbox(
            'Track method', ('ellipse', 'skeleton', 'box'))
        videotype = loc4.selectbox('Video type', ('avi', 'mp4'))

        if (background_filename != None) & (track_video_dir != None) & (MRCNN_MODEL_PATH != None) & (dlc_config != None):
            frames_dir = os.path.join(os.path.splitext(track_video_dir)[0], "images")

            # -------------background subtraction-----------------

            hybrid_loc1, hybrid_loc2, hybrid_loc3, hybrid_loc4 = st.columns(4)

            check = hybrid_loc1.checkbox('Hybrid approach')

            if check:
                # Select number of cpus
                morphology_disk_radius = hybrid_loc2.number_input('Morphology radius', min_value=1, value=9)
                min_blob = hybrid_loc3.number_input('Min blob', min_value=1, value=1000)
                cpus = hybrid_loc4.number_input('#CPUs', min_value=1, max_value=multiprocessing.cpu_count(
                ), value=max(1, int(multiprocessing.cpu_count()/2-2)))

                display_loc1, display_loc2, display_loc3 = st.columns(3)
                
                
                display_check = display_loc1.checkbox('Validate settings')
                if display_check:
                   
                    track_frames_dir = os.path.join(os.path.splitext(track_video_dir)[0], "images")
                    frame_num = 0
                    select_frame = os.path.join(track_frames_dir, str(frame_num)+'.jpg')
                    example = img_as_float(skimage.io.imread(select_frame))

                    
                    if example.ndim == 2:
                        example = gray2rgb(example)


                    display_loc2.image(example, caption='Frame '+str(frame_num))

                    mask = foreground_detection(select_frame,background_filename,morphology_disk_radius, min_blob)

                    rr, cc = skimage.draw.disk((int(mask.shape[0]/2), int(mask.shape[1]/2)), np.sqrt(min_blob/np.pi))
                    mask[rr, cc, 0:2] = 255

                    display_loc3.image(mask, caption='Morphological image (Objects less than yellow min blob are ignored)')

            # ----------select dlc result -----------------------

            click2 = st.button('Track the animals')
            if click2:
                print('Tracking animals!')

                if check:
                    video_dict, components = background_subtraction_parallel_h5(
                        frames_dir, background_filename, morphology_disk_radius, min_blob, num_processors=cpus)

                else:
                    frames_dir = os.path.join(os.path.splitext(
                        track_video_dir)[0], "images")
                    n_frames = len(os.listdir(frames_dir))
                    components = np.zeros((n_frames,))

                    video_dict = {}

                    # Directory to save logs and trained model

                MODEL_DIR = os.path.join(ROOT_DIR, "mrcnn_models")

                # Directory to load weights of a model. If MRCNN_MODEL_PATH=None, the latest weights in MODEL_DIR will be loaded
                if MRCNN_MODEL_PATH != None:
                    if not os.path.exists(MRCNN_MODEL_PATH):
                        print(
                            "Please read mrcnn_models/README.md to download our trained model")

                    else:
                        video_dict, components = mouse_mrcnn_segmentation_h5(video_dict,
                            components, frames_dir, background_filename, model_dir=MODEL_DIR, model_path=MRCNN_MODEL_PATH)

                elif MRCNN_MODEL_PATH == None:
                    if check_mrcnn_model_path(MODEL_DIR):
                        print("The latest trained model will be loaded")
                        video_dict, components = mouse_mrcnn_segmentation_h5(video_dict,
                            components, frames_dir, background_filename, model_dir=MODEL_DIR, model_path=MRCNN_MODEL_PATH)
                    else:
                        print(
                            "Could not find model directory under {}".format(MODEL_DIR))
                        print(
                            "Please follow the pipeline in mrcnn_training.ipynb to train your own model, then run this step again.")


        # -----------------tracking inference---------------------------

                video_tracking_dict=tracking_inference_h5(video_dict, frames_dir, components, (BG.shape[0], BG.shape[1]))

        # ---------------deeplabcut detection-------------------------
                dlc_result = deeplabcut_detection_multi_without_refine(config_path=dlc_config,
                                                                       video_path=track_video_dir, shuffle=shuffle,
                                                                       trainingsetindex=trainingsetindex,
                                                                       track_method=track_method, videotype=videotype)

                df_mouse1_dlc, df_mouse2_dlc = dlc_snout_tailbase(dlc_result)



        # ---------------mask-based detection--------------------------

                floor = [[st.session_state.y_offset+1, st.session_state.x_offset+1],
                         [BG.shape[0]-st.session_state.y_offset, BG.shape[1]-st.session_state.x_offset]]

                df_mouse1_md, df_mouse2_md = mask_based_detection_h5(
                    video_tracking_dict, frames_dir, components, floor=floor, image_shape=(BG.shape[0], BG.shape[1]))

 

                # -----------------ensemble--------------------

                df_mouse1_ensemble, df_mouse2_ensemble = ensemble_features_multi_h5(
                    df_mouse1_md, df_mouse2_md, df_mouse1_dlc, df_mouse2_dlc, components, video_tracking_dict, (BG.shape[0], BG.shape[1]), frames_dir)

                print('finish tracking: ', track_video_dir)

        # ------------Validating results---------------------------------------

    elif tab=="4.Validate Identities":
        track_left_column, track_right_column = st.columns(2)  

        track_video_dir = file_selector_location(folder_path=args.video, location=track_left_column, title="Select video:", format='avi')

        # Extracting frames from the video
        frames_dir = os.path.join(os.path.splitext(track_video_dir)[0], "images")

        df_swap = initial_swap_status(frames_dir)  # -----------------

        if os.path.exists(frames_dir):

            n_frames = len(os.listdir(frames_dir))
            try:
                df_mouse1_ensemble, df_mouse2_ensemble = load_tracking_results(
                    os.path.dirname(frames_dir))

                video_correct_dict = dd.io.load(os.path.join(os.path.dirname(frames_dir), 'masks.h5'))


                frame_index_main = st.slider(
                    'frame slider', 0, n_frames-1, 0)

                frame_index = st.number_input(
                    'frame number', min_value=0, max_value=n_frames-1, value=frame_index_main)
                frame_file = str(frame_index) + '.jpg'
                frame_path = os.path.join(frames_dir, frame_file)

                image = img_as_float(skimage.io.imread(frame_path))

                if image.ndim == 2:
                    image = gray2rgb(image)

                # mouse 1
                center1_x = (
                    df_mouse1_ensemble.iloc[frame_index, 0] + df_mouse1_ensemble.iloc[frame_index, 2])/2
                center1_y = (
                    df_mouse1_ensemble.iloc[frame_index, 1] + df_mouse1_ensemble.iloc[frame_index, 3])/2

                print('frame ' + str(frame_index) + ':', center1_x, center1_y)

                # mouse 2

                center2_x = (
                    df_mouse2_ensemble.iloc[frame_index, 0] + df_mouse2_ensemble.iloc[frame_index, 2])/2
                center2_y = (
                    df_mouse2_ensemble.iloc[frame_index, 1] + df_mouse2_ensemble.iloc[frame_index, 3])/2

                if ~df_swap.loc[frame_index, 'swap']:
                    image = draw_points_on_img(
                        image, center1_y, center1_x, color='red', intensity=1)

                    image = draw_points_on_img(
                        image, center2_y, center2_x, color='green', intensity=1)
                else:
                    image = draw_points_on_img(
                        image, center1_y, center1_x, color='green', intensity=1)

                    image = draw_points_on_img(
                        image, center2_y, center2_x, color='red', intensity=1)

                st.image(image, caption='frame: ' + str(frame_index))

                validation_bt1, validation_bt2 = st.columns(2)
                click6 = validation_bt1.button('Switch identities')

                if click6:
                    df_swap.loc[frame_index:,
                                'swap'] = ~ df_swap.loc[frame_index:, 'swap']

                click7 = validation_bt2.button('Save correction')
                if click7:

                    # correct coordinates
                    df_mouse1_ensemble_temp = df_mouse1_ensemble.copy()
                    df_mouse1_ensemble[df_swap['swap']
                                    ] = df_mouse2_ensemble[df_swap['swap']]
                    df_mouse2_ensemble[df_swap['swap']
                                    ] = df_mouse1_ensemble_temp[df_swap['swap']]

                    df_mouse1_ensemble.to_csv(os.path.dirname(
                        frames_dir) + '/mouse1.csv', index=False)
                    df_mouse2_ensemble.to_csv(os.path.dirname(
                        frames_dir) + '/mouse2.csv', index=False)

                    # correct masks

                    swap_frames_list = df_swap.index[df_swap['swap'] == True].tolist()

                    for i in swap_frames_list:

                        frame_current_dict = video_correct_dict[str(i)]

                        mouse_temp = frame_current_dict['mouse1'].copy()
                        frame_current_dict['mouse1'] = frame_current_dict['mouse2'] 
                        frame_current_dict['mouse2'] = mouse_temp
                        video_correct_dict[str(i)] = frame_current_dict

                        dd.io.save(os.path.join(os.path.dirname(frames_dir), 'masks.h5'), video_correct_dict, compression=True)

                    # reset swap
                    df_swap['swap'] = False
            except:
                st.write('There is no tracking results for the selected video!')

    elif tab=="5.Validate Keypoints":
        track_left_column, track_right_column = st.columns(2)  

        track_video_dir = file_selector_location(folder_path=args.video, location=track_left_column, title="Select video:", format='avi')

        frames_dir = os.path.join(os.path.splitext(
            track_video_dir)[0], "images")

        if os.path.exists(frames_dir):
            n_frames = len(os.listdir(frames_dir))
            try:
                df_mouse1_ensemble, df_mouse2_ensemble = load_tracking_results(
                    os.path.dirname(frames_dir))

                frame_index2 = st.slider('frame number', 0, n_frames-1, 0)

                frame_file = str(frame_index2) + '.jpg'
                frame_path = os.path.join(frames_dir, frame_file)

                image = img_as_float(skimage.io.imread(frame_path))

                if image.ndim == 2:
                    image = gray2rgb(image)

                # mouse 1

                image = draw_points_on_img(
                    image, df_mouse1_ensemble.iloc[frame_index2, 1], df_mouse1_ensemble.iloc[frame_index2, 0], color='red', intensity=1)
                image = draw_points_on_img(
                    image, df_mouse1_ensemble.iloc[frame_index2, 3], df_mouse1_ensemble.iloc[frame_index2, 2], color='red', intensity=0.7)

                # mouse 2

                image = draw_points_on_img(
                    image, df_mouse2_ensemble.iloc[frame_index2, 1], df_mouse2_ensemble.iloc[frame_index2, 0], color='green', intensity=1)
                image = draw_points_on_img(
                    image, df_mouse2_ensemble.iloc[frame_index2, 3], df_mouse2_ensemble.iloc[frame_index2, 2], color='green', intensity=0.7)

                st.image(image, caption='frame: ' + str(frame_index2))

                st.write(df_mouse1_ensemble.iloc[frame_index2, 1], df_mouse1_ensemble.iloc[frame_index2, 0])
                st.write(df_mouse1_ensemble.iloc[frame_index2, 3], df_mouse1_ensemble.iloc[frame_index2, 2])
                st.write(df_mouse2_ensemble.iloc[frame_index2, 1], df_mouse2_ensemble.iloc[frame_index2, 0])
                st.write(df_mouse2_ensemble.iloc[frame_index2, 3], df_mouse2_ensemble.iloc[frame_index2, 2])
            except:
                st.write('There is no tracking results for the selected video!')


if __name__ == "__main__":
    main()
