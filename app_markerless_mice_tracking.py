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
from mouse.utils import mouse_mrcnn_segmentation_h5, background_subtraction_parallel_h5
import shutil
import ntpath
import glob
import numpy as np
import pandas as pd
import streamlit as st
import deepdish as dd
import os
import sys


ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

#--------------------------------------

# Root directory of the project
#ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from imgaug import augmenters as iaa
from mouse.utils import video2frames, background_subtraction, split_train_val, create_dataset,  background_subtraction_parallel
from mouse.mouse import MouseConfig, MouseDataset
import multiprocessing





# Root directory of the project
# ----------select background--------------------------

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


#@st.cache()
def mrcnn_model_selector(folder_path, last_train=None):
    filenames = glob.glob(folder_path+'/*.h5')


    list_model = [ntpath.basename(file) for file in filenames]

    if last_train !=None:
        filenames.append(last_train)
        list_model.append('Last train')

    selected_filename = st.selectbox('STEP 2: Select an Mask-RCNN model', list_model)
    file_index = list_model.index(selected_filename)
    return filenames[file_index]

# ----------select dlc result -----------------------


@st.cache()
def dlc_result_selector(folder_path):
    filenames = glob.glob(folder_path+'/*.h5')

    list_file = [ntpath.basename(file) for file in filenames]
    selected_filename = st.selectbox('Pick DLC result', list_file)

    file_index = list_file.index(selected_filename)
    return filenames[file_index]


def dlc_config_selector(dlc_project):
    filenames = glob.glob(dlc_project+'/*.yaml')

    list_file = [ntpath.basename(file) for file in filenames]
    selected_filename = st.selectbox(
        'STEP 3: Select DeepLabCut config file', list_file)

    file_index = list_file.index(selected_filename)
    return filenames[file_index]

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


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
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

    parser.add_argument('--mrcnn_dataset', required=False,
                        metavar="/path/to/mrcnn_dataset/",
                        default='datasets/mrcnn',
                        help='Path to mrcnn dataset')


    args = parser.parse_args()

    #st.session_state.just_train=None
    if 'just_train' not in st.session_state:
        st.session_state.just_train = None
    #-------------------------------------------------
    tab  = st.sidebar.selectbox('Select tab', ('Train Mask-RCNN', 'Train maDLC', 'Tracking Pipeline', 'Validate Identities', 'Validate Keypoints'))

    #-------------------Mask RCNN --------------------
    #training_maskrcnn = st.sidebar.checkbox('Train MaskRCNN')

    if tab=="Train Mask-RCNN":
        st.subheader('Train Mask-RCNN Model')

        # select video for training
        video_left_column, video_right_column = st.columns(2)  
        train_video_dir = file_selector_location(folder_path=args.video_train, location=video_left_column, title="Select video:", format='avi')

        # Extracting frames from the video
        train_frames_dir = os.path.join(os.path.splitext(train_video_dir)[0], 'images')
        click_extract = video_right_column.button('STEP 1: extract to frames')
        if click_extract:
            video2frames(train_video_dir)

        # Select background image
        image_left_column, image_right_column = st.columns(2) 
        background_filename = file_selector_location(folder_path=args.background, location=image_left_column, title="Select image:", format='jpg')

        click_image = image_right_column.checkbox('Display background')
        if click_image:
            BG = img_as_float(skimage.io.imread(background_filename))
            if BG.ndim == 2:
                BG = gray2rgb(BG)

            st.image(BG, caption='Background')


        # get image for annotation
        left_column, right_column = st.columns(2)   
        num_img = left_column.number_input('Dataset size', value=20)
        select_images = right_column.button('STEP 2: select images for annotation')

        if select_images:
            components = background_subtraction_parallel(train_frames_dir, background_filename, num_processors=2) 

            create_dataset(train_frames_dir,components, num_annotations=num_img)   # increase dataset by increasing num_annotations 

        dataset_dir = os.path.join(os.path.dirname(train_frames_dir), 'dataset')

        # Label images
        label_left_column, label_right_column = st.columns(2)
        click_annotation = label_left_column.button('STEP 3: annotation')
        click_split = label_right_column.button('STEP 4: split dataset')

        if click_annotation:
            os.system('labelme')

        if click_split:
            split_train_val(dataset_dir, frac_split_train=0.8)


        

        # -----------train the model--------------
        train_left_column, train_right_column = st.columns(2) 
        init_with  = train_left_column.selectbox('Initial weights?', ('coco', 'imagenet', 'last'))
        click_train = train_right_column .button('STEP 5: train Mask RCNN')


        
        if click_train:
            # Mask-RCNN model
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

            #dataset_dir= 'datasets/mrcnn'

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
            elif init_with == "last":
                # Load the last model you trained and continue training
                model.load_weights(model.find_last(), by_name=True)


            # Image augmentation
            # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
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
            epochs=10,
            augmentation=augmentation,
            layers='heads')

            st.session_state.just_train = model.find_last()
            #st.session_state.just_train = model.find_last()
            print('call 1: ' + st.session_state.just_train)
    #-------------------DLC-----------------------------
    elif tab=="Train maDLC":
        st.subheader('Train maDeepLabCut Model')
        os.system('python -m deeplabcut')

    

    elif tab=="Tracking Pipeline":
        # select video for training
        track_left_column, track_right_column = st.columns(2)  
        track_video_dir = file_selector_location(folder_path=args.video_train, location=track_left_column, title="Select video:", format='avi')

        # Extracting frames from the video
        track_frames_dir = os.path.join(os.path.splitext(track_video_dir)[0], 'images')
        click_extract = track_right_column.button('Extract to frames')
        if click_extract:
            video2frames(track_video_dir)

        #----------------------
        # Select background image

        st.subheader('Tracking settings')

        image_left_column, image_right_column = st.columns(2) 
        background_filename = file_selector_location(folder_path=args.background, location=image_left_column, title="Select background image:", format='jpg')
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

            #st.image(BG, caption='Background')



        #----------------------
        # tracking_video_dir = video_file_selector(folder_path=args.video)
        # click1 = st.button('Extract the video into frames')

        # st.subheader('Tracking settings')

        

        # # Select data and model
        # background_filename = background_file_selector(
        #     folder_path=args.background)

        # BG = img_as_float(skimage.io.imread(background_filename))
        # if BG.ndim == 2:
        #     BG = gray2rgb(BG)

        # left_column_floor, right_column_floor = st.columns(2)

        # # ---------------------------------------------------------------------------

        # x_offset = left_column_floor.number_input(
        #     'Floor_offset_x', min_value=0, max_value=BG.shape[1], value=50)
        # y_offset = right_column_floor.number_input(
        #     'Floor_offset_y', min_value=0, max_value=BG.shape[0], value=50)

        # rr, cc = skimage.draw.rectangle(start=(y_offset, x_offset), end=(
        #     BG.shape[0]-y_offset, BG.shape[1]-x_offset), shape=(BG.shape[0], BG.shape[1]))
        # BG[rr, cc, 0:2] = 1
        # right_column_floor.image(BG, caption='Background')

        # ----------------------------------------------------------------------
        #print(st.session_state.just_train)
        print(st.session_state.just_train)
        MRCNN_MODEL_PATH = mrcnn_model_selector(folder_path=args.mrcnn_model, last_train=st.session_state.just_train)

        # --------------------------------------------------------------------
        dlc_config = dlc_config_selector(dlc_project=args.dlc_project)

        loc1, loc2, loc3, loc4 = st.columns(4)
        shuffle = loc1.number_input('Shuffle', value=1)
        trainingsetindex = loc2.number_input('Training set index', value=0)
        track_method = loc3.selectbox(
            'Track method', ('ellipse', 'skeleton', 'box'))
        videotype = loc4.selectbox('Video type', ('avi', 'mp4'))

        if (background_filename != None) & (track_video_dir != None) & (MRCNN_MODEL_PATH != None) & (dlc_config != None):
            frames_dir = os.path.join(os.path.splitext(
                track_video_dir)[0], "images")
            # ----------- Extracting the video into frames --------
            
            # if click1:
            #     print('Extracting video to frames!')
            #     frames_dir = video2frames(tracking_video_dir)

            # -------------background subtraction-----------------
            left_column_approach, right_column_approach = st.columns(2)
            check = left_column_approach.checkbox('Hybrid approach')

            if check:
                # Select number of cpus
                cpus = right_column_approach.number_input('#CPUs', min_value=1, max_value=multiprocessing.cpu_count(
                ), value=max(1, int(multiprocessing.cpu_count()/2-2)))

            # ----------select dlc result -----------------------

            click2 = st.button('Track the animals')
            if click2:
                print('Tracking animals!')

                if check:
                    video_dict, components = background_subtraction_parallel_h5(
                        frames_dir, background_filename, num_processors=cpus)

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

    #validate1 = st.sidebar.checkbox('Validate Tracking Results')

    elif tab=="Validate Identities":

        frames_dir = os.path.join(os.path.splitext(
            track_video_dir)[0], "images")

        df_swap = initial_swap_status(frames_dir)  # -----------------

        if os.path.exists(frames_dir):

            n_frames = len(os.listdir(frames_dir))
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

    #validate2 = st.sidebar.checkbox('Validate keypoints')
    elif tab=="Validate Keypoints":
        frames_dir = os.path.join(os.path.splitext(
            tracking_video_dir)[0], "images")

        if os.path.exists(frames_dir):
            n_frames = len(os.listdir(frames_dir))
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


if __name__ == "__main__":
    main()
