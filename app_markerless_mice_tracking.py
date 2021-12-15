# streamlit run app_markerless_mice_tracking.py -- --video=/path/to/video_dir/ --background=/path/to/background_dir/ --mrcnn_model=/path/to/model_dir/
import argparse
from skimage.color import gray2rgb
from skimage.util import img_as_float
import skimage.io
import skimage
from PIL import Image, ImageDraw
import multiprocessing
from mouse.utils import dlc_snout_tailbase, behavior_feature_extraction, deeplabcut_detection_multi_without_refine
from mouse.utils import check_mrcnn_model_path, tracking_inference_h5, mask_based_detection_h5, ensemble_features_multi_h5
from mouse.utils import video2frames, background_subtraction_parallel, mouse_mrcnn_segmentation
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

# ---------select MaskRCNN model----------------------


#@st.cache()
def mrcnn_model_selector(folder_path):
    filenames = glob.glob(folder_path+'/*.h5')
    list_file = [ntpath.basename(file) for file in filenames]
    selected_filename = st.selectbox('Select an Mask-RCNN model', list_file)
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


def dlc_config_selector(dlc_project):
    filenames = glob.glob(dlc_project+'/*.yaml')

    list_file = [ntpath.basename(file) for file in filenames]
    selected_filename = st.selectbox(
        'Select DeepLabCut config file', list_file)

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

    tracking = st.sidebar.checkbox('Tracking Pipeline')
    tracking_video_dir = video_file_selector(folder_path=args.video)

    if tracking:
        st.subheader('Tracking settings')

        # Select data and model
        background_filename = background_file_selector(
            folder_path=args.background)

        BG = img_as_float(skimage.io.imread(background_filename))
        if BG.ndim == 2:
            BG = gray2rgb(BG)

        left_column_floor, right_column_floor = st.beta_columns(2)

        # ---------------------------------------------------------------------------

        x_offset = left_column_floor.number_input(
            'Floor_offset_x', min_value=0, max_value=BG.shape[1], value=50)
        y_offset = right_column_floor.number_input(
            'Floor_offset_y', min_value=0, max_value=BG.shape[0], value=50)

        rr, cc = skimage.draw.rectangle(start=(y_offset, x_offset), end=(
            BG.shape[0]-y_offset, BG.shape[1]-x_offset), shape=BG.shape)
        BG[rr, cc, 0:2] = 1
        right_column_floor.image(BG, caption='Background')

        # ----------------------------------------------------------------------
        MRCNN_MODEL_PATH = mrcnn_model_selector(folder_path=args.mrcnn_model)

        # --------------------------------------------------------------------
        dlc_config = dlc_config_selector(dlc_project=args.dlc_project)

        loc1, loc2, loc3, loc4 = st.beta_columns(4)
        shuffle = loc1.number_input('Shuffle', value=1)
        trainingsetindex = loc2.number_input('Training set index', value=0)
        track_method = loc3.selectbox(
            'Track method', ('ellipse', 'skeleton', 'box'))
        videotype = loc4.selectbox('Video type', ('avi', 'mp4'))

        if (background_filename != None) & (tracking_video_dir != None) & (MRCNN_MODEL_PATH != None) & (dlc_config != None):
            frames_dir = os.path.join(os.path.splitext(
                tracking_video_dir)[0], "images")
            # ----------- Extracting the video into frames --------
            click1 = st.sidebar.button('Extract the video into frames')
            if click1:
                frames_dir = video2frames(tracking_video_dir)

            # -------------background subtraction-----------------
            left_column_approach, right_column_approach = st.beta_columns(2)
            check = left_column_approach.checkbox('Hybrid approach')

            if check:
                # Select number of cpus
                cpus = right_column_approach.number_input('#CPUs', min_value=1, max_value=multiprocessing.cpu_count(
                ), value=max(1, int(multiprocessing.cpu_count()/2-2)))

            # ----------select dlc result -----------------------

            click2 = st.sidebar.button('Track the animals')
            if click2:

                if check:
                    video_dict, components = background_subtraction_parallel_h5(
                        frames_dir, background_filename, num_processors=cpus)

                else:
                    frames_dir = os.path.join(os.path.splitext(
                        tracking_video_dir)[0], "images")
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

                video_tracking_dict=tracking_inference_h5(video_dict, frames_dir, components, BG.shape)

        # ---------------deeplabcut detection-------------------------
                dlc_result = deeplabcut_detection_multi_without_refine(config_path=dlc_config,
                                                                       video_path=tracking_video_dir, shuffle=shuffle,
                                                                       trainingsetindex=trainingsetindex,
                                                                       track_method=track_method, videotype=videotype)

                df_mouse1_dlc, df_mouse2_dlc = dlc_snout_tailbase(dlc_result)


        # ---------------mask-based detection--------------------------

                floor = [[y_offset+1, x_offset+1],
                         [BG.shape[0]-y_offset, BG.shape[1]-x_offset]]

                df_mouse1_md, df_mouse2_md = mask_based_detection_h5(
                    video_tracking_dict, frames_dir, components, floor=floor, image_shape=(BG.shape[0], BG.shape[1]))

                # -----------------ensemble--------------------

                 df_mouse1_ensemble, df_mouse2_ensemble = ensemble_features_multi_h5(
                    df_mouse1_md, df_mouse2_md, df_mouse1_dlc, df_mouse2_dlc, components, video_tracking_dict, BG.shape, frames_dir)

        # ------------Validating results---------------------------------------

    validate1 = st.sidebar.checkbox('Validate Tracking Results')

    if validate1:

        frames_dir = os.path.join(os.path.splitext(
            tracking_video_dir)[0], "images")

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

            validation_bt1, validation_bt2 = st.beta_columns(2)
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
                # tracking_dir = os.path.join(
                #     os.path.dirname(frames_dir), 'tracking')
                for i in swap_frames_list:

                    frame_current_dict = video_correct_dict[str(i)]

                    mouse_temp = frame_current_dict['mouse1'].copy()
                    frame_current_dict['mouse1'] = frame_current_dict['mouse2'] 
                    frame_current_dict['mouse2'] = mouse_temp
                    video_correct_dict[str(i)] = frame_current_dict

                    dd.io.save(os.path.join(os.path.dirname(frames_dir), 'masks.h5'), video_correct_dict, compression=True)

                # reset swap
                df_swap['swap'] = False

    validate2 = st.sidebar.checkbox('Validate keypoints')
    if validate2:
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
