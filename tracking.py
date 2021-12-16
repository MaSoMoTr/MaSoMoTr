# python tracking.py --video=H:\Projects\Tracking\thesis\github\markerless-mice-tracking\videos\test1 --background=H:\Projects\Tracking\thesis\github\markerless-mice-tracking\videos\BG.jpg --mrcnn_model=H:\Projects\Tracking\thesis\github\markerless-mice-tracking\mrcnn_models\mask_rcnn_mouse_0025.h5 --dlc_config=path_to/config.yaml


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


# ---------------------------------------------------------

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run tracking pipeline.')
    parser.add_argument("--video", required=True,
                        metavar="/path/to/video_dir/",
                        help="Path to video_dir containing subfolder images")

    parser.add_argument("--background", required=True,
                        metavar="/path/to/bg/",
                        help="Path to  background in jpg")

    parser.add_argument('--mrcnn_model', required=True,
                        metavar="/path/to/mrcnn_model/",
                        help='Path to mask rcnn model .h5')

    parser.add_argument('--floor_offset_x', required=False,
                        metavar="integer number",
                        default=50,
                        help='offset x')

    parser.add_argument('--floor_offset_y', required=False,
                        metavar="integer number",
                        default=50,
                        help='offset y')

    parser.add_argument('--hybrid', required=False,
                        metavar="Boolean",
                        default=False,
                        help='set the algorithm in hybrid mode')

    # --------------deeplabcut args------------------------
    parser.add_argument("--dlc_config", required=True,
                        metavar="/path/to/config.yaml/",
                        help="Path to deeplabcut config file")

    parser.add_argument('--dlc_shuffle', required=False,
                        metavar="integer number",
                        default=1,
                        help='deeplabcut shuffle')

    parser.add_argument('--dlc_trainingsetindex', required=False,
                        metavar="integer number",
                        default=0,
                        help='deeplabcut trainingsetindex')

    parser.add_argument("--dlc_track_method", required=False,
                        metavar="skeleton or box or ellipse",
                        default='ellipse',
                        help="set track method")

    parser.add_argument("--videotype", required=False,
                        metavar=".avi or .mp4",
                        default='avi',
                        help="video type")

    args = parser.parse_args()


    #----run DLC------------
    video_path = args.video + '.' + args.videotype
    dlc_result = deeplabcut_detection_multi_without_refine(config_path=args.dlc_config,
                                                                       video_path=video_path, 
                                                                       shuffle=int(args.dlc_shuffle),
                                                                       trainingsetindex=int(args.dlc_trainingsetindex),
                                                                       track_method=args.dlc_track_method, 
                                                                       videotype=args.videotype)

    df_mouse1_dlc, df_mouse2_dlc = dlc_snout_tailbase(dlc_result)


    # ----------------------------------------------------
    frames_dir = os.path.join(args.video, "images")

    if args.hybrid:
        cpus = max(1, int(multiprocessing.cpu_count()/2))

        video_dict, components = background_subtraction_parallel_h5(
                        frames_dir, args.background, num_processors=cpus)

    else:

        n_frames = len(os.listdir(frames_dir))
        components = np.zeros((n_frames,))
        video_dict = {}

    # segment

    video_dict, components = mouse_mrcnn_segmentation_h5(video_dict,
                                                         components, 
                                                         frames_dir, 
                                                         args.background, 
                                                         model_dir=os.path.dirname(args.mrcnn_model), 
                                                         model_path=args.mrcnn_model)

    # -----------------tracking inference---------------------------

    BG =  img_as_float(skimage.io.imread(args.background))

    video_tracking_dict=tracking_inference_h5(video_dict, frames_dir, components, (BG.shape[0], BG.shape[1]))


    # -------------mask-based detection----------------------------
    y_offset = int(args.floor_offset_y)
    x_offset = int(args.floor_offset_x)
    floor = [[y_offset+1, x_offset+1],
             [BG.shape[0]-y_offset, BG.shape[1]-x_offset]]
    df_mouse1_md, df_mouse2_md = mask_based_detection_h5(
                    video_tracking_dict, frames_dir, components, floor=floor, image_shape=(BG.shape[0], BG.shape[1]))


    #------------ensemble------------------------------
    df_mouse1_ensemble, df_mouse2_ensemble = ensemble_features_multi_h5(df_mouse1_md, df_mouse2_md, df_mouse1_dlc, df_mouse2_dlc, components, video_tracking_dict, (BG.shape[0], BG.shape[1]), frames_dir)



    print('finish tracking: ', video_path)


if __name__ == "__main__":
    main()
