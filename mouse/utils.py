import mrcnn.model as modellib
import os
import sys
import cv2
import random
import numpy as np
import pandas as pd
import deeplabcut
import json
import skimage
import skimage.io
from skimage.util import img_as_ubyte, img_as_float
from skimage import morphology, measure, filters
from shutil import copyfile
from skimage.measure import regionprops
from skimage.measure import find_contours
from skimage.morphology import square, dilation
from skimage.color import rgb2gray
from .mouse import MouseDataset
from .mouse import InferenceConfig
from .shape import shapes_to_labels_masks
from multiprocessing import Pool
import deepdish as dd
import math
import numpy as np
import PIL.Image
import PIL.ImageDraw


import shutil
import time
import errno
import ntpath
import glob

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


def onerror(function, path, exc_info):
    # Handle ENOTEMPTY for rmdir
    if (function is os.rmdir
        and issubclass(exc_info[0], OSError)
            and exc_info[1].errno == errno.ENOTEMPTY):
        timeout = 0.001
        while timeout < 2:
            if not os.listdir(path):
                return os.rmdir(path)
            time.sleep(timeout)
            timeout *= 2
    raise


def clean_dir_safe(path):
    if os.path.isdir(path):
        shutil.rmtree(path, onerror=onerror)
    # rmtree didn't fail, but path may still be linked if there is or was
    # a handle that shares delete access. Assume the owner of the handle
    # is watching for changes and will close it ASAP. So retry creating
    # the directory by using a loop with an increasing timeout.
        timeout = 0.001
        while True:
            try:
                return os.mkdir(path)
            except PermissionError as e:
                # Getting access denied (5) when trying to create a file or
                # directory means either the caller lacks access to the
                # parent directory or that a file or directory with that
                # name exists but is in the deleted state. Handle both cases
                # the same way. Otherwise, re-raise the exception for other
                # permission errors, such as a sharing violation (32).
                if e.winerror != 5 or timeout >= 2:
                    raise
                time.sleep(timeout)
                timeout *= 2


def video2frames(video_dir):
    """Convert a video into frames saved in a directory named as the video name.
    Args:
        video_dir: path to the video
    """
    cap = cv2.VideoCapture(video_dir)
    nframes = int(cap.get(7))

    data_dir = os.path.splitext(video_dir)[0]

    frames_dir = os.path.join(data_dir, "images")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(frames_dir):
        os.mkdir(frames_dir)

    for index in range(nframes):
        cap.set(1, index)  # extract a particular frame
        ret, frame = cap.read()
        if ret:
            image = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            img_name = os.path.join(frames_dir, str(index) + ".jpg")

            skimage.io.imsave(img_name, image)
    return frames_dir


def background_subtraction(frames_dir, background_dir):
    """Generate foregrounds corresponding to frames
    Args:
        frames_dir: path to directory containing frames
        background_dir: path to the background image
    Returns:
        components: 1D array of number of blobs in each frame.
    """
    fg_dir = os.path.join(os.path.dirname(frames_dir), 'FG')

    try:
        os.mkdir(fg_dir)
    except FileExistsError:
        shutil.rmtree(fg_dir)
        os.mkdir(fg_dir)

    bg = img_as_float(skimage.io.imread(background_dir))
    if bg.ndim == 3:
        bg = rgb2gray(bg)

    threshold = bg * 0.5

    frames_list = os.listdir(frames_dir)
    components = np.zeros(len(frames_list), dtype=int)

    for frame in range(len(frames_list)):
        im = img_as_float(skimage.io.imread(
            os.path.join(frames_dir, str(frame) + '.jpg')))

        if im.ndim == 3:
            im = rgb2gray(im)

        fg = (bg - im) > threshold
        bw1 = morphology.remove_small_objects(fg, 1000)
        bw2 = morphology.binary_closing(bw1, morphology.disk(radius=10))
        bw3 = morphology.binary_opening(bw2, morphology.disk(radius=10))
        label = measure.label(bw3)
        num_fg = np.max(label)

        masks = np.zeros([bg.shape[0], bg.shape[1], 3], dtype=np.uint8)

        if num_fg == 2:
            bw3_1 = label == 1
            bw4_1 = morphology.binary_closing(
                bw3_1, morphology.disk(radius=30))
            bw5_1 = filters.median(bw4_1, morphology.disk(10))

            bw3_2 = label == 2
            bw4_2 = morphology.binary_closing(
                bw3_2, morphology.disk(radius=30))
            bw5_2 = filters.median(bw4_2, morphology.disk(10))

            # masks[:, :, 0] = img_as_bool(bw5_1)
            # masks[:, :, 1] = img_as_bool(bw5_2)
            masks[:, :, 0] = img_as_ubyte(bw5_1)
            masks[:, :, 1] = img_as_ubyte(bw5_2)
        else:
            masks[:, :, 0] = img_as_ubyte(bw3)

        components[frame] = num_fg
        # masks = masks.astype(np.uint8)
        skimage.io.imsave(os.path.join(fg_dir, str(frame) + '.png'), masks)

    components_df = pd.DataFrame({'components': components})
    components_df.to_csv(os.path.join(os.path.dirname(
        frames_dir), 'components.csv'), index=False)

    return components


def split_train_val(dataset_dir, frac_split_train):
    """Split a dataset into subsets train and val inside dataset directory
    Args:
        dataset_dir: path to the dataset containing images and their annotation json files
        frac_split_train: fraction of train subset in the dataset
    Returns:
    """
    json_ids = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]
    random.shuffle(json_ids)

    train_dir = os.path.join(dataset_dir, 'train')
    os.mkdir(train_dir)

    val_dir = os.path.join(dataset_dir, 'val')
    os.mkdir(val_dir)

    for json_id in json_ids[: int(frac_split_train * len(json_ids))]:
        copyfile(os.path.join(dataset_dir, json_id),
                 os.path.join(train_dir, json_id))
        os.remove(os.path.join(dataset_dir, json_id))

        copyfile(os.path.join(dataset_dir, os.path.splitext(json_id)[0] + '.jpg'),
                 os.path.join(train_dir, os.path.splitext(json_id)[0] + '.jpg'))
        os.remove(os.path.join(
            dataset_dir, os.path.splitext(json_id)[0] + '.jpg'))

    for json_id in json_ids[int(frac_split_train * len(json_ids)):]:
        copyfile(os.path.join(dataset_dir, json_id),
                 os.path.join(val_dir, json_id))
        os.remove(os.path.join(dataset_dir, json_id))

        copyfile(os.path.join(dataset_dir, os.path.splitext(json_id)[0] + '.jpg'),
                 os.path.join(val_dir, os.path.splitext(json_id)[0] + '.jpg'))
        os.remove(os.path.join(
            dataset_dir, os.path.splitext(json_id)[0] + '.jpg'))


def create_dataset(images_dir, components_info, num_annotations):
    """Randomly choose images which have one blob in their foreground
    Args:
        images_dir: path to images directory
        components_info: path to a csv file or an array
        num_annotations: the number of images will be picked
    Returns:
    """
    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    dataset_dir = os.path.join(os.path.dirname(images_dir), 'dataset')
    os.mkdir(dataset_dir)

    touching = [i for i in range(len(components)) if components[i] == 1]
    if (components == 1).sum() > num_annotations:
        random.shuffle(touching)
        for image_id in touching[:num_annotations]:
            copyfile(os.path.join(images_dir, str(image_id) + '.jpg'),
                     os.path.join(dataset_dir, str(image_id) + '.jpg'))
    else:
        for image_id in touching:
            copyfile(os.path.join(images_dir, str(image_id) + '.jpg'),
                     os.path.join(dataset_dir, str(image_id) + '.jpg'))


def correct_segmentation_errors(components_info, fix_dir, frames_dir):
    """Count and pick one failed frame in every 3 consecutive fail frames for correcting
    Args:
        components_info: path to a csv file or an array
        fix_dir: path to directory for saving frames chosen
        frames_dir: path to directory containing frames
    Returns:
        correct_frames: the number of frames picked up
    """
    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    errors = np.array(components != 2, dtype=int)
    errors_accumulate = np.zeros(len(errors))
    interval_start = 0

    for i in range(len(errors)):
        if (errors[i] == 1) & (interval_start == 0):
            interval_start = 1
        elif errors[i] == 0:
            interval_start = 0

        if (interval_start == 1) & (i > 0):
            errors_accumulate[i] = errors_accumulate[i - 1] + 1

    # plt.plot(errors_accumulate)
    correct_frames = 0

    if components[0] != 2:
        copyfile(os.path.join(frames_dir, '0.jpg'),
                 os.path.join(fix_dir, '0.jpg'))
        correct_frames = correct_frames + 1

    for i in range(len(errors_accumulate)):
        if (errors_accumulate[i] > 0) & (errors_accumulate[i] % 3 == 0):
            copyfile(os.path.join(frames_dir, str(i) + '.jpg'),
                     os.path.join(fix_dir, str(i) + '.jpg'))
            correct_frames = correct_frames + 1
    return correct_frames


def tracking_inference(fg_dir, components_info):
    """Track the identities of mice
    Args:
        fg_dir: path to directory containing foreground
        components_info: path to a csv file or an array
    """
    tracking_dir = os.path.join(os.path.dirname(fg_dir), 'tracking')

    if not os.path.exists(tracking_dir):
        os.mkdir(tracking_dir)

    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    #I = skimage.io.imread(os.path.join(fg_dir, str(0) + '.png'))
    #skimage.io.imsave(os.path.join(tracking_dir, str(0) + '.png'), I)
    flag = 1
    index = 0
    while(flag):
        if components[index]==2:
            flag = 0
        else:
            index = index + 1

    I = skimage.io.imread(os.path.join(fg_dir, str(index) + '.png'))
    skimage.io.imsave(os.path.join(tracking_dir, str(0) + '.png'), I)

    I = img_as_ubyte(I/255)

    for i in range(1, components.shape[0]):
        I1 = I[:, :, 0]
        I2 = I[:, :, 1]

        if components[i] == 2:
            J = skimage.io.imread(os.path.join(
                fg_dir, str(i) + '.png')) / 255.0

            J1 = J[:, :, 0]
            J2 = J[:, :, 1]

            overlap_1 = np.sum(np.multiply(J1, I1)[:]) / np.sum(I1[:])
            overlap_2 = np.sum(np.multiply(J2, I1)[:]) / np.sum(I1[:])
            overlap_12 = np.abs(overlap_1 - overlap_2)

            overlap_3 = np.sum(np.multiply(J1, I2)[:]) / np.sum(I2[:])
            overlap_4 = np.sum(np.multiply(J2, I2)[:]) / np.sum(I2[:])
            overlap_34 = np.abs(overlap_3 - overlap_4)

            if overlap_12 >= overlap_34:
                if overlap_1 >= overlap_2:
                    I[:, :, 0] = J1
                    I[:, :, 1] = J2
                else:
                    I[:, :, 0] = J2
                    I[:, :, 1] = J1
            else:
                if overlap_3 >= overlap_4:
                    I[:, :, 1] = J1
                    I[:, :, 0] = J2
                else:
                    I[:, :, 1] = J2
                    I[:, :, 0] = J1

            I = I.astype(np.uint8) * 255
            skimage.io.imsave(os.path.join(tracking_dir, str(i) + '.png'), I)

        else:
            #I = I.astype(np.uint8) * 255
            skimage.io.imsave(os.path.join(tracking_dir, str(i) + '.png'), I)


def tracking_inference_h5(frames_dir, components_info, img_shape=[540,540]):
    """Track the identities of mice
    Args:
        fg_dir: path to directory containing foreground
        components_info: path to a csv file or an array

    Returns:
        video_tracking_dict
    """
    # tracking_dir = os.path.join(os.path.dirname(frames_dir), 'tracking')

    # if not os.path.exists(tracking_dir):
    #     os.mkdir(tracking_dir)

    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    #I = skimage.io.imread(os.path.join(fg_dir, str(0) + '.png'))
    #skimage.io.imsave(os.path.join(tracking_dir, str(0) + '.png'), I)
    flag = 1
    index = 0
    while(flag):
        if components[index]==2:
            flag = 0
        else:
            index = index + 1


    #-------------------------------------------
    video_dict = dd.io.load(os.path.join(os.path.dirname(frames_dir), 'masks.h5'))

    video_tracking_dict = {}
    video_tracking_dict[str(0)] = video_dict[str(index)]

    frame_dict = video_dict[str(index)]
    I = boundary2mask(frame_dict, img_shape, num_mice=2)
    #--------------------------------------------
    # I = skimage.io.imread(os.path.join(fg_dir, str(index) + '.png'))
    # skimage.io.imsave(os.path.join(tracking_dir, str(0) + '.png'), I)
    #---------------------------------------------------------

    #I = img_as_ubyte(I/255)

    for i in range(1, components.shape[0]):
        frame_tracking_dict = {}

        I1 = I[:, :, 0]
        I2 = I[:, :, 1]

        if components[i] == 2:
            #J = skimage.io.imread(os.path.join(
            #    frames_dir, str(i) + '.png')) / 255.0
            frame_dict = video_dict[str(i)]
            J = boundary2mask(frame_dict, img_shape, num_mice=2)

            J1 = J[:, :, 0]
            J2 = J[:, :, 1]

            overlap_1 = np.sum(np.multiply(J1, I1)[:]) / np.sum(I1[:])
            overlap_2 = np.sum(np.multiply(J2, I1)[:]) / np.sum(I1[:])
            overlap_12 = np.abs(overlap_1 - overlap_2)

            overlap_3 = np.sum(np.multiply(J1, I2)[:]) / np.sum(I2[:])
            overlap_4 = np.sum(np.multiply(J2, I2)[:]) / np.sum(I2[:])
            overlap_34 = np.abs(overlap_3 - overlap_4)

            if overlap_12 >= overlap_34:
                if overlap_1 >= overlap_2:
                    I[:, :, 0] = J1
                    I[:, :, 1] = J2
                else:
                    I[:, :, 0] = J2
                    I[:, :, 1] = J1
            else:
                if overlap_3 >= overlap_4:
                    I[:, :, 1] = J1
                    I[:, :, 0] = J2
                else:
                    I[:, :, 1] = J2
                    I[:, :, 0] = J1


            mouse1_boundary = find_contours(img_as_ubyte(I[:, :, 0] ), 0.5)[0].astype(int)
            mouse1_temp = mouse1_boundary[:, 0].copy()
            mouse1_boundary[:, 0] = mouse1_boundary[:,1]
            mouse1_boundary[:, 1] = mouse1_temp

            mouse2_boundary = find_contours(img_as_ubyte(I[:, :, 1] ), 0.5)[0].astype(int)
            mouse2_temp = mouse2_boundary[:, 0].copy()
            mouse2_boundary[:, 0] = mouse2_boundary[:,1]
            mouse2_boundary[:, 1] = mouse2_temp

            
            frame_tracking_dict['mouse1'] = mouse1_boundary
            frame_tracking_dict['mouse2'] = mouse2_boundary

            #I = I.astype(np.uint8) * 255
            #skimage.io.imsave(os.path.join(tracking_dir, str(i) + '.png'), I)

        else:
            frame_tracking_dict['mouse1'] = frame_dict['mouse1']
            frame_tracking_dict['mouse2'] = frame_dict['mouse2']
            #I = I.astype(np.uint8) * 255
            #skimage.io.imsave(os.path.join(tracking_dir, str(i) + '.png'), I)

        video_tracking_dict[str(i)] = frame_tracking_dict
    dd.io.save(os.path.join(os.path.dirname(frames_dir), 'masks_tracking.h5'), video_dict, compression=None)

    return video_tracking_dict



def tracking_inference_marker(fg_dir, components_info):
    """Track the identities of mice
    Args:
        fg_dir: path to directory containing foreground
        components_info: path to a csv file or an array
    """
    tracking_dir = os.path.join(os.path.dirname(fg_dir), 'tracking')

    if not os.path.exists(tracking_dir):
        os.mkdir(tracking_dir)

    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    # I = skimage.io.imread(os.path.join(fg_dir, str(0) + '.png'))
    # skimage.io.imsave(os.path.join(tracking_dir, str(0) + '.png'), I)
    flag = 1
    index = 0
    while(flag):
        if components[index]==2:
            flag = 0
        else:
            index = index + 1

    I = skimage.io.imread(os.path.join(fg_dir, str(index) + '.png'))
    skimage.io.imsave(os.path.join(tracking_dir, str(0) + '.png'), I)


    for i in range(1, components.shape[0]):

        if components[i] == 2:
            J = skimage.io.imread(os.path.join(fg_dir, str(i) + '.png')) 
            skimage.io.imsave(os.path.join(tracking_dir, str(i) + '.png'), J)

        else:
            J = skimage.io.imread(os.path.join(fg_dir, str(i-1) + '.png')) 
            skimage.io.imsave(os.path.join(tracking_dir, str(i) + '.png'), J)


def mask_based_detection(tracking_dir, components_info, floor=[[51, 51], [490, 490]], image_shape=(540, 540)):
    """Detect snout and tailbase coordinated from masks
    Args:
        tracking_dir: path to directory containing masks corresponding to identities
        components_info: path to a csv file or an array
        floor: coordinates of top left and bottom right corners of rectangular floor zone
        image_shape: size of frames (height, width)
    Returns:
        np.array(features_mouse1_df): coordinates of snout and tailbase of mouse 1
        np.array(features_mouse2_df): coordinates of snout and tailbase of mouse 2
    """
    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    features_mouse1 = np.zeros((len(components), 4))
    features_mouse2 = np.zeros((len(components), 4))

    floor_zone = np.zeros(image_shape)
    floor_zone[floor[0][0]:floor[1][0], floor[0][1]:floor[1][1]] = 1

    for i in range(len(components)):
        #print('frames: ', i)
        I = (skimage.io.imread(os.path.join(
            tracking_dir, str(i) + '.png')) / 255.0).astype(int)

        I1 = I[:, :, 0]
        I2 = I[:, :, 1]

        properties1 = regionprops(I1.astype(int), I1.astype(float))
        center_of_mass1 = properties1[0].centroid

        properties2 = regionprops(I2.astype(int), I2.astype(float))
        center_of_mass2 = properties2[0].centroid

        BB1 = find_contours(I1, 0.5)[0]
        BB2 = find_contours(I2, 0.5)[0]

        # mouse 1
        center_BB1 = np.sum((BB1 - center_of_mass1) ** 2, axis=1)
        index1 = np.argmax(center_BB1)
        I1_end1 = BB1[index1]

        end1_BB1 = np.sum((BB1 - I1_end1) ** 2, axis=1)
        index2 = np.argmax(end1_BB1)
        I1_end_max = np.max(end1_BB1)
        I1_end2 = BB1[index2]

        condition_mouse1 = np.sum(np.multiply(
            floor_zone, I1)[:]) / np.sum(I1[:])

        if i == 0:
            features_mouse1[i, :2] = I1_end1
            features_mouse1[i, 2:] = I1_end2
        else:
            if ((I1_end_max >= 90) & (condition_mouse1 == 1)):
                features_mouse1[i, :2] = I1_end1
                features_mouse1[i, 2:] = I1_end2
            else:
                end1_nose = np.sum((I1_end1 - features_mouse1[i - 1, :2]) ** 2)
                end1_tail = np.sum((I1_end1 - features_mouse1[i - 1, 2:]) ** 2)

                if end1_nose < end1_tail:
                    features_mouse1[i, :2] = I1_end1
                    features_mouse1[i, 2:] = I1_end2
                else:
                    features_mouse1[i, :2] = I1_end2
                    features_mouse1[i, 2:] = I1_end1

                    # mouse 2
        center_BB2 = np.sum((BB2 - center_of_mass2) ** 2, axis=1)
        index1 = np.argmax(center_BB2)
        I2_end1 = BB2[index1]

        end1_BB2 = np.sum((BB2 - I2_end1) ** 2, axis=1)
        index2 = np.argmax(end1_BB2)
        I2_end_max = np.max(end1_BB2)
        I2_end2 = BB2[index2]

        condition_mouse2 = np.sum(np.multiply(
            floor_zone, I2)[:]) / np.sum(I2[:])

        if i == 0:
            features_mouse2[i, :2] = I2_end1
            features_mouse2[i, 2:] = I2_end2
        else:
            if ((I2_end_max >= 90) & (condition_mouse2 == 1)):
                features_mouse2[i, :2] = I2_end1
                features_mouse2[i, 2:] = I2_end2
            else:
                end1_nose = np.sum((I2_end1 - features_mouse2[i - 1, :2]) ** 2)
                end1_tail = np.sum((I2_end1 - features_mouse2[i - 1, 2:]) ** 2)

                if end1_nose < end1_tail:
                    features_mouse2[i, :2] = I2_end1
                    features_mouse2[i, 2:] = I2_end2
                else:
                    features_mouse2[i, :2] = I2_end2
                    features_mouse2[i, 2:] = I2_end1

    features_mouse1 = np.round(features_mouse1, 2)

    features_mouse1_df = pd.DataFrame({'snout_x': features_mouse1[:, 1],
                                       'snout_y': features_mouse1[:, 0],
                                       'tailbase_x': features_mouse1[:, 3],
                                       'tailbase_y': features_mouse1[:, 2]})
    features_mouse1_df.to_csv(os.path.join(os.path.dirname(tracking_dir), 'features_mouse1_md.csv'),
                              index=False)

    features_mouse2 = np.round(features_mouse2, 2)
    features_mouse2_df = pd.DataFrame({'snout_x': features_mouse2[:, 1],
                                       'snout_y': features_mouse2[:, 0],
                                       'tailbase_x': features_mouse2[:, 3],
                                       'tailbase_y': features_mouse2[:, 2]})
    features_mouse2_df.to_csv(os.path.join(os.path.dirname(tracking_dir), 'features_mouse2_md.csv'),
                              index=False)

    return np.array(features_mouse1_df), np.array(features_mouse2_df)


def mask_based_detection_h5(video_tracking_dict, frames_dir, components_info, floor=[[51, 51], [490, 490]], image_shape=(540, 540)):
    """Detect snout and tailbase coordinated from masks
    Args:
        tracking_dir: path to directory containing masks corresponding to identities
        components_info: path to a csv file or an array
        floor: coordinates of top left and bottom right corners of rectangular floor zone
        image_shape: size of frames (height, width)
    Returns:
        np.array(features_mouse1_df): coordinates of snout and tailbase of mouse 1
        np.array(features_mouse2_df): coordinates of snout and tailbase of mouse 2
    """
    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    features_mouse1 = np.zeros((len(components), 4))
    features_mouse2 = np.zeros((len(components), 4))

    floor_zone = np.zeros(image_shape)
    floor_zone[floor[0][0]:floor[1][0], floor[0][1]:floor[1][1]] = 1

    for i in range(len(components)):
        #print('frames: ', i)
        # I = (skimage.io.imread(os.path.join(
        #     tracking_dir, str(i) + '.png')) / 255.0).astype(int)

        I = boundary2mask(video_tracking_dict[str(i)], image_shape, num_mice=2)

        I1 = I[:, :, 0]
        I2 = I[:, :, 1]

        properties1 = regionprops(I1.astype(int), I1.astype(float))
        center_of_mass1 = properties1[0].centroid

        properties2 = regionprops(I2.astype(int), I2.astype(float))
        center_of_mass2 = properties2[0].centroid

        BB1 = find_contours(I1, 0.5)[0]
        BB2 = find_contours(I2, 0.5)[0]

        # mouse 1
        center_BB1 = np.sum((BB1 - center_of_mass1) ** 2, axis=1)
        index1 = np.argmax(center_BB1)
        I1_end1 = BB1[index1]

        end1_BB1 = np.sum((BB1 - I1_end1) ** 2, axis=1)
        index2 = np.argmax(end1_BB1)
        I1_end_max = np.max(end1_BB1)
        I1_end2 = BB1[index2]

        condition_mouse1 = np.sum(np.multiply(
            floor_zone, I1)[:]) / np.sum(I1[:])

        if i == 0:
            features_mouse1[i, :2] = I1_end1
            features_mouse1[i, 2:] = I1_end2
        else:
            if ((I1_end_max >= 90) & (condition_mouse1 == 1)):
                features_mouse1[i, :2] = I1_end1
                features_mouse1[i, 2:] = I1_end2
            else:
                end1_nose = np.sum((I1_end1 - features_mouse1[i - 1, :2]) ** 2)
                end1_tail = np.sum((I1_end1 - features_mouse1[i - 1, 2:]) ** 2)

                if end1_nose < end1_tail:
                    features_mouse1[i, :2] = I1_end1
                    features_mouse1[i, 2:] = I1_end2
                else:
                    features_mouse1[i, :2] = I1_end2
                    features_mouse1[i, 2:] = I1_end1

                    # mouse 2
        center_BB2 = np.sum((BB2 - center_of_mass2) ** 2, axis=1)
        index1 = np.argmax(center_BB2)
        I2_end1 = BB2[index1]

        end1_BB2 = np.sum((BB2 - I2_end1) ** 2, axis=1)
        index2 = np.argmax(end1_BB2)
        I2_end_max = np.max(end1_BB2)
        I2_end2 = BB2[index2]

        condition_mouse2 = np.sum(np.multiply(
            floor_zone, I2)[:]) / np.sum(I2[:])

        if i == 0:
            features_mouse2[i, :2] = I2_end1
            features_mouse2[i, 2:] = I2_end2
        else:
            if ((I2_end_max >= 90) & (condition_mouse2 == 1)):
                features_mouse2[i, :2] = I2_end1
                features_mouse2[i, 2:] = I2_end2
            else:
                end1_nose = np.sum((I2_end1 - features_mouse2[i - 1, :2]) ** 2)
                end1_tail = np.sum((I2_end1 - features_mouse2[i - 1, 2:]) ** 2)

                if end1_nose < end1_tail:
                    features_mouse2[i, :2] = I2_end1
                    features_mouse2[i, 2:] = I2_end2
                else:
                    features_mouse2[i, :2] = I2_end2
                    features_mouse2[i, 2:] = I2_end1

    features_mouse1 = np.round(features_mouse1, 2)

    features_mouse1_df = pd.DataFrame({'snout_x': features_mouse1[:, 1],
                                       'snout_y': features_mouse1[:, 0],
                                       'tailbase_x': features_mouse1[:, 3],
                                       'tailbase_y': features_mouse1[:, 2]})
    features_mouse1_df.to_csv(os.path.join(os.path.dirname(frames_dir), 'features_mouse1_md.csv'),
                              index=False)

    features_mouse2 = np.round(features_mouse2, 2)
    features_mouse2_df = pd.DataFrame({'snout_x': features_mouse2[:, 1],
                                       'snout_y': features_mouse2[:, 0],
                                       'tailbase_x': features_mouse2[:, 3],
                                       'tailbase_y': features_mouse2[:, 2]})
    features_mouse2_df.to_csv(os.path.join(os.path.dirname(frames_dir), 'features_mouse2_md.csv'),
                              index=False)

    return np.array(features_mouse1_df), np.array(features_mouse2_df)


def mice_separation(tracking_dir, frames_dir, bg_dir):
    """Separate the sequence of frames into 2 videos. Each video contains one mouse
    Args:
        tracking_dir: path to directory containing masks corresponding to identities
        frames_dir: path to frames directory
        bg_dir: path to the background image
    """
    bg = img_as_ubyte(skimage.io.imread(bg_dir))
    num_images = len(os.listdir(tracking_dir))

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    video_mouse1_dir = os.path.join(
        os.path.dirname(tracking_dir), 'mouse1.avi')
    video1 = cv2.VideoWriter(video_mouse1_dir, fourcc,
                             30, (bg.shape[1], bg.shape[0]), 0)

    video_mouse2_dir = os.path.join(
        os.path.dirname(tracking_dir), 'mouse2.avi')
    video2 = cv2.VideoWriter(video_mouse2_dir, fourcc,
                             30, (bg.shape[1], bg.shape[0]), 0)

    for i in range(num_images):
        masks = skimage.io.imread(os.path.join(
            tracking_dir, str(i) + '.png')) / 255
        image = skimage.io.imread(os.path.join(frames_dir, str(i) + '.jpg'))

        mask1 = masks[:, :, 0].astype(np.uint8)
        mask1 = dilation(mask1, square(10))

        mask2 = masks[:, :, 1].astype(np.uint8)
        mask2 = dilation(mask2, square(10))

        mouse2_remove = (mask2 != 1) | (mask1 == 1)
        mouse1 = np.multiply(image, mouse2_remove) + \
            np.multiply(bg, (1 - mouse2_remove))
        mouse1 = img_as_ubyte(mouse1)

        mouse1_remove = (mask1 != 1) | (mask2 == 1)
        mouse2 = np.multiply(image, mouse1_remove) + \
            np.multiply(bg, (1 - mouse1_remove))
        mouse2 = img_as_ubyte(mouse2)

        video1.write(mouse1)
        video2.write(mouse2)

    cv2.destroyAllWindows()
    video1.release()
    video2.release()


def deeplabcut_detection(config_dir, video_dir):
    """Detect snout and tailbase coordinated with Deeplabcut model
    Args:
        config_dir: path to config file
        video_dir: path to video input
    Returns:
        features_mouse1: coordinates of snout and tailbase of mouse 1
        features_mouse2: coordinates of snout and tailbase of mouse 2
    """
    deeplabcut.analyze_videos(config_dir, video_dir, videotype='.avi')

    dlc_output = [f for f in os.listdir(
        os.path.dirname(video_dir[0])) if f.endswith('.h5')]

    # mouse1
    mouse1_dlc = pd.read_hdf(os.path.join(
        os.path.dirname(video_dir[0]), dlc_output[0]))
    features_mouse1 = mouse1_dlc.values[:, [0, 1, 9, 10]]
    features_mouse1 = np.round(features_mouse1, 2)
    features_mouse1_df = pd.DataFrame({'snout_x': np.round(mouse1_dlc.values[:, 0], 2),
                                       'snout_y': np.round(mouse1_dlc.values[:, 1], 2),
                                       'tailbase_x': np.round(mouse1_dlc.values[:, 9], 2),
                                       'tailbase_y': np.round(mouse1_dlc.values[:, 10], 2)})
    features_mouse1_df.to_csv(os.path.join(os.path.dirname(video_dir[0]), 'features_mouse1_dlc.csv'),
                              index=False)

    mouse2_dlc = pd.read_hdf(os.path.join(
        os.path.dirname(video_dir[0]), dlc_output[1]))
    features_mouse2 = mouse2_dlc.values[:, [0, 1, 9, 10]]
    features_mouse2 = np.round(features_mouse2, 2)
    features_mouse2_df = pd.DataFrame({'snout_x': np.round(mouse2_dlc.values[:, 0], 2),
                                       'snout_y': np.round(mouse2_dlc.values[:, 1], 2),
                                       'tailbase_x': np.round(mouse2_dlc.values[:, 9], 2),
                                       'tailbase_y': np.round(mouse2_dlc.values[:, 10], 2)})
    features_mouse2_df.to_csv(os.path.join(os.path.dirname(video_dir[0]), 'features_mouse2_dlc.csv'),
                              index=False)

    return features_mouse1, features_mouse2


def ensemble_features(features_mouse_md, features_mouse_dlc, tracking_dir, mouse_id=1):
    """Ensemble the result of mask-based detection and deeplabcut-based detection
    Args:
        features_mouse_md: coordinates of snout and tailbase generated by mask-based detection
        features_mouse_dlc: coordinates of snout and tailbase generated by deeplabcut detection
        tracking_dir: path to directory containing masks corresponding to identities
        mouse_id: mouse id ( 1 or 2)
    Returns:
        features_ensemble: ensemble coordinates of snout and tailbase
    """
    features_ensemble = np.zeros(features_mouse_md.shape)
    for i in range(len(features_mouse_md)):
        masks = skimage.io.imread(os.path.join(
            tracking_dir, str(i) + '.png')) / 255.0

        mask = masks[:, :, mouse_id - 1].astype(int)
        mask = dilation(mask, square(15))

        nose_DLC = np.zeros(mask.shape)
        tailbase_DLC = np.zeros(mask.shape)

        nose_DLC[int(features_mouse_dlc[i, 1]),
                 int(features_mouse_dlc[i, 0])] = 1
        tailbase_DLC[int(features_mouse_dlc[i, 3]),
                     int(features_mouse_dlc[i, 2])] = 1

        if np.sum(np.multiply(mask, nose_DLC)[:]) > 0:
            features_ensemble[i, :2] = features_mouse_dlc[i, :2]
        else:
            features_ensemble[i, :2] = features_mouse_md[i, :2]

        if np.sum(np.multiply(mask, tailbase_DLC)[:]) > 0:
            features_ensemble[i, 2:] = features_mouse_dlc[i, 2:]
        else:
            features_ensemble[i, 2:] = features_mouse_md[i, 2:]

    features_ensemble_df = pd.DataFrame({'snout_x': features_ensemble[:, 0],
                                         'snout_y': features_ensemble[:, 1],
                                         'tailbase_x': features_ensemble[:, 2],
                                         'tailbase_y': features_ensemble[:, 3]})
    features_ensemble_df.to_csv(os.path.join(os.path.dirname(tracking_dir),
                                             'features_mouse' + str(mouse_id) + '_ensemble.csv'), index=False)

    return features_ensemble


def labelmejson_to_png(fix_dir, output_dir):
    """Convert annotations created by labelme to images
    Args:
        fix_dir: path to directory for saving frames chosen
        output_dir: path to save output
    Returns:
    """
    json_ids = [f for f in os.listdir(fix_dir) if f.endswith('.json')]

    dataset_fix = MouseDataset()
    dataset_fix.load_mouse(fix_dir, "")

    class_name_to_id = {label["name"]: label["id"]
                        for label in dataset_fix.class_info}
    # Read mask file from json
    for json_id in json_ids:
        json_id_dir = os.path.join(fix_dir, json_id)
        with open(json_id_dir) as f:
            data = json.load(f)
            image_shape = (data['imageHeight'], data['imageWidth'])

            cls, masks = shapes_to_labels_masks(img_shape=image_shape,
                                                shapes=data['shapes'],
                                                label_name_to_value=class_name_to_id)

        masks_rgb = np.zeros(
            (data['imageHeight'], data['imageWidth'], 3), dtype=np.float)
        masks_rgb[:, :, :2] = masks[:, :, :2]
        skimage.io.imsave(os.path.join(
            output_dir, os.path.splitext(json_id)[0] + '.png'), masks_rgb)


def mouse_mrcnn_segmentation(components_info, frames_dir, background_dir, model_dir, model_path=None):
    """Segment mice using Mask-RCNN model
    Args:
        components_info: path to a csv file or an array
        frames_dir: path to frames directory
        background_dir: path to background image
        model_dir: path to save log and trained model
        model_path: path to model weights
    Returns:
        components: array of the number of blobs in each frames
    """
    config = InferenceConfig()
    # config.set_config(batch_size=1)

    # Create model object in inference mode.
    model = modellib.MaskRCNN(
        mode="inference", model_dir=model_dir, config=config)

    if model_path:
        model.load_weights(model_path, by_name=True)
    else:
        model_path = model.find_last()
        model.load_weights(model_path, by_name=True)

    output_dir = os.path.join(os.path.dirname(frames_dir), 'FG')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    bg = cv2.imread(background_dir)

    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    print("The video has {} frames: ".format(components.shape[0]))

    for i in range(components.shape[0]):
        if components[i] != 2:
            image_name = str(i) + '.jpg'
            image = skimage.io.imread(frames_dir + '/' + image_name)

            if image.ndim == 2:
                image_rgb = skimage.color.gray2rgb(image)
            else:
                image_rgb = image

            results = model.detect([image_rgb], verbose=0)

            results_package = results[0]

            masks_rgb = np.zeros((bg.shape[0], bg.shape[1], 3), dtype=np.uint8)

            if len(results_package["scores"]) >= 2:
                class_ids = results_package['class_ids'][:2]
                scores = results_package['scores'][:2]
                masks = results_package['masks'][:, :, :2]  # Bool
                rois = results_package['rois'][:2, :]

                masks_1 = morphology.remove_small_objects(masks[:, :, 0], 1000)
                masks_1 = morphology.binary_dilation(
                    masks_1, morphology.disk(radius=3))

                masks_2 = morphology.remove_small_objects(masks[:, :, 1], 1000)
                masks_2 = morphology.binary_dilation(
                    masks_2, morphology.disk(radius=3))

                if (masks_1.sum().sum() > 0) & (masks_2.sum().sum() > 0):
                    masks_rgb[:, :, 0] = img_as_ubyte(masks_1)
                    masks_rgb[:, :, 1] = img_as_ubyte(masks_2)

                    components[i] = 2

            skimage.io.imsave(os.path.join(
                output_dir, str(i) + '.png'), masks_rgb)

    components_df = pd.DataFrame({'components': components})
    components_df.to_csv(os.path.join(os.path.dirname(
        frames_dir), 'components.csv'), index=False)

    return components


def mouse_mrcnn_segmentation_h5(components_info, frames_dir, background_dir, model_dir, model_path=None):
    """Segment mice using Mask-RCNN model
    Args:
        components_info: path to a csv file or an array
        frames_dir: path to frames directory
        background_dir: path to background image
        model_dir: path to save log and trained model
        model_path: path to model weights
    Returns:
        components: array of the number of blobs in each frames
    """
    config = InferenceConfig()
    # config.set_config(batch_size=1)

    # Create model object in inference mode.
    model = modellib.MaskRCNN(
        mode="inference", model_dir=model_dir, config=config)

    if model_path:
        model.load_weights(model_path, by_name=True)
    else:
        model_path = model.find_last()
        model.load_weights(model_path, by_name=True)

    output_dir = os.path.join(os.path.dirname(frames_dir), 'FG')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    bg = cv2.imread(background_dir)

    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    print("The video has {} frames: ".format(components.shape[0]))

    video_dict = dd.io.load(os.path.join(os.path.dirname(frames_dir), 'masks_fg.h5'))

    for i in range(components.shape[0]):
        frame_dict = {}
        if components[i] != 2:
            image_name = str(i) + '.jpg'
            image = skimage.io.imread(frames_dir + '/' + image_name)

            if image.ndim == 2:
                image_rgb = skimage.color.gray2rgb(image)
            else:
                image_rgb = image

            results = model.detect([image_rgb], verbose=0)

            results_package = results[0]

            masks_rgb = np.zeros((bg.shape[0], bg.shape[1], 3), dtype=np.uint8)

            if len(results_package["scores"]) >= 2:
                class_ids = results_package['class_ids'][:2]
                scores = results_package['scores'][:2]
                masks = results_package['masks'][:, :, :2]  # Bool
                rois = results_package['rois'][:2, :]

                masks_1 = morphology.remove_small_objects(masks[:, :, 0], 1000)
                masks_1 = morphology.binary_dilation(
                    masks_1, morphology.disk(radius=3))

                masks_2 = morphology.remove_small_objects(masks[:, :, 1], 1000)
                masks_2 = morphology.binary_dilation(
                    masks_2, morphology.disk(radius=3))

                if (masks_1.sum().sum() > 0) & (masks_2.sum().sum() > 0):
                    masks_rgb[:, :, 0] = img_as_ubyte(masks_1)
                    masks_rgb[:, :, 1] = img_as_ubyte(masks_2)

                    components[i] = 2


                    mouse1_boundary = find_contours(img_as_ubyte(masks_1), 0.5)[0].astype(int)
                    mouse1_temp = mouse1_boundary[:, 0].copy()
                    mouse1_boundary[:, 0] = mouse1_boundary[:,1]
                    mouse1_boundary[:, 1] = mouse1_temp


                    mouse2_boundary = find_contours(img_as_ubyte(masks_2), 0.5)[0].astype(int)
                    mouse2_temp = mouse2_boundary[:, 0].copy()
                    mouse2_boundary[:, 0] = mouse2_boundary[:,1]
                    mouse2_boundary[:, 1] = mouse2_temp

                    # frame_dict['mouse1'] = find_contours(img_as_ubyte(masks_1), 0.5)[0].astype(int)
                    # frame_dict['mouse2'] = find_contours(img_as_ubyte(masks_2), 0.5)[0].astype(int)
                    frame_dict['mouse1'] =  mouse1_boundary
                    frame_dict['mouse2'] =  mouse2_boundary


            skimage.io.imsave(os.path.join(
                output_dir, str(i) + '.png'), masks_rgb)

        video_dict[str(i)] = frame_dict            
    
    dd.io.save(os.path.join(os.path.dirname(frames_dir), 'masks.h5'), video_dict, compression=None)

    components_df = pd.DataFrame({'components': components})
    components_df.to_csv(os.path.join(os.path.dirname(
        frames_dir), 'components.csv'), index=False)

    return components



def mouse_mrcnn_segmentation_multi_images(components_info, frames_dir, background_dir, model_dir, model_path=None, batch_size=2):
    """Segment mice using Mask-RCNN model
    Args:
        components_info: path to a csv file or an array
        frames_dir: path to frames directory
        background_dir: path to background image
        model_dir: path to save log and trained model
        model_path: path to model weights
        batch_size: int
    Returns:
        components: array of the number of blobs in each frames
    """
    config = InferenceConfig()
    config.IMAGES_PER_GPU = batch_size

    # Create model object in inference mode.
    model = modellib.MaskRCNN(
        mode="inference", model_dir=model_dir, config=config)

    if model_path:
        model.load_weights(model_path, by_name=True)
    else:
        model_path = model.find_last()
        model.load_weights(model_path, by_name=True)

    output_dir = os.path.join(os.path.dirname(frames_dir), 'FG')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    bg = cv2.imread(background_dir)

    if isinstance(components_info, str):
        components = pd.read_csv(components_info)
        components = np.array(components.loc[:, 'components'])
    else:
        components = components_info

    print("The video has {} frames: ".format(components.shape[0]))

    nbatchs = int(np.ceil(components.shape[0]/batch_size))

    for batch in range(nbatchs):
        if batch < nbatchs-1:
            nimages = batch_size
        else:
            nimages = components.shape[0] - batch_size*(nbatchs-1)

        image_rgb_batch = []

        # ----inference----------------
        for ind in range(nimages):
            image_name = str(batch*batch_size + ind) + '.jpg'
            image = skimage.io.imread(frames_dir + '/' + image_name)
            if image.ndim == 2:
                image_rgb = skimage.color.gray2rgb(image)
            else:
                image_rgb = image

            image_rgb_batch.append(image_rgb)

        results = model.detect(image_rgb_batch, verbose=0)

        # ----save results-------------
        for ind in range(nimages):
            print('Segment: ', batch*batch_size + ind)
            image_name = str(batch*batch_size + ind) + '.png'
            results_package = results[ind]
            masks_rgb = np.zeros((bg.shape[0], bg.shape[1], 3), dtype=np.uint8)

            if len(results_package["scores"]) >= 2:
                class_ids = results_package['class_ids'][:2]
                scores = results_package['scores'][:2]
                masks = results_package['masks'][:, :, :2]  # Bool
                rois = results_package['rois'][:2, :]

                masks_1 = morphology.remove_small_objects(masks[:, :, 0], 1000)
                masks_1 = morphology.binary_dilation(
                    masks_1, morphology.disk(radius=3))

                masks_2 = morphology.remove_small_objects(masks[:, :, 1], 1000)
                masks_2 = morphology.binary_dilation(
                    masks_2, morphology.disk(radius=3))

                if (masks_1.sum().sum() > 0) & (masks_2.sum().sum() > 0):
                    masks_rgb[:, :, 0] = img_as_ubyte(masks_1)
                    masks_rgb[:, :, 1] = img_as_ubyte(masks_2)

                    components[batch*batch_size + ind] = 2

            skimage.io.imsave(os.path.join(output_dir, image_name), masks_rgb)

    components_df = pd.DataFrame({'components': components})
    components_df.to_csv(os.path.join(os.path.dirname(
        frames_dir), 'components.csv'), index=False)

    return components


def check_mrcnn_model_path(model_dir):
    dir_names = next(os.walk(model_dir))[1]
    key = "mask_rcnn"
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
        return False
    else:
        return True


def dlc_snout_tailbase(dlc_file):
    """Extract coordinates of snout and tailbase 
    Args:
        dlc_file: path to .h5 file
    Returns:
        df_mouse1: pd.DataFrame
        df_mouse2: pd.DataFrame
    """

    df_dlc = pd.read_hdf(dlc_file)
    scorer = df_dlc.columns[1][0]

    df_mouse1 = pd.DataFrame()
    df_mouse1['snout_x'] = df_dlc[scorer, 'mouse1', 'snout', 'x']
    df_mouse1['snout_y'] = df_dlc[scorer, 'mouse1', 'snout', 'y']
    df_mouse1['tailbase_x'] = df_dlc[scorer, 'mouse1', 'tailbase', 'x']
    df_mouse1['tailbase_y'] = df_dlc[scorer, 'mouse1', 'tailbase', 'y']

    df_mouse2 = pd.DataFrame()
    df_mouse2['snout_x'] = df_dlc[scorer, 'mouse2', 'snout', 'x']
    df_mouse2['snout_y'] = df_dlc[scorer, 'mouse2', 'snout', 'y']
    df_mouse2['tailbase_x'] = df_dlc[scorer, 'mouse2', 'tailbase', 'x']
    df_mouse2['tailbase_y'] = df_dlc[scorer, 'mouse2', 'tailbase', 'y']

    return df_mouse1, df_mouse2


def deeplabcut_detection_multi(config_path, video_path, shuffle=1, trainingsetindex=0, track_method='skeleton', videotype='.avi'):
    """Function to get snout and tailbase through deeplabcut
    Args:

    Returns:
    """
    videos = [video_path]
    deeplabcut.analyze_videos(config_path, videos, videotype=videotype)
    deeplabcut.convert_detections2tracklets(config_path, videos, videotype=videotype,
                                            shuffle=shuffle, trainingsetindex=trainingsetindex, track_method=track_method)

    # --------------bypass refining tracklets-----------------------
    # --------find pickle file containing tracklets------------------
    file_list = [f for f in glob.glob(os.path.join(
        os.path.dirname(video_path), '*.pickle'))]
    video_name = ntpath.basename(video_path).split('.')[0]
    if track_method == 'skeleton':
        for filename in file_list:
            if (video_name in filename) and ('sk.pickle' in filename):
                tracklet_result = filename
    elif track_method == 'box':
        for filename in file_list:
            if (video_name in filename) and ('bx.pickle' in filename):
                tracklet_result = filename

    df_tracklet = pd.read_pickle(tracklet_result)
    mouse1_columns = [(col[0], 'mouse1', col[1], col[2])
                      for col in df_tracklet['header']]
    mouse2_columns = [(col[0], 'mouse2', col[1], col[2])
                      for col in df_tracklet['header']]
    mice_columns = pd.MultiIndex.from_tuples(
        mouse1_columns + mouse2_columns, names=["scorer", "individuals", "bodyparts", "coords"])

    df_mouse1 = pd.DataFrame(df_tracklet[0])
    df_mouse2 = pd.DataFrame(df_tracklet[1])

    df_mice = pd.concat([df_mouse1.T, df_mouse2.T], axis=1)
    df_mice_all = pd.DataFrame(df_mice.values, columns=mice_columns)
    df_mice_all.to_hdf(os.path.splitext(tracklet_result)
                       [0]+'.h5', key='df', mode='w')

    # -----------filter-----------------------------------------------
    deeplabcut.filterpredictions(
        config_path, video_path, track_method='skeleton')

    # ----------find filter result-----------------------------------
    file_list = [f for f in glob.glob(
        os.path.join(os.path.dirname(video_path), '*.h5'))]
    video_name = ntpath.basename(video_path).split('.')[0]
    if track_method == 'skeleton':
        for filename in file_list:
            if (video_name in filename) and ('sk_filtered.h5' in filename):
                filter_result = filename
    elif track_method == 'box':
        for filename in file_list:
            if (video_name in filename) and ('bx_filtered.h5' in filename):
                filter_result = filename

    #df_mouse1, df_mouse2 = dlc_snout_tailbase(filter_result)

    # #---------Extracting snout and tailbase coordinates-------------
    # df = pd.read_pickle(dlc_result)
    # df_mouse1_all = pd.DataFrame(df[0])
    # df_mouse1_all= pd.DataFrame(df_mouse1_all.T.values, columns=df['header'])

    # df_mouse2_all = pd.DataFrame(df[1])
    # df_mouse2_all = pd.DataFrame(df_mouse2_all.T.values, columns=df['header'])

    # scorer = df_mouse1_all.columns[0][0]

    # df_mouse1 = pd.DataFrame()
    # df_mouse1['snout_x'] = df_mouse1_all[scorer,'snout', 'x']
    # df_mouse1['snout_y'] = df_mouse1_all[scorer,'snout', 'y']
    # df_mouse1['tailbase_x'] = df_mouse1_all[scorer,'tailbase', 'x']
    # df_mouse1['tailbase_y'] = df_mouse1_all[scorer,'tailbase', 'y']

    # df_mouse2 = pd.DataFrame()
    # df_mouse2['snout_x'] = df_mouse2_all[scorer,'snout', 'x']
    # df_mouse2['snout_y'] = df_mouse2_all[scorer,'snout', 'y']
    # df_mouse2['tailbase_x'] = df_mouse2_all[scorer,'tailbase', 'x']
    # df_mouse2['tailbase_y'] = df_mouse2_all[scorer,'tailbase', 'y']
    return filter_result


def deeplabcut_detection_multi_without_refine(config_path, video_path, shuffle=1, trainingsetindex=0, track_method='skeleton', videotype='.avi'):
    """Function to get snout and tailbase through deeplabcut
    Args:
        config_path: path to config.yaml
        video_path: video path
        shuffle: int
        trainingsetindex: int
        track_method: str
        videotype: str
    Returns:
        filter_result: path to filtered result
    """
    videos = [video_path]
    deeplabcut.analyze_videos(config_path, videos, videotype=videotype)
    deeplabcut.convert_detections2tracklets(config_path, videos, videotype=videotype,
                                            shuffle=shuffle, trainingsetindex=trainingsetindex, track_method=track_method)

    # --------find pickle file containing tracklets------------------
    file_list = [f for f in glob.glob(os.path.join(
        os.path.dirname(video_path), '*.pickle'))]
    video_name = ntpath.basename(video_path).split('.')[0]
    if track_method == 'skeleton':
        for filename in file_list:
            if (video_name in filename) and ('sk.pickle' in filename):
                tracklet_result = filename
    elif track_method == 'box':
        for filename in file_list:
            if (video_name in filename) and ('bx.pickle' in filename):
                tracklet_result = filename
    elif track_method == 'ellipse':
        for filename in file_list:
            if (video_name in filename) and ('el.pickle' in filename):
                tracklet_result = filename

    # -------find pickle file containing assemblies ------------
    file_list = [f for f in glob.glob(os.path.join(
        os.path.dirname(video_path), '*.pickle'))]
    video_name = ntpath.basename(video_path).split('.')[0]

    for filename in file_list:
        if (video_name in filename) and ('assemblies' in filename):
            assembly_result = filename

    # -----------columns----------------------------
    tracklets = pd.read_pickle(tracklet_result)
    scorer = tracklets['header'][0][0]

    mouse1_columns = [(col[0], 'mouse1', col[1], col[2])
                      for col in tracklets['header']]
    mouse2_columns = [(col[0], 'mouse2', col[1], col[2])
                      for col in tracklets['header']]
    mice_columns = mouse1_columns + mouse2_columns

    all_columns = pd.MultiIndex.from_tuples(
        mice_columns, names=["scorer", "individuals", "bodyparts", "coords"])

    # -------------bypass refining----------------
    assemblies = pd.read_pickle(assembly_result)

    cap = cv2.VideoCapture(video_path)
    nframes = int(cap.get(7))

    # ---------------------------------------------------
    mouse1 = np.empty((nframes, len(mouse1_columns)))
    mouse1[:] = np.NaN

    mouse2 = np.empty((nframes, len(mouse2_columns)))
    mouse2[:] = np.NaN
    # ---------------------------------------------------

    for frame in assemblies.keys():

        try:
            mouse1[frame, :] = assemblies[frame][0][:, 0:3].flatten()
        except:
            continue

        try:
            mouse2[frame, :] = assemblies[frame][1][:, 0:3].flatten()
        except:
            continue

    mice = np.concatenate((mouse1, mouse2), axis=1)
    df_mice = pd.DataFrame(mice, columns=all_columns)

    # ------------refine--------------------------
    df_mice.to_hdf(os.path.splitext(tracklet_result)
                   [0]+'.h5', key='df', mode='w')
    # ----------- filter--------------------------
    deeplabcut.filterpredictions(
        config_path, video_path, track_method=track_method)

    # ---------- find filter result-----------------------------------
    file_list = [f for f in glob.glob(
        os.path.join(os.path.dirname(video_path), '*.h5'))]
    video_name = ntpath.basename(video_path).split('.')[0]
    if track_method == 'skeleton':
        for filename in file_list:
            if (video_name in filename) and ('sk_filtered.h5' in filename):
                filter_result = filename
    elif track_method == 'box':
        for filename in file_list:
            if (video_name in filename) and ('bx_filtered.h5' in filename):
                filter_result = filename
    elif track_method == 'ellipse':
        for filename in file_list:
            if (video_name in filename) and ('el_filtered.h5' in filename):
                filter_result = filename

    return filter_result


def ensemble_features_multi(mouse1_md, mouse2_md, mouse1_dlc, mouse2_dlc, tracking_dir):
    """Ensemble the result of mask-based detection and deeplabcut-based detection
    Args:
        mouse1_md: coordinates of snout and tailbase generated by mask-based detection
        mouse2_md: coordinates of snout and tailbase generated by mask-based detection
        mouse1_dlc: coordinates of snout and tailbase generated by deeplabcut detection
        mouse1_dlc: coordinates of snout and tailbase generated by deeplabcut detection
        tracking_dir: path to directory containing masks corresponding to identities

    Returns:
        df_mouse1_ensemble: ensemble coordinates of snout and tailbase of mouse1
        df_mouse2_ensemble: ensemble coordinates of snout and tailbase of mouse1
    """

    components = pd.read_csv(os.path.join(
        os.path.dirname(tracking_dir), 'components.csv'))

    mouse1_ensemble = np.zeros(mouse1_md.shape)
    mouse2_ensemble = np.zeros(mouse2_md.shape)

    mouse1_dlc = np.array(mouse1_dlc)
    mouse2_dlc = np.array(mouse2_dlc)

    flag1 = np.zeros((len(mouse1_md),))
    flag2 = np.zeros((len(mouse2_md),))

    for i in range(len(mouse1_md)):

        masks = skimage.io.imread(os.path.join(
            tracking_dir, str(i) + '.png')) / 255.0
        mask1 = masks[:, :, 0].astype(int)
        mask2 = masks[:, :, 1].astype(int)

        nose1_DLC = np.zeros(mask1.shape)
        tail1_DLC = np.zeros(mask1.shape)

        nose2_DLC = np.zeros(mask2.shape)
        tail2_DLC = np.zeros(mask2.shape)

        try:
            nose1_DLC[mouse1_dlc[i, 1].astype(
                int), mouse1_dlc[i, 0].astype(int)] = 1
            tail1_DLC[mouse1_dlc[i, 3].astype(
                int), mouse1_dlc[i, 2].astype(int)] = 1
        except:
            pass

        try:
            nose2_DLC[mouse2_dlc[i, 1].astype(
                int), mouse2_dlc[i, 0].astype(int)] = 1
            tail2_DLC[mouse2_dlc[i, 3].astype(
                int), mouse2_dlc[i, 2].astype(int)] = 1

        except:
            pass

        # -----------mouse 1---------------------

        if (np.sum(np.sum(nose1_DLC*mask1)) > 0) & (np.sum(np.sum(tail1_DLC*mask1)) > 0):

            mouse1_ensemble[i, 0:2] = mouse1_dlc[i, 0:2]
            mouse1_ensemble[i, 2:4] = mouse1_dlc[i, 2:4]
            flag1[i] = 1

        elif (np.sum(np.sum(nose2_DLC*mask1)) > 0) & (np.sum(np.sum(tail2_DLC*mask1)) > 0):
            mouse1_ensemble[i, 0:2] = mouse2_dlc[i, 0:2]
            mouse1_ensemble[i, 2:4] = mouse2_dlc[i, 2:4]
            flag1[i] = 1

        else:
            mouse1_ensemble[i, 0:2] = mouse1_md[i, 0:2]
            mouse1_ensemble[i, 2:4] = mouse1_md[i, 2:4]
            flag1[i] = 0

        # --------logic to fix swapping: ------------
        if i > 0:
            if (flag1[i] == 0) & (flag1[i-1] == 1) & (components.loc[i, 'components'] == 2):
                mouse1_snout2snout = np.sum(
                    (mouse1_ensemble[i, 0:2]-mouse1_ensemble[i-1, 0:2]) ** 2)
                mouse1_snout2tail = np.sum(
                    (mouse1_ensemble[i, 0:2]-mouse1_ensemble[i-1, 2:4]) ** 2)
                if mouse1_snout2tail < mouse1_snout2snout:
                    temp1 = mouse1_ensemble[i, 0:2].copy()
                    mouse1_ensemble[i, 0:2] = mouse1_ensemble[i, 2:4]
                    mouse1_ensemble[i, 2:4] = temp1

        # --------mouse 2-------------------------
        if (np.sum(np.sum(nose1_DLC*mask2)) > 0) & (np.sum(np.sum(tail1_DLC*mask2)) > 0):

            mouse2_ensemble[i, 0:2] = mouse1_dlc[i, 0:2]
            mouse2_ensemble[i, 2:4] = mouse1_dlc[i, 2:4]
            flag2[i] = 1

        elif (np.sum(np.sum(nose2_DLC*mask2)) > 0) & (np.sum(np.sum(tail2_DLC*mask2)) > 0):
            mouse2_ensemble[i, 0:2] = mouse2_dlc[i, 0:2]
            mouse2_ensemble[i, 2:4] = mouse2_dlc[i, 2:4]
            flag2[i] = 1

        else:
            mouse2_ensemble[i, 0:2] = mouse2_md[i, 0:2]
            mouse2_ensemble[i, 2:4] = mouse2_md[i, 2:4]
            flag2[i] = 0

        # --------logic to fix swapping: ------------
        if i > 0:
            if (flag2[i] == 0) & (flag2[i-1] == 1) & (components.loc[i, 'components'] == 2):
                mouse2_snout2snout = np.sum(
                    (mouse2_ensemble[i, 0:2]-mouse2_ensemble[i-1, 0:2]) ** 2)
                mouse2_snout2tail = np.sum(
                    (mouse2_ensemble[i, 0:2]-mouse2_ensemble[i-1, 2:4]) ** 2)
                if mouse2_snout2tail < mouse2_snout2snout:
                    temp2 = mouse2_ensemble[i, 0:2].copy()
                    mouse2_ensemble[i, 0:2] = mouse2_ensemble[i, 2:4]
                    mouse2_ensemble[i, 2:4] = temp2

    # mouse1_ensemble[:,1] = 540-mouse1_ensemble[:,1]
    # mouse1_ensemble[:,3] = 540-mouse1_ensemble[:,3]

    # mouse2_ensemble[:,1] = 540-mouse2_ensemble[:,1]
    # mouse2_ensemble[:,3] = 540-mouse2_ensemble[:,3]

    df_mouse1_ensemble = pd.DataFrame({'snout_x': mouse1_ensemble[:, 0],
                                       'snout_y': mouse1_ensemble[:, 1],
                                       'tailbase_x': mouse1_ensemble[:, 2],
                                       'tailbase_y': mouse1_ensemble[:, 3]})

    df_mouse2_ensemble = pd.DataFrame({'snout_x': mouse2_ensemble[:, 0],
                                       'snout_y': mouse2_ensemble[:, 1],
                                       'tailbase_x': mouse2_ensemble[:, 2],
                                       'tailbase_y': mouse2_ensemble[:, 3]})

    df_mouse1_ensemble.to_csv(os.path.join(os.path.dirname(tracking_dir),
                                           'mouse1_ensemble.csv'), index=False)

    df_mouse2_ensemble.to_csv(os.path.join(os.path.dirname(tracking_dir),
                                           'mouse2_ensemble.csv'), index=False)

    return df_mouse1_ensemble, df_mouse2_ensemble


def ensemble_features_multi_h5(mouse1_md, mouse2_md, mouse1_dlc, mouse2_dlc, components, video_tracking_dict, img_shape, frames_dir):
    """Ensemble the result of mask-based detection and deeplabcut-based detection
    Args:
        mouse1_md: coordinates of snout and tailbase generated by mask-based detection
        mouse2_md: coordinates of snout and tailbase generated by mask-based detection
        mouse1_dlc: coordinates of snout and tailbase generated by deeplabcut detection
        mouse1_dlc: coordinates of snout and tailbase generated by deeplabcut detection
        tracking_dir: path to directory containing masks corresponding to identities

    Returns:
        df_mouse1_ensemble: ensemble coordinates of snout and tailbase of mouse1
        df_mouse2_ensemble: ensemble coordinates of snout and tailbase of mouse1
    """

    # components = pd.read_csv(os.path.join(
    #     os.path.dirname(tracking_dir), 'components.csv'))

    mouse1_ensemble = np.zeros(mouse1_md.shape)
    mouse2_ensemble = np.zeros(mouse2_md.shape)

    mouse1_dlc = np.array(mouse1_dlc)
    mouse2_dlc = np.array(mouse2_dlc)

    flag1 = np.zeros((len(mouse1_md),))
    flag2 = np.zeros((len(mouse2_md),))

    for i in range(len(mouse1_md)):

        # masks = skimage.io.imread(os.path.join(
        #     tracking_dir, str(i) + '.png')) / 255.0

        masks = boundary2mask(video_tracking_dict[str(i)], img_shape, num_mice=2)

        mask1 = masks[:, :, 0].astype(int)
        mask2 = masks[:, :, 1].astype(int)

        nose1_DLC = np.zeros(mask1.shape)
        tail1_DLC = np.zeros(mask1.shape)

        nose2_DLC = np.zeros(mask2.shape)
        tail2_DLC = np.zeros(mask2.shape)

        try:
            nose1_DLC[mouse1_dlc[i, 1].astype(
                int), mouse1_dlc[i, 0].astype(int)] = 1
            tail1_DLC[mouse1_dlc[i, 3].astype(
                int), mouse1_dlc[i, 2].astype(int)] = 1
        except:
            pass

        try:
            nose2_DLC[mouse2_dlc[i, 1].astype(
                int), mouse2_dlc[i, 0].astype(int)] = 1
            tail2_DLC[mouse2_dlc[i, 3].astype(
                int), mouse2_dlc[i, 2].astype(int)] = 1

        except:
            pass

        # -----------mouse 1---------------------

        if (np.sum(np.sum(nose1_DLC*mask1)) > 0) & (np.sum(np.sum(tail1_DLC*mask1)) > 0):

            mouse1_ensemble[i, 0:2] = mouse1_dlc[i, 0:2]
            mouse1_ensemble[i, 2:4] = mouse1_dlc[i, 2:4]
            flag1[i] = 1

        elif (np.sum(np.sum(nose2_DLC*mask1)) > 0) & (np.sum(np.sum(tail2_DLC*mask1)) > 0):
            mouse1_ensemble[i, 0:2] = mouse2_dlc[i, 0:2]
            mouse1_ensemble[i, 2:4] = mouse2_dlc[i, 2:4]
            flag1[i] = 1

        else:
            mouse1_ensemble[i, 0:2] = mouse1_md[i, 0:2]
            mouse1_ensemble[i, 2:4] = mouse1_md[i, 2:4]
            flag1[i] = 0

        # --------logic to fix swapping: ------------
        if i > 0:
            #if (flag1[i] == 0) & (flag1[i-1] == 1) & (components.loc[i, 'components'] == 2):
            if (flag1[i] == 0) & (flag1[i-1] == 1) & (components[i] == 2):
                mouse1_snout2snout = np.sum(
                    (mouse1_ensemble[i, 0:2]-mouse1_ensemble[i-1, 0:2]) ** 2)
                mouse1_snout2tail = np.sum(
                    (mouse1_ensemble[i, 0:2]-mouse1_ensemble[i-1, 2:4]) ** 2)
                if mouse1_snout2tail < mouse1_snout2snout:
                    temp1 = mouse1_ensemble[i, 0:2].copy()
                    mouse1_ensemble[i, 0:2] = mouse1_ensemble[i, 2:4]
                    mouse1_ensemble[i, 2:4] = temp1

        # --------mouse 2-------------------------
        if (np.sum(np.sum(nose1_DLC*mask2)) > 0) & (np.sum(np.sum(tail1_DLC*mask2)) > 0):

            mouse2_ensemble[i, 0:2] = mouse1_dlc[i, 0:2]
            mouse2_ensemble[i, 2:4] = mouse1_dlc[i, 2:4]
            flag2[i] = 1

        elif (np.sum(np.sum(nose2_DLC*mask2)) > 0) & (np.sum(np.sum(tail2_DLC*mask2)) > 0):
            mouse2_ensemble[i, 0:2] = mouse2_dlc[i, 0:2]
            mouse2_ensemble[i, 2:4] = mouse2_dlc[i, 2:4]
            flag2[i] = 1

        else:
            mouse2_ensemble[i, 0:2] = mouse2_md[i, 0:2]
            mouse2_ensemble[i, 2:4] = mouse2_md[i, 2:4]
            flag2[i] = 0

        # --------logic to fix swapping: ------------
        if i > 0:
            #if (flag2[i] == 0) & (flag2[i-1] == 1) & (components.loc[i, 'components'] == 2):
            if (flag2[i] == 0) & (flag2[i-1] == 1) & (components[i] == 2):
                mouse2_snout2snout = np.sum(
                    (mouse2_ensemble[i, 0:2]-mouse2_ensemble[i-1, 0:2]) ** 2)
                mouse2_snout2tail = np.sum(
                    (mouse2_ensemble[i, 0:2]-mouse2_ensemble[i-1, 2:4]) ** 2)
                if mouse2_snout2tail < mouse2_snout2snout:
                    temp2 = mouse2_ensemble[i, 0:2].copy()
                    mouse2_ensemble[i, 0:2] = mouse2_ensemble[i, 2:4]
                    mouse2_ensemble[i, 2:4] = temp2

    # mouse1_ensemble[:,1] = 540-mouse1_ensemble[:,1]
    # mouse1_ensemble[:,3] = 540-mouse1_ensemble[:,3]

    # mouse2_ensemble[:,1] = 540-mouse2_ensemble[:,1]
    # mouse2_ensemble[:,3] = 540-mouse2_ensemble[:,3]

    df_mouse1_ensemble = pd.DataFrame({'snout_x': mouse1_ensemble[:, 0],
                                       'snout_y': mouse1_ensemble[:, 1],
                                       'tailbase_x': mouse1_ensemble[:, 2],
                                       'tailbase_y': mouse1_ensemble[:, 3]})

    df_mouse2_ensemble = pd.DataFrame({'snout_x': mouse2_ensemble[:, 0],
                                       'snout_y': mouse2_ensemble[:, 1],
                                       'tailbase_x': mouse2_ensemble[:, 2],
                                       'tailbase_y': mouse2_ensemble[:, 3]})

    df_mouse1_ensemble.to_csv(os.path.join(os.path.dirname(frames_dir),
                                           'mouse1_ensemble.csv'), index=False)

    df_mouse2_ensemble.to_csv(os.path.join(os.path.dirname(frames_dir),
                                           'mouse2_ensemble.csv'), index=False)

    return df_mouse1_ensemble, df_mouse2_ensemble



def background_subtraction_single(frames_dir, fg_dir, background, threshold, frame_index):
    """Generate foregrounds corresponding to frames
    Args:
        frames_dir: path to directory containing frames
        fg_dir: path to save foreground
        background_dir: path to the background image
        threshold: np.array
        frame_index: int
    Returns:
        components: 1D array of number of blobs in each frame.
    """

    im = img_as_float(skimage.io.imread(
        os.path.join(frames_dir, str(frame_index) + '.jpg')))

    if im.ndim == 3:
        im = rgb2gray(im)

    fg = (background - im) > threshold
    bw1 = morphology.remove_small_objects(fg, 1000)

    bw2 = morphology.binary_closing(bw1, morphology.disk(radius=10))

    bw3 = bw2
    label = measure.label(bw3)
    num_fg = np.max(label)

    masks = np.zeros(
        [background.shape[0], background.shape[1], 3], dtype=np.uint8)

    if num_fg == 2:
        bw3_1 = label == 1
        bw4_1 = morphology.binary_dilation(bw3_1, morphology.disk(radius=3))

        bw3_2 = label == 2
        bw4_2 = morphology.binary_dilation(bw3_2, morphology.disk(radius=3))

        masks[:, :, 0] = img_as_ubyte(bw4_1)
        masks[:, :, 1] = img_as_ubyte(bw4_2)


    else:
        masks[:, :, 0] = img_as_ubyte(bw3)

    skimage.io.imsave(os.path.join(fg_dir, str(frame_index) + '.png'), masks)

    return num_fg


def background_subtraction_single_h5(frames_dir, background, threshold, frame_index):
    """Generate foregrounds corresponding to frames
    Args:
        frames_dir: path to directory containing frames
        fg_dir: path to save foreground
        background_dir: path to the background image
        threshold: np.array
        frame_index: int
    Returns:
        components: 1D array of number of blobs in each frame.
    """
    frame_dict = {}

    im = img_as_float(skimage.io.imread(
        os.path.join(frames_dir, str(frame_index) + '.jpg')))

    if im.ndim == 3:
        im = rgb2gray(im)

    fg = (background - im) > threshold
    bw1 = morphology.remove_small_objects(fg, 1000)

    bw2 = morphology.binary_closing(bw1, morphology.disk(radius=10))

    bw3 = bw2
    label = measure.label(bw3)
    num_fg = np.max(label)

    masks = np.zeros(
        [background.shape[0], background.shape[1], 3], dtype=np.uint8)

    if num_fg == 2:
        bw3_1 = label == 1
        bw4_1 = morphology.binary_dilation(bw3_1, morphology.disk(radius=3))

        bw3_2 = label == 2
        bw4_2 = morphology.binary_dilation(bw3_2, morphology.disk(radius=3))

        masks[:, :, 0] = img_as_ubyte(bw4_1)
        masks[:, :, 1] = img_as_ubyte(bw4_2)

        # frame_dict['mouse1'] = find_contours(img_as_ubyte(bw4_1), 0.5)[0].astype(int)
        # frame_dict['mouse2'] = find_contours(img_as_ubyte(bw4_2), 0.5)[0].astype(int)


        mouse1_boundary = find_contours(img_as_ubyte(bw4_1), 0.5)[0].astype(int)
        mouse1_temp = mouse1_boundary[:, 0].copy()
        mouse1_boundary[:, 0] = mouse1_boundary[:,1]
        mouse1_boundary[:, 1] = mouse1_temp

        mouse2_boundary = find_contours(img_as_ubyte(bw4_2), 0.5)[0].astype(int)
        mouse2_temp = mouse2_boundary[:, 0].copy()
        mouse2_boundary[:, 0] = mouse2_boundary[:,1]
        mouse2_boundary[:, 1] = mouse2_temp

        
        frame_dict['mouse1'] = mouse1_boundary
        frame_dict['mouse2'] = mouse2_boundary


    else:
        masks[:, :, 0] = img_as_ubyte(bw3)

    skimage.io.imsave(os.path.join(os.path.dirname(frames_dir), 'FG_parallel/'+str(frame_index) + '.png'), masks)

    return (frame_index, frame_dict, num_fg)
 


def background_subtraction_parallel(frames_dir, background_path, num_processors=None):
    """Generate foregrounds corresponding to frames
    Args:
        frames_dir: path to directory containing frames
        background_dir: path to the background image
        num_processors: int
    returns:
        components: 1D array of number of blobs in each frame.
    """

    fg_dir = os.path.join(os.path.dirname(frames_dir), 'FG')

    if not os.path.exists(fg_dir):
        os.mkdir(fg_dir)
    # clean_dir_safe(fg_dir)

    background = img_as_float(skimage.io.imread(background_path))
    if background.ndim == 3:
        background = rgb2gray(background)

    threshold = background * 0.5

    frames_list = os.listdir(frames_dir)

    p = Pool(processes=num_processors)
    output = p.starmap(background_subtraction_single, [(
        frames_dir, fg_dir, background, threshold, i) for i in range(0, len(frames_list))])


    return np.array(output)


def background_subtraction_parallel_h5(frames_dir, background_path, num_processors=None):
    """Generate foregrounds corresponding to frames
    Args:
        frames_dir: path to directory containing frames
        background_dir: path to the background image
        num_processors: int
    returns:
        components: 1D array of number of blobs in each frame.
    """
    video_dict = {}
    fg_dir = os.path.join(os.path.dirname(frames_dir), 'FG')

    if not os.path.exists(fg_dir):
        os.mkdir(fg_dir)
    # clean_dir_safe(fg_dir)

    background = img_as_float(skimage.io.imread(background_path))
    if background.ndim == 3:
        background = rgb2gray(background)

    threshold = background * 0.5

    frames_list = os.listdir(frames_dir)

    p = Pool(processes=num_processors)
    output = p.starmap(background_subtraction_single_h5, [(
        frames_dir, background, threshold, i) for i in range(0, len(frames_list))])

    num_fg_list =[]
    for (frame_index, frame_dict, num_fg) in output:
        video_dict[str(frame_index)] = frame_dict
        num_fg_list.append(num_fg)

    dd.io.save(os.path.join(os.path.dirname(frames_dir), 'masks_fg.h5'), video_dict, compression=None)
    
    return np.array(num_fg_list)



def behavior_feature_extraction(resident, intruder, tracking_dir, order=[1, 2]):
    '''Function to extract features for quantifying behavior
    Args:
        resident: coordinates of snout and tailbase of the resident (pd.DataFrame)
        intruder: coordinates of snout and tailbase of the intruder (pd.DataFrame)
        tracking_dir: directory of tracked masks (str)
        order: mask index of resident and intruder (List)
    Returns:
        df_features: resident to intruder features (pd.DataFrame)
    '''

    df_features = pd.DataFrame()

    for i in range(resident.shape[0]):
        df_features.loc[i, 'snout2snout'] = np.sqrt(np.sum((resident.loc[i, [
                                                    'snout_y', 'snout_x']].values-intruder.loc[i, ['snout_y', 'snout_x']].values) ** 2))
        df_features.loc[i, 'snout2tailbase'] = np.sqrt(np.sum((resident.loc[i, [
                                                       'snout_y', 'snout_x']].values-intruder.loc[i, ['tailbase_y', 'tailbase_x']].values) ** 2))

        masks = skimage.io.imread(os.path.join(
            tracking_dir, str(i) + '.png')) / 255.0
        mask_resident = masks[:, :, order[0]].astype(int)
        mask_intruder = masks[:, :, order[1]].astype(int)

        resident_border = find_contours(mask_resident, 0.5)[0]
        intruder_border = find_contours(mask_intruder, 0.5)[0]

        resident_snout = np.zeros(mask_resident.shape)
        resident_snout[int(resident.loc[i, 'snout_y']),
                       int(resident.loc[i, 'snout_x'])] = 1

        if np.sum(np.sum(resident_snout*mask_intruder)) > 0:
            df_features.loc[i, 'snout2body'] = 0
        else:
            df_features.loc[i, 'snout2body'] = np.min(np.sqrt(np.sum(
                (resident.loc[i, ['snout_y', 'snout_x']].values - intruder_border) ** 2, axis=1)))

        if np.sum(np.sum(mask_resident*mask_intruder)) > 0:
            df_features.loc[i, 'body2body'] = 0
        else:
            distance = np.zeros((len(resident_border), len(intruder_border)))

            for j in range(len(resident_border)):
                distance[j, :] = np.sqrt(
                    np.sum((resident_border[j] - intruder_border) ** 2, axis=1))

            df_features.loc[i, 'body2body'] = np.min(distance)

    df_features.to_csv(os.path.join(os.path.dirname(tracking_dir),
                                    'resident2intruder.csv'), index=False)

    return df_features


def boundary2mask(frame_dict, img_shape, num_mice=2):

    masks = np.zeros((img_shape[0], img_shape[1], num_mice), dtype=bool)


    for i in range(num_mice):
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        draw = PIL.ImageDraw.Draw(mask)

        mouse = frame_dict['mouse' + str(i+1)]

        xy = [tuple(point) for point in mouse]
    
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)

        mask = np.array(mask, dtype=bool)

        masks[:,:, i]= mask
    return masks


def shape_to_mask(img_shape, points, shape_type=None,
                  line_width=10, point_size=5):
    """Generate an instance mask from a set of points
    Args:
        img_shape: Size of mask (height, width)
        points: [[x1,y1],[x2,y2],....]
        shape_type: str ('circle', 'rectangle', 'line', 'linestrip', 'point')
        line_width: int
        point_size: int
    Returns: 
        mask: A bool array of shape [height, width]
    """

    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask