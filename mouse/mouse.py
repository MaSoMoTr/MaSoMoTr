from .shape import shapes_to_labels_masks
from mrcnn import utils
from mrcnn.config import Config
import os
import sys
import json
import numpy as np

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library


class MouseConfig(Config):
    """Configuration for training Mice Tracking  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "mouse"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + mouse + brown_mouse

    # Number of training steps per epoch
    STEPS_PER_EPOCH = int(200 // IMAGES_PER_GPU)  #
    VALIDATION_STEPS = max(1, 40 // IMAGES_PER_GPU)  #

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9  #

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    LEARNING_RATE = 0.0005  # 0.00134

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512

    IMAGE_MIN_SCALE = 0

    IMAGE_CHANNEL_COUNT = 3  # -----------

    # Image mean (RGB)
    MEAN_PIXEL = np.array([128, 128, 128])


# Dataset
class MouseDataset(utils.Dataset):
    """Loading the dataset annotated by Labelme
    Structure of dataset: dataset_dir/train/ -> a.jpg, b.jpg, c.jpg, ...
                                             -> a.json, b.json, c.json
                          dataset_dir/val/   -> d.jpg, e.jpg, ...
                                             -> d.json, e.json, ...
    """

    def load_mouse(self, dataset_dir, subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: "train" or "val"
        """
        # Add classes. We have only one class to add.
        self.add_class("mouse", 1, "mouse")

        # Train or validation dataset?
        assert subset in ["train", "val", ""]
        dataset_dir = os.path.join(dataset_dir, subset)
        image_ids = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
        for image_id, image_name in enumerate(image_ids):
            # if image_id.endswith(".jpg"):
            self.add_image("mouse",
                           image_id=image_id,
                           path=os.path.join(dataset_dir, image_name))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]

        # Get mask directory from image path
        mask_dir = os.path.splitext(info['path'])[0] + '.json'
        class_name_to_id = {label["name"]: label["id"]
                            for label in self.class_info}

        # Read mask file from json
        with open(mask_dir) as f:
            data = json.load(f)
            image_shape = (data['imageHeight'], data['imageWidth'])

            cls, masks = shapes_to_labels_masks(img_shape=image_shape,
                                                shapes=data['shapes'],
                                                label_name_to_value=class_name_to_id)
        return masks, cls

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "mouse":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class InferenceConfig(MouseConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
