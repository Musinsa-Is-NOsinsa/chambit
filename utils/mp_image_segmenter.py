import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(model_asset_path="selfie_multiclass_256x256.tflite")
options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

# Create the image segmenter
segmenter = vision.ImageSegmenter.create_from_options(options)


def segment_image(image):
    # Load the input image from a numpy array.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Retrieve the masks for the segmented image
    segmentation_result = segmenter.segment(mp_image)
    category_mask = segmentation_result.category_mask

    condition = category_mask.numpy_view() > 0.2
    mask = np.where(condition, 1, 0)

    return mask
