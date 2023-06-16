import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

from utils.mp_holistic import get_upper_landmarks
from utils.mp_image_segmenter import segment_image


def apply_mask_and_crop(image, masks):
    h, w = masks.shape[-2:]
    mask_image = masks.reshape(h, w, 1)

    mask_image = mask_image.astype(np.uint8) * 255

    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    image = cv2.bitwise_and(image, mask_image)

    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = np.concatenate(contours)

    (x, y, w, h) = cv2.boundingRect(contours)

    image = image[y : y + h, x : x + w]

    return image


def mask_image(images):
    masked_images = []

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    for image in images:
        predictor.set_image(image)

        upper_landmarks = get_upper_landmarks(image)
        if upper_landmarks:
            input_point = np.array(upper_landmarks)
            input_label = np.array([1] * len(upper_landmarks))

            mask, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

        else:
            mask = segment_image(image)

        masked_image = apply_mask_and_crop(image, mask)
        masked_images.append(masked_image)

    return masked_images
