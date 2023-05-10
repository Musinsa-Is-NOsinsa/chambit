import os

import _init_paths
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

from utils.mp_holistic import get_upper_landmarks


def mask_data(image, masks):
    h, w = masks.shape[-2:]
    mask_image = masks.reshape(h, w, 1)

    mask_image = mask_image.astype(np.uint8) * 255

    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    image = cv2.bitwise_and(image, mask_image)

    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = np.concatenate(contours)

    (x, y, w, h) = cv2.boundingRect(contours)

    image = image[y : y + h, x : x + w]

    return image


def main():
    sam_checkpoint = "pretrained_models\segment_anything\sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    for (path, dir, files) in os.walk("data"):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == ".jpg":
                filepath = os.path.join(path, filename)
                print(f"Processing '{filepath}'...")
                image = cv2.imread(filepath)

                predictor.set_image(image)

                upper_landmarks = get_upper_landmarks(image)
                if upper_landmarks:
                    input_point = np.array(upper_landmarks)
                    input_label = np.array([1] * len(upper_landmarks))

                    masks, _, _ = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=False,
                    )

                    image = mask_data(image, masks)

                    prefix = "mask_"
                    dir = os.path.dirname(prefix + filepath)
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                    cv2.imwrite(prefix + filepath, image)

    print("Done!")


if __name__ == "__main__":
    main()
