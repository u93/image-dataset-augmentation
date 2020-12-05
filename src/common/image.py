import math
import time
import uuid

import cv2
import numpy as np

from src.settings.local import (
    LOCAL_DESTINATION_IMAGE_DIRECTORY,
    LOCAL_SOURCE_IMAGE_DIRECTORY,
    LOCAL_IMAGE_INPUT_SOURCE
)


def overlay_image(front_images: list, background_image: np.ndarray):
    final_images = list()
    for image in front_images:
        print("Background image:")
        background_height, background_width, background_channels = background_image.shape
        print(background_height, background_width, background_channels)

        print("TARGET_PIXEL_AREA:")
        TARGET_PIXEL_AREA = (background_height * background_width)
        print(TARGET_PIXEL_AREA)

        print("Front image:")
        front_height, front_width, front_channels = image.shape
        print(front_height, front_width, front_channels)

        print("Front image ratio:")
        ratio = float(front_width) / float(front_height)
        print(ratio)

        print("Front image new height, width:")
        s_new_h = int(math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
        s_new_w = int((s_new_h * ratio) + 0.5)
        print(s_new_h, s_new_w)

        front_image = cv2.resize(image, (background_width, background_height))

        x_offset = 0
        y_offset = 0

        print("Height offset:")
        y1, y2 = y_offset, y_offset + front_image.shape[0]
        print(y1, y2)

        print("Width offset:")
        x1, x2 = x_offset, x_offset + front_image.shape[1]
        print(x1, x2)

        print("Front image resize:")
        front_height, front_width, front_channels = front_image.shape
        print(front_height, front_width, front_channels)

        print("Background image resize:")
        background_height, background_width, background_channels = background_image.shape
        print(background_height, background_width, background_channels)

        print("Final Background image resize:")
        background_height, background_width, background_channels = background_image.shape
        print(background_height, background_width, background_channels)

        overlay_mask = (front_image == 0)  # TODO: RETURN FOR FRONT IMAGE ALSO A FRAME WITH FILLED VALUES AS DIFFERENT THAN 0
        overlayed_image = np.copy(front_image)
        overlayed_image[overlay_mask] = background_image[overlay_mask]

        final_images.append(overlayed_image)

    return final_images


def read_static_image(filename=None) -> np.ndarray:
    if filename is not None:
        image_path = f"{LOCAL_SOURCE_IMAGE_DIRECTORY}/{filename}"
    else:
        image_path = f"{LOCAL_SOURCE_IMAGE_DIRECTORY}/{LOCAL_IMAGE_INPUT_SOURCE}"

    numpy_frame = cv2.imread(filename=image_path)
    print(f"Height: {numpy_frame.shape[0]}, Width: {numpy_frame.shape[1]}, Channels: {numpy_frame.shape[2]}")

    return numpy_frame


def save_image(numpy_frame: np.ndarray, index: int, operation_uuid=None):
    if operation_uuid is None:
        operation_uuid = str(uuid.uuid4())
    filename = f"{LOCAL_DESTINATION_IMAGE_DIRECTORY}/{operation_uuid}-{index}.jpg"

    print(f"Saving {filename}...")
    cv2.imwrite(filename=filename, img=numpy_frame)


def view_image(numpy_frame: np.ndarray):
    cv2.imshow("Test Image", numpy_frame)
    while True:
        time.sleep(0.1)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break


class CoordinateStore:
    def __init__(self, numpy_frame: np.ndarray):
        self.points = []
        self.numpy_frame = numpy_frame

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(self.numpy_frame, (x, y), 3, (255, 0, 0), 2)
            self.points.append([x, y])
