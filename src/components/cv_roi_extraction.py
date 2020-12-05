import sys

import cv2
import numpy as np

from src.common.image import CoordinateStore
from src.settings.local import LOCAL_X_AXIS_ITERATIONS, LOCAL_Y_AXIS_ITERATIONS


def roi_selection(numpy_frame: np.ndarray):
    """
    Generates the Region of Interest points selection for posterior processing.
    :param numpy_frame: Input image decided in the project settings read as a Numpy frame.
    :return: Points list.
    """
    image = numpy_frame
    coordinate_handler = CoordinateStore(numpy_frame=image)

    cv2.namedWindow("Test Image")
    cv2.setMouseCallback("Test Image", coordinate_handler.select_point)

    while True:
        cv2.imshow("Test Image", image)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break

    print("Selected Coordinates: ")
    print(coordinate_handler.points)

    return coordinate_handler.points


def roi_processing(numpy_frame: np.ndarray, roi_points: list):
    """
    Processing of the Region of Interest points on the input image and overlays it on the background image by rotating it
    across the X-Axis and Y-Axis.
    :param numpy_frame: Input image decided in the project settings read as a Numpy frame.
    :param roi_points: Points list.
    :return: Generated masks with the Region of Interest rotated.
    """
    final_images = list()
    if len(roi_points) == 0:
        print("Incorrect number of Region of Interest points selected... Exiting")
        sys.exit(1)

    base_masked_out = roi_extraction(roi_points=roi_points, numpy_frame=numpy_frame)

    x_intervals = round(numpy_frame.shape[1] / 20)
    y_intervals = round(numpy_frame.shape[0] / 30)
    x_start_interval = 0
    y_start_interval = 0

    for x_index in range(0, LOCAL_X_AXIS_ITERATIONS):
        x_start_interval, generated_images = roi_rotation(
            masked_image=base_masked_out,
            x_start_interval=x_start_interval,
            y_start_interval=y_start_interval,
            x_intervals=x_intervals,
            y_intervals=y_intervals
        )
        final_images.extend(generated_images)

    return final_images


def roi_extraction(roi_points: list, numpy_frame: np.ndarray):
    points = np.array(roi_points, dtype=np.int32)
    image = numpy_frame

    # Masked image generation
    mask = np.zeros((image.shape[0], image.shape[1]))

    cv2.fillConvexPoly(mask, points, 1)
    mask = mask.astype(np.bool)

    masked_out = np.zeros_like(image)
    masked_out[mask] = image[mask]
    base_masked_out = masked_out

    return base_masked_out


def roi_rotation(masked_image, x_start_interval, y_start_interval, x_intervals, y_intervals):
    images = list()

    masked_out = np.roll(masked_image, (x_start_interval, 0), axis=(1, 0))  # Add shifting capabilities to the image
    images.append(masked_out)

    frame_mask_scan(masked_out)

    for y_index in range(0, LOCAL_Y_AXIS_ITERATIONS):
        # Add shifting capabilities to the image
        masked_out = np.roll(
            masked_image, (x_start_interval, y_start_interval), axis=(0, 1)
        )
        images.append(masked_out)
        y_start_interval = y_start_interval + y_intervals

    x_start_interval = x_start_interval + x_intervals
    images.append(masked_out)

    return x_start_interval, images


def frame_mask_scan(numpy_frame: np.ndarray):
    threshold_value = 0

    # Columns 0 -> Last
    column_indexes = list()
    limit = numpy_frame.shape[1]
    for i in range(0, limit):
        current_value = np.count_nonzero(numpy_frame[:, i])
        if current_value != 0:
            column_indexes.append(i)
        previous_value = current_value
        if len(column_indexes) > 0 and previous_value == threshold_value and previous_value == threshold_value:
            break

    start_x = column_indexes[1]
    end_x = column_indexes[-1]

    threshold_value = 0

    # Rows 0 -> Last
    row_indexes = list()
    limit = numpy_frame.shape[0]
    for i in range(0, limit):
        current_value = np.count_nonzero(numpy_frame[i, :])
        if current_value != 0:
            row_indexes.append(i)
        previous_value = current_value
        if len(row_indexes) > 0 and previous_value == threshold_value and previous_value == threshold_value:
            break

    start_y = row_indexes[1]
    end_y = row_indexes[-1]

    # roi_centroid = (round((start_x + end_x) / 2), round((start_y + end_y) / 2))
    
    return start_x, end_x, start_y, end_y
