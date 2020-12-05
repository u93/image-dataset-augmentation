import cv2
import numpy as np
import time

from src.tests.common.image import overlay_image, read_static_image, save_image, view_image, CoordinateStore
from src.settings.local import LOCAL_X_AXIS_ITERATIONS, LOCAL_Y_AXIS_ITERATIONS, TEST_ROI_POINTS


def roi_extraction(roi_points: list, numpy_frame: np.ndarray):
    desired_images = list()
    if len(roi_points) == 0:
        raise RuntimeError

    points = np.array(roi_points, dtype=np.int32)
    (mean_x, mean_y) = points.mean(axis=0)
    print(mean_x, mean_y)

    image = numpy_frame

    # extracted_out
    mask = np.ones((image.shape[0], image.shape[1]))

    cv2.fillConvexPoly(mask, points, 0)
    mask = mask.astype(np.bool)

    extracted_out = np.ones_like(image)
    extracted_out[mask] = image[mask]

    # masked_out
    mask = np.zeros((image.shape[0], image.shape[1]))

    cv2.fillConvexPoly(mask, points, 1)
    mask = mask.astype(np.bool)

    masked_out = np.zeros_like(image)
    masked_out[mask] = image[mask]
    base_masked_out = masked_out

    x_intervals = round(numpy_frame.shape[1] / 20)
    y_intervals = round(numpy_frame.shape[0] / 30)
    x_start_interval = 0
    y_start_interval = 0

    for x_index in range(0, LOCAL_X_AXIS_ITERATIONS):
        print(x_start_interval, y_start_interval)
        masked_out = np.roll(base_masked_out, (x_start_interval, 0), axis=(1, 0))  # Add shifting capabilities to the image
        desired_images.append(masked_out)

        frame_mask_scan(masked_out)

        for y_index in range(0, LOCAL_Y_AXIS_ITERATIONS):
            print(x_start_interval, y_start_interval)
            masked_out = np.roll(base_masked_out, (x_start_interval, y_start_interval), axis=(0, 1))   # Add shifting capabilities to the image
            desired_images.append(masked_out)
            y_start_interval = y_start_interval + y_intervals

        x_start_interval = x_start_interval + x_intervals
        desired_images.append(masked_out)

    return extracted_out, masked_out, desired_images


def test_roi_extraction(numpy_frame: np.ndarray):
    image = numpy_frame
    points = np.array(TEST_ROI_POINTS, dtype=np.int32)

    mask = np.zeros((image.shape[0], image.shape[1]))

    cv2.fillConvexPoly(mask, points, 1)
    mask = mask.astype(np.bool)

    out = np.zeros_like(image)
    out[mask] = image[mask]

    view_image(out)


def roi_selection(numpy_frame: np.ndarray):
    image = numpy_frame
    coordinate_handler = CoordinateStore(numpy_frame=image)

    cv2.namedWindow("Test Image")
    cv2.setMouseCallback("Test Image", coordinate_handler.select_point)
    # cv2.imshow("Test Image", image)

    while True:
        cv2.imshow("Test Image", image)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break

    print("Selected Coordinates: ")
    print(coordinate_handler.points)

    return coordinate_handler.points


def frame_extraction(image_path=None):
    input_image = read_static_image(filename=image_path)

    points = roi_selection(numpy_frame=input_image)
    extracted_image, desired_image, desired_images = roi_extraction(roi_points=points, numpy_frame=input_image)

    view_image(desired_image)


def frame_mask_scan(numpy_frame: np.ndarray):
    current_value = None
    previous_value = None
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

    current_value = None
    previous_value = None
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

    print(start_x, end_x, start_y, end_y)
    roi_centroid = (round((start_x + end_x) / 2), round((start_y + end_y) / 2))
    print(roi_centroid)
    
    return start_x, end_x, start_y, end_y
