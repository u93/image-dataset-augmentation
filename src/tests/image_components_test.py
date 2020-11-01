import uuid

from src.settings.local import LOCAL_IMAGE_BACKGROUND
from src.tests.components.cv_roi_extraction import (
    overlay_image,
    read_static_image,
    roi_extraction,
    roi_selection,
    save_image,
    view_image
)


def test_frame_extraction_overlay(input_image_path=None, background_image_path=None):
    operation_uuid = str(uuid.uuid4())

    input_image = read_static_image(filename=input_image_path)
    background_image = read_static_image(filename=LOCAL_IMAGE_BACKGROUND)

    points = roi_selection(numpy_frame=input_image)
    extracted_image, desired_image, desired_images = roi_extraction(roi_points=points, numpy_frame=input_image)
    # for image in desired_images:
    #     pass
    #     view_image(numpy_frame=image)

    modified_background_images = overlay_image(front_images=desired_images, background_image=background_image)
    for index, image in enumerate(modified_background_images):
        # view_image(numpy_frame=image)
        save_image(numpy_frame=image, operation_uuid=operation_uuid, index=index)


if __name__ == "__main__":
    test_frame_extraction_overlay()
