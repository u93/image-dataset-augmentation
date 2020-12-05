import uuid

from src.components.cv_roi_extraction import roi_processing, roi_selection
from src.common.image import overlay_image, read_static_image, save_image
from src.settings.local import LOCAL_IMAGE_BACKGROUND


def frame_extraction_overlay(input_image_path=None, background_image_path=LOCAL_IMAGE_BACKGROUND):
    # Generate the operation unique ID, will be used for the image's name.
    operation_uuid = str(uuid.uuid4())

    # Reads the desired base image and the background image and returns as Numpy arrays.
    input_image = read_static_image(filename=input_image_path)
    background_image = read_static_image(filename=background_image_path)

    # Allows selection of a polygon on the input image for later overlay.
    points = roi_selection(numpy_frame=input_image)
    desired_images = roi_processing(numpy_frame=input_image, roi_points=points)

    # Overlay and save generated images in the default directory specified in settings.
    modified_background_images = overlay_image(front_images=desired_images, background_image=background_image)
    for index, image in enumerate(modified_background_images):
        save_image(numpy_frame=image, operation_uuid=operation_uuid, index=index)


def main():
    frame_extraction_overlay()


if __name__ == "__main__":
    main()
