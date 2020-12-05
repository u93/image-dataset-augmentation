# Image Data Augmentation 

## Goal

Be able to generate an image dataset with a desired image overlaid on top.

## How to run

- Move into the project root file

- Install base Python dependencies 
    - `pip install -r etc/local/requirements.txt`
    
- Set the paths for the INPUT and BACKGROUND images, DESTINATION directory and X & Y Axis iterations
    - Set ENVIRONMENT VARIABLES for the paths to the images, destination dirs, iterations
        - `export LOCAL_IMAGE_INPUT_SOURCE=${IMAGE_PATH}`
        - `export LOCAL_IMAGE_BACKGROUND=${IMAGE_PATH}`
        - `export LOCAL_DESTINATION_IMAGE_DIRECTORY=${DESTINATION_DIR_PATH}`
        - `export LOCAL_X_AXIS_ITERATIONS=${X_AXIS_ITERATIONS}`
        - `export LOCAL_Y_AXIS_ITERATIONS=${Y_AXIS_ITERATIONS}`
        
    - For default values, modify `src/settings/local.py` to have your desired input image path and background image path
        - LOCAL_IMAGE_INPUT_SOURCE
        - LOCAL_IMAGE_BACKGROUND
        - LOCAL_DESTINATION_IMAGE_DIRECTORY
        - LOCAL_X_AXIS_ITERATIONS
        - LOCAL_Y_AXIS_ITERATIONS
        
- Run the project applicaton 
    - `python src/app.py`