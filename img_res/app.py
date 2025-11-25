import os
from PIL import Image
from storage_helper import download_file, upload_file

TMP_DIR = "/tmp/"

def lambda_handler(event):
    input_filename = event.get("input_file", "img10.jpg")
    pid = str(os.getpid())
    local_input_path = os.path.join(TMP_DIR, f"{pid}_{input_filename}")
    
    download_file(input_filename, local_input_path)
    
    image = Image.open(local_input_path)

    width, height = image.size
    # Setting the points for cropped image
    left = 4
    top = height / 5
    right = 100
    bottom = 3 * height / 5
    im1 = image.crop((left, top, right, bottom))

    output_filename = f"resized-{input_filename}" 
    local_output_path = os.path.join(TMP_DIR, output_filename)

    im1.save(local_output_path)

    upload_file(local_output_path, output_filename)

    return {"Image":"resized"}