import os
import time
from PIL import Image
from storage_helper import download_file, upload_file
from dnld_blob import download_blob_new

TMP_DIR = "/tmp/"

def lambda_handler(event):
    input_filename = event.get("input_file", "img10.jpg")
    pid = str(os.getpid())

    base = os.path.basename(input_filename)
    name, ext = os.path.splitext(base)
    proc_blob_name = f"{name}_{pid}{ext}"
    local_file_path = os.path.join(TMP_DIR, proc_blob_name)

    try:
        download_blob_new(input_filename)
        timeout = 2.0
        t0 = time.time()
        while not os.path.exists(local_file_path) and (time.time() - t0) < timeout:
            time.sleep(0.01)
        if not os.path.exists(local_file_path):
            download_file(input_filename, local_file_path)
    except Exception:
        download_file(input_filename, local_file_path)
    
    image = Image.open(local_file_path)

    width, height = image.size
    # Setting the points for cropped image
    left = 4
    top = height / 5
    right = 100
    bottom = 3 * height / 5
    im1 = image.crop((left, top, right, bottom))

    output_filename = f"resized-{pid}-{input_filename}" 
    local_output_path = os.path.join(TMP_DIR, output_filename)

    im1.save(local_output_path)

    # upload_file(local_output_path, output_filename)

    try:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        if os.path.exists(local_output_path):
            os.remove(local_output_path)
    except:
        pass

    return {"Image":"resized", "output_file": output_filename}