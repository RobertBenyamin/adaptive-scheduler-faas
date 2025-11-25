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
    img = image.transpose(Image.ROTATE_90)

    output_filename = f"rotated-{input_filename}"
    local_output_path = os.path.join(TMP_DIR, output_filename)

    img.save(local_output_path)

    upload_file(local_output_path, output_filename)
    
    return {"Image":"rotated"}