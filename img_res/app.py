import os
from PIL import Image
from storage_helper import download_file, upload_file

current_path = "/app/pythonAction"

def lambda_handler(event):
    blobName = event.get("input_file", "img10.jpg")
    download_file(blobName, f"{current_path}/{blobName}")
    
    image = Image.open(f"{current_path}/{blobName}")
    width, height = image.size
    # Setting the points for cropped image
    left = 4
    top = height / 5
    right = 100
    bottom = 3 * height / 5
    im1 = image.crop((left, top, right, bottom))
    im1.save('tempImage_'+str(os.getpid())+'.jpeg')

    blobName = "img10_res.jpg"
    upload_file(f"{current_path}/{blobName}", blobName)

    return {"Image":"resized"}