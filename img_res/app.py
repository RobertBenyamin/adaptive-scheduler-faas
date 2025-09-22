import os
from PIL import Image
from storage_helper import download_file, upload_file
import dnld_blob

current_path = "/app/pythonAction"

def lambda_handler():
    blobName = "img10.jpg"
    # dnld_blob.download_blob_new(blobName)
    # full_blob_name = blobName.split(".")
    # proc_blob_name = full_blob_name[0] + "_" + str(os.getpid()) + "." + full_blob_name[1]
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

    fReadname = 'tempImage_'+str(os.getpid())+'.jpeg'
    blobName = "img10_res.jpg"
    # dnld_blob.upload_blob_new(blobName, fReadname)
    upload_file(f"{current_path}/{blobName}", blobName)

    return {"Image":"resized"}