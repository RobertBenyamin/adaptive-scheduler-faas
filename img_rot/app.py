import os
from PIL import Image
import dnld_blob
from storage_helper import download_file, upload_file

current_path = "/app/pythonAction"

fileAppend = open("../funcs.txt", "a")

def lambda_handler():
    blobName = "img10.jpg"
    # dnld_blob.download_blob_new(blobName)
    # full_blob_name = blobName.split(".")
    # proc_blob_name = full_blob_name[0] + "_" + str(os.getpid()) + "." + full_blob_name[1]
    download_file(blobName, f"{current_path}/{blobName}")

    image = Image.open(f"{current_path}/{blobName}")
    img = image.transpose(Image.ROTATE_90)
    img.save('tempImage_'+str(os.getpid())+'.jpeg')

    fReadname = 'tempImage_'+str(os.getpid())+'.jpeg'
    blobName = "img10_rot.jpg"
    # dnld_blob.upload_blob_new(blobName, fReadname)
    upload_file(f"{current_path}/{blobName}", blobName)
    
    return {"Image":"rotated"}