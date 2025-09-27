import os
from PIL import Image
from storage_helper import download_file, upload_file

current_path = "/app/pythonAction"

fileAppend = open("../funcs.txt", "a")

def lambda_handler(event):
    blobName = event.get("input_file", "img10.jpg")
    download_file(blobName, f"{current_path}/{blobName}")

    image = Image.open(f"{current_path}/{blobName}")
    img = image.transpose(Image.ROTATE_90)
    img.save('tempImage_'+str(os.getpid())+'.jpeg')

    blobName = "rotated-"+blobName
    upload_file(f"{current_path}/{blobName}", blobName)
    
    return {"Image":"rotated"}