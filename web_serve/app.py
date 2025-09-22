import os
import dnld_blob
from storage_helper import download_file, upload_file


current_path = "/app/pythonAction"

def lambda_handler():
    blobName = "money.txt"
    # dnld_blob.download_blob_new(blobName)
    download_file(blobName, f"{current_path}/{blobName}")
    
    moneyF = open(f"{current_path}/{blobName}", "r")
    money = float(moneyF.readline())
    moneyF.close()
    money -= 100.0
    new_file = open("moneyTemp"+str(os.getpid())+".txt", "w")
    new_file.write(str(money))
    new_file.close()
    fReadname = "moneyTemp"+str(os.getpid())+".txt" 
    blobName = "money.txt"
    # dnld_blob.upload_blob_new(blobName, fReadname)
    upload_file(f"{current_path}/{blobName}", blobName)

    return {"Money":"withdrawn"}