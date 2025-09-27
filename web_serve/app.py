import os
from storage_helper import download_file, upload_file


current_path = "/app/pythonAction"

def lambda_handler(event):
    blobName = event.get("input_file", "money.txt")
    download_file(blobName, f"{current_path}/{blobName}")
    
    moneyF = open(f"{current_path}/{blobName}", "r")
    money = float(moneyF.readline())
    moneyF.close()
    money -= 100.0
    new_file = open("moneyTemp"+str(os.getpid())+".txt", "w")
    new_file.write(str(money))
    new_file.close()

    upload_file(f"{current_path}/{blobName}", blobName)

    return {"Money":"withdrawn"}