import os
import cv2
from storage_helper import download_file, upload_file

tmp = "/tmp/"
current_path = "/app/pythonAction"

vid_name = 'vid1.mp4'

result_file_path = tmp + vid_name

def lambda_handler(event):
    blobName = event.get("input_file", "vid1.mp4")
    # dnld_blob.download_blob_new(blobName)
    download_file(blobName, f"{current_path}/{blobName}")
    video = cv2.VideoCapture("vid1_"+str(os.getpid())+".mp4")

    width = int(video.get(3))
    height = int(video.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter('output_'+str(os.getpid())+'.avi',fourcc, 20.0, (width, height))

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tmp_file_path = tmp+'tmp'+str(os.getpid())+'.jpg'
            cv2.imwrite(tmp_file_path, gray_frame)
            gray_frame = cv2.imread(tmp_file_path)
            out.write(gray_frame)
            break
        else:
            break

    blobName = 'output_'+str(os.getpid())+'.avi'
    upload_file(f"{current_path}/{blobName}", blobName)

    video.release()
    out.release()

    return {"Video": "Done"}