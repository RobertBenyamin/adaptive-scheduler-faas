import os
import cv2
from storage_helper import download_file, upload_file

TMP_DIR = "/tmp/"

def lambda_handler(event):
    input_filename = event.get("input_file", "vid1.mp4")
    pid = str(os.getpid())

    local_input_path = os.path.join(TMP_DIR, f"input_{pid}_{input_filename}")
    local_output_path = os.path.join(TMP_DIR, f"output_{pid}.avi")
    
    download_file(input_filename, local_input_path)
    video = cv2.VideoCapture(local_input_path)
    if not video.isOpened():
        return {"Error": "Could not open video file"}

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS) or 20.0

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(local_output_path, fourcc, fps, (width, height), isColor=False)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray_frame)

    video.release()
    out.release()

    output_blob_name = f"processed_{pid}.avi"
    upload_file(local_output_path, output_blob_name)

    if os.path.exists(local_input_path): os.remove(local_input_path)
    if os.path.exists(local_output_path): os.remove(local_output_path)

    return {"Video": "Done"}