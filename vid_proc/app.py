import os
import time
import cv2
from storage_helper import download_file, upload_file
from dnld_blob import download_blob_new

TMP_DIR = "/tmp/"

def lambda_handler(event):
    input_filename = event.get("input_file", "vid1.mp4")
    pid = str(os.getpid())

    base = os.path.basename(input_filename)
    name, ext = os.path.splitext(base)
    proc_blob_name = f"{name}_{pid}{ext}"
    local_file_path = os.path.join(TMP_DIR, proc_blob_name)
    local_output_path = os.path.join(TMP_DIR, f"output_{pid}.avi")

    try:
        download_blob_new(input_filename)
        timeout = 5.0
        t0 = time.time()
        while not os.path.exists(local_file_path) and (time.time() - t0) < timeout:
            time.sleep(0.05)
        if not os.path.exists(local_file_path):
            download_file(input_filename, local_file_path)
    except Exception:
        download_file(input_filename, local_file_path)

    video = cv2.VideoCapture(local_file_path)
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

    output_filename = f"processed_{pid}_{name}.avi"
    upload_file(local_output_path, output_filename)

    try:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        if os.path.exists(local_output_path):
            os.remove(local_output_path)
    except:
        pass

    return {"Video": "Done", "output_file": output_filename}