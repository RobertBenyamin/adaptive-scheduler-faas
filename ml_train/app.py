import os
import time
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re
import warnings
from storage_helper import download_file, upload_file
from dnld_blob import download_blob_new

warnings.filterwarnings("ignore")

cleanup_re = re.compile('[^a-z]+')

TMP_DIR = "/tmp/"

def cleanup(sentence):
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    return sentence

df_name = 'minioDataset.csv'

def lambda_handler(event):
    t1 = time.time()
    blobName = event.get("input_file", df_name)
    pid = str(os.getpid())

    base = os.path.basename(blobName)
    name, ext = os.path.splitext(base)
    proc_blob_name = f"{name}_{pid}{ext}"
    local_file_path = os.path.join(TMP_DIR, proc_blob_name)

    try:
        download_blob_new(blobName)
        timeout = 10.0
        t0 = time.time()
        while not os.path.exists(local_file_path) and (time.time() - t0) < timeout:
            time.sleep(0.05)
        if not os.path.exists(local_file_path):
            download_file(blobName, local_file_path)
    except Exception:
        download_file(blobName, local_file_path)
    
    t2 = time.time()
    print("Time 1 = " + str(t2-t1))

    df = pd.read_csv(local_file_path)
    df['train'] = df['Text'].apply(cleanup)

    model = LogisticRegression(max_iter=10)
    tfidf_vector = TfidfVectorizer(min_df=1000).fit(df['train'])
    train = tfidf_vector.transform(df['train'])
    model.fit(train, df['Score'])
    t3 = time.time()
    print("Time 2 = " + str(t3-t2))

    output_filename = f'finalized_model_{pid}_{name}.sav'
    local_output_path = os.path.join(TMP_DIR, output_filename)

    with open(local_output_path, 'wb') as f:
        pickle.dump(model, f)

    upload_file(local_output_path, output_filename)
    t4 = time.time()
    print("Time 3 = " + str(t4-t3))

    try:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        if os.path.exists(local_output_path):
            os.remove(local_output_path)
    except:
        pass

    return {"ML": "Trained", "output_file": output_filename}