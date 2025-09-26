import os
import time
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re
import warnings
from storage_helper import download_file, upload_file

warnings.filterwarnings("ignore")

cleanup_re = re.compile('[^a-z]+')

current_path = "/app/pythonAction"


def cleanup(sentence):
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    return sentence

df_name = 'minioDataset.csv'

def lambda_handler(event):

    t1 = time.time()
    blobName = event.get("input_file", df_name)
    download_file(blobName, f"{current_path}/{blobName}")
    
    t2 = time.time()
    print("Time 1 = " + str(t2-t1))

    df = pd.read_csv(f"{current_path}/{blobName}")
    df['train'] = df['Text'].apply(cleanup)

    model = LogisticRegression(max_iter=10)
    tfidf_vector = TfidfVectorizer(min_df=1000).fit(df['train'])
    train = tfidf_vector.transform(df['train'])
    model.fit(train, df['Score'])
    t3 = time.time()
    print("Time 2 = " + str(t3-t2))

    filename = 'finalized_model_'+str(os.getpid())+'.sav'
    pickle.dump(model, open(filename, 'wb'))

    blobName = 'finalized_model_'+str(os.getpid())+'.sav'
    upload_file(f"{current_path}/{blobName}", blobName)
    t4 = time.time()
    print("Time 3 = " + str(t4-t3))

    return {"Ok":"done"}