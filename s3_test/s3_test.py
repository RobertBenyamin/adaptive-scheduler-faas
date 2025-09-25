import boto3
import os


# make s3 client, uncomment the following lines to use your own credentials
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN')
)

# Nama bucket dan nama file
bucket_name = 'faas-scheduler-data'
object_key = 'img10.jpg'
local_filename = 'img10.png'

# Download file dari S3
s3_client.download_file(bucket_name, object_key, local_filename)

print(f"File {object_key} berhasil diunduh ke {local_filename}")
