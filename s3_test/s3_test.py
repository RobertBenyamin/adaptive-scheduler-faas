import boto3


# make s3 client, uncomment the following lines to use your own credentials
# s3_client = boto3.client()

# Nama bucket dan nama file
bucket_name = 'mxfaas'
object_key = 'img10.jpg'
local_filename = 'img10.png'

# Download file dari S3
s3_client.download_file(bucket_name, object_key, local_filename)

print(f"File {object_key} berhasil diunduh ke {local_filename}")