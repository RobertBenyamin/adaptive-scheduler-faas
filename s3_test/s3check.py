import boto3
import os

# make s3 client, uncomment the following lines to use your own credentials
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

# List semua bucket
response = s3_client.list_buckets()
for bucket in response['Buckets']:
    print(f"Nama bucket : {bucket['Name']}")

    # 1. List semua object dalam bucket
    response = s3_client.list_objects_v2(Bucket=bucket['Name'])

    # Periksa apakah bucket memiliki object
    if 'Contents' in response:
        for obj in response['Contents']:
            object_key = obj['Key']
            print(f"File: {object_key}")
