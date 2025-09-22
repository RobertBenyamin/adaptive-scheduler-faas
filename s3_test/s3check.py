import boto3

# make s3 client, uncomment the following lines to use your own credentials
# s3_client = boto3.client()

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