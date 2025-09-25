import boto3
import os


def download_file(object_name, local_filename):
    """
    Mengunduh file dari S3 ke local.

    :param object_name: Nama objek (path di S3)
    :param local_filename: Nama file lokal tempat menyimpan unduhan
    """

    # Path objek di S3
    # object_s3_name = f"assets/{object_name}"

    try:
        # Unduh file dari S3 ke local
        s3_client.download_file(BUCKET_NAME, object_name, local_filename)
        print(f"File downloaded successfully into {local_filename}")
    except Exception as e:
        print(f"Error downloading file: {e}")


# make s3 client, uncomment the following lines to use your own credentials
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)


# Nama bucket dan nama file
BUCKET_NAME = 'faas-scheduler-data'
object_name = 'img10.png'
local_filename = '/home/ec2-user/KNative_prototype/img10.png'

# Download file dari S3
download_file(object_name=object_name, local_filename=local_filename)

print(f"File {object_name} berhasil diunduh ke {local_filename}")
