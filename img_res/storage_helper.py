import boto3

# Nama bucket di S3
BUCKET_NAME = "mxfaas-prod-1"

# make s3 client, uncomment the following lines to use your own credentials
# s3_client = boto3.client()

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

def upload_file(source_file_name, destination_blob_name):
    """
    Mengunggah file dari lokal ke S3.
    
    :param source_file_name: Path file lokal
    :param destination_blob_name: Path tujuan di S3
    """

    try:
        # Unggah file ke S3
        s3_client.upload_file(source_file_name, BUCKET_NAME, destination_blob_name)
        print(f"File {source_file_name} uploaded to {destination_blob_name}.")
    except Exception as e:
        print(f"Error uploading file: {e}")