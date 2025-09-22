from google.cloud import storage


def download_file(object_name, local_filename): 
    # Define the bucket and object
    bucket_name = "mxfaas-storage"
    object_gcs_name = f"assets/{object_name}"

    # Instantiates a client
    client = storage.Client()

    # Download the object and handle exceptions
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_gcs_name)

        # Download to a local file with the same name as the object
        blob.download_to_filename(local_filename)

        print(f"File downloaded successfully into {local_filename}")
    except Exception as e:
        print(f"Error downloading file: {e}")

def upload_file(source_file_name, destination_blob_name):
    # Define the bucket and object
    bucket_name = "mxfaas-storage"

    # Uploads a file to the bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

