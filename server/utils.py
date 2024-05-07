
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from google.cloud import storage
import os


def download_blobs(bucket_name, source_blob_names, destination_directory):
    """Downloads multiple blobs from a bucket to a local directory."""

    storage_client = storage.Client.from_service_account_json('secrets/mlops-422314-d808df4f88f0.json')
    bucket = storage_client.bucket(bucket_name)

    for source_blob_name in source_blob_names:
        blob = bucket.blob(source_blob_name)
        destination_file_name = os.path.join(destination_directory, source_blob_name.split("/")[-1]) 
        blob.download_to_filename(destination_file_name)

        print(
            f"Blob {source_blob_name} downloaded to {destination_file_name}."
        )


def load_model():
    labels = read_label_file('artifcats/labels.txt')
    interpreter = make_interpreter('artifacts/model.tflite')
    interpreter.allocate_tensors()
    return interpreter, labels