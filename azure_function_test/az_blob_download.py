import os
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import ContainerClient

storage_connection_string = ''
# blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)

# data_container_url = 'https://cytotoxstorageaccount.blob.core.windows.net/compounds-images'
# service = BlobServiceClient(account_url=data_container_url, credential=storage_connection_string)


container_name="compounds-images"
blob_name = 'reference_dataset_compounds_v1'
container = ContainerClient.from_connection_string(conn_str=storage_connection_string, container_name="compounds-images", blob=blob_name)

# blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
# container_client = blob_service_client.get_container_client("compounds-images")
# blob_client = container_client.get_blob_client(blob_name)
# properties = blob_client.get_blob_properties()


blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
container_client = blob_service_client.get_container_client(container_name)
for file in container_client.walk_blobs('reference_dataset_compounds_v1/', delimiter='/'):
    print(file.name)