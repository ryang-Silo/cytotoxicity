import os
from azure.storage.blob import BlobServiceClient

storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=cytotoxstorageaccount;AccountKey=o1S2QHEh/t24vV3lMUoL9Q1wbldlWIY4LIn/KrGTCN21xK30D2bCVVucMC1FsaybfYc1P9BycXZJ+AStf0AK8g==;EndpointSuffix=core.windows.net' 
blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)

# Create a container
container_name = 'imagefolder'
container_client = blob_service_client.create_container(container_name)



# upload to Contatiner 
files = '/home/azureuser/cloudfiles/code/workspace/cytotoxicity/sample_files'
for file in os.listdir(files):
    blob_obj = blob_service_client.get_blob_client(container=container_name, blob=file)
    print(f"Uploading file: {file}...")
    
    with open(os.path.join(files, file), mode='rb') as file_data:
        blob_obj.upload_blob(file_data)