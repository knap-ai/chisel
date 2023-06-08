import re
from azure.storage.blob import BlobServiceClient

from chisel.data_sources.base_data_source import BaseDataSource


class AzureBlob(BaseDataSource):
    def __init__(self, connection_string, container_name):
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    def upload_file(self, local_file_path, blob_name):
        """
        Uploads a file to Azure Blob Storage.

        Args:
            local_file_path (str): Path to the local file.
            blob_name (str): Name of the blob in Azure Blob Storage.

        Returns:
            bool: True if the upload was successful, False otherwise.
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
            )
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data)
            return True
        except Exception as e:
            print(f"Upload failed: {str(e)}")
            return False

    def download_file(self, blob_name, local_file_path):
        """
        Downloads a file from Azure Blob Storage.

        Args:
            blob_name (str): Name of the blob in Azure Blob Storage.
            local_file_path (str): Path to save the downloaded file locally.

        Returns:
            bool: True if the download was successful, False otherwise.
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, blob=blob_name
            )
            with open(local_file_path, "wb") as data:
                data.write(blob_client.download_blob().readall())
            return True
        except Exception as e:
            print(f"Download failed: {str(e)}")
            return False

    def list_files_matching_regex(self, directory, regex_pattern):
        """
        Lists filenames in a directory in Azure Blob Storage container
        that match the provided regex pattern.

        Args:
            directory (str): Directory path in Azure Blob Storage.
            regex_pattern (str): Regular expression pattern to match filenames.

        Returns:
            list: List of filenames that match the regex pattern.
        """
        try:
            blob_client = self.blob_service_client.get_container_client(self.container_name)
            blob_list = blob_client.list_blobs(name_starts_with=directory)
            matching_files = [blob.name for blob in blob_list if re.match(regex_pattern, blob.name)]
            return matching_files
        except Exception as e:
            print(f"Failed to list files: {str(e)}")
            return []
