import boto3
import re

from chisel.data_sources.base_data_source import BaseDataSource


class S3(BaseDataSource):
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.s3 = boto3.client('s3')

    def upload_file(self, local_file_path, s3_key):
        """
        Uploads a file to S3.

        Args:
            local_file_path (str): Path to the local file.
            s3_key (str): Key of the file in S3.

        Returns:
            bool: True if the upload was successful, False otherwise.
        """
        try:
            self.s3.upload_file(local_file_path, self.bucket_name, s3_key)
            return True
        except Exception as e:
            print(f"Upload failed: {str(e)}")
            return False

    def download_file(self, s3_key, local_file_path):
        """
        Downloads a file from S3.

        Args:
            s3_key (str): Key of the file in S3.
            local_file_path (str): Path to save the downloaded file locally.

        Returns:
            bool: True if the download was successful, False otherwise.
        """
        try:
            self.s3.download_file(self.bucket_name, s3_key, local_file_path)
            return True
        except Exception as e:
            print(f"Download failed: {str(e)}")
            return False

    def list_files_matching_regex(self, directory, regex_pattern):
        """
        Lists filenames in a directory in S3 that match the provided regex pattern.

        Args:
            directory (str): Directory path in S3.
            regex_pattern (str): Regular expression pattern to match filenames.

        Returns:
            list: List of filenames that match the regex pattern.
        """
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=directory)
            filenames = [obj['Key'] for obj in response.get('Contents', [])]
            matching_files = [filename for filename in filenames
                              if re.match(regex_pattern, filename)]
            return matching_files
        except Exception as e:
            print(f"Failed to list files: {str(e)}")
            return []
