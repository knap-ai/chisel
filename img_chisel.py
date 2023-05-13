import os
import random
import requests
import string
from os.path import expanduser, join
from pathlib import Path
from typing import Any


class ImgChisel(object):
    def __init__(self, storage_dir: str = "~/.chisel"):
        self.storage_dir = expanduser(storage_dir)
        self.tmp_storage = Path(join(self.storage_dir, "tmp"))
        self.tmp_storage.mkdir(parents=True, exist_ok=True)

    def _download_img_from_url(
        self,
        url: str,
        filename: str = None,
        ext: str = None
    ) -> Path:
        if filename is None:
            # TODO: would be nice to abstract this away from only tmp storage.
            filename = self._get_random_tmp_filename(ext)
        full_path = self.tmp_storage / filename
        # urllib.request.urlretrieve(url, full_path)
        # r = requests.get(
        #     settings.STATICMAP_URL.format(**data),
        #     stream=True
        # )
        print("FULL PATH: ", full_path)
        r = requests.get(url, stream=True)
        print("REQUESTS RESPONSE: ", r)
        if r.status_code == 200:
            with open(full_path, 'wb') as f:
                for chunk in r.iter_content(4096):
                    f.write(chunk)
        return full_path

    def _write_to_tmp(self, obj: Any, filename: Any, ext: str = None):
        if filename is None:
            filename = self._get_random_tmp_filename(ext)

        full_path = self.tmp_storage / filename
        with open(str(full_path), "wb") as f:
            f.write(obj)

    def _get_random_tmp_filename(self, ext: str) -> Path:
        if ext is None:
            raise ValueError("To create a random tmp filename, an extension must be given.")
        N = 20
        # using random.choices()
        # generating random strings
        tmp_filename = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=N)
        )
        return Path(tmp_filename + ext)

    def _cleanup_tmp_storage(self):
        for file_name in os.listdir(self.tmp_storage):
            file_path = os.path.join(self.tmp_storage, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
