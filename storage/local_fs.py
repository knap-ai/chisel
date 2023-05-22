import random
import string
from os import listdir, unlink
from os.path import expanduser, isfile, join
from pathlib import Path
from typing import Any

import PIL


class LocalFS(object):
    def __init__(self, storage_dir: str) -> None:
        self.storage_dir = expanduser(storage_dir)
        self.tmp_storage = Path(join(self.storage_dir, "tmp"))
        self.tmp_storage.mkdir(parents=True, exist_ok=True)

    def write_to_tmp(
        self,
        obj: Any,
        filename: Any = None,
        ext: str = None,
    ) -> Path:
        if filename is None:
            filename = self._get_random_tmp_filename(ext)

        full_path = self.tmp_storage / filename
        with open(str(full_path), "wb") as f:
            f.write(obj)
        return full_path

    def write_img_to_tmp(self, img: PIL.Image, filename: str) -> Path:
        full_path = self.tmp_storage / Path(filename)
        img.save(str(full_path))
        return full_path

    def stream_to_tmp(
        self,
        requests_result=None,
        filename: Any = None,
        ext: str = None,
    ) -> Path:
        if filename is None:
            filename = self._get_random_tmp_filename(ext)

        full_path = self.tmp_storage / filename
        with open(str(full_path), "wb") as f:
            for chunk in requests_result.iter_content(4096):
                f.write(chunk)
        return full_path

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

    def cleanup_tmp_storage(self):
        for file_name in listdir(self.tmp_storage):
            file_path = join(self.tmp_storage, file_name)
            try:
                if isfile(file_path):
                    unlink(file_path)
                    print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
