from abc import abstractmethod, ABCMeta
from pathlib import Path
from typing import Any, Dict, Optional

import PIL
import requests

from chisel.storage.local_fs import LocalFS


class BaseAPIProvider(metaclass=ABCMeta):
    def __init__(self, storage_dir: str = "~/.chisel"):
        self.storage = LocalFS(storage_dir)

    @abstractmethod
    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        return None

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def set_params(self, params: Dict[str, Any]) -> Any:
        # TODO: would be nice to be able to validate params as they come in.
        for k, v in params.items():
            if k in self.params.keys():
                self.params[k] = v

    def _download_img_from_url(
        self,
        url: str,
        filename: str = None,
        ext: str = None
    ) -> Path:
        r = requests.get(url, stream=True)
        full_path = None
        if r.status_code == 200:
            full_path = self.storage.stream_to_tmp(filename, r)
        return full_path


class APIResult(object):
    def __init__(self) -> None:
        self.results = []

    def add(
        self,
        local_filename: str = None,
        remote_url: str = None,
    ) -> None:
        self.results.append({
            "img": PIL.Image.open(str(local_filename)),
            "local_filename": local_filename,
            "remote_url": remote_url,
        })

    def get_image(self, idx: int) -> PIL.Image:
        return self.results[idx].get("img", None)

    def __len__(self) -> int:
        return len(self.results)
