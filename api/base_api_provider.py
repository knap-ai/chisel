from abc import abstractmethod, ABCMeta
from pathlib import Path
from typing import Any, Dict, Optional

import PIL
import requests
from requests.adapters import HTTPAdapter, Retry

from chisel.storage.local_fs import LocalFS
from chisel.util.files import get_ext


class BaseAPIProvider(metaclass=ABCMeta):
    def __init__(self, storage_dir: str = "~/.chisel"):
        self.storage = LocalFS(storage_dir)
        s = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.1,
            status_forcelist=[404, 500, 502, 503, 504]
        )
        s.mount('http://', HTTPAdapter(max_retries=retries))
        s.mount('https://', HTTPAdapter(max_retries=retries))
        self.requests_session = s

    @abstractmethod
    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        return None

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def set_params(self, params: Dict[str, Any]) -> Any:
        # TODO: would be nice to be able to validate params as they come in.
        if params is not None and isinstance(params, dict):
            for k, v in params.items():
                if k in self.params.keys():
                    self.params[k] = v

    def _write_img(self, img: PIL.Image, filename: str) -> Path:
        return self.storage.write_img_to_tmp(img, filename)

    def _download_img_from_url(
        self,
        url: str,
        filename: str = None,
        ext: str = None
    ) -> Path:
        r = self.requests_session.get(url, stream=True)

        full_path = None
        if r.status_code == 200:
            if filename is None and ext is None:
                ext = get_ext(url)
            elif filename is not None and ext is None:
                ext = get_ext(filename)
            full_path = self.storage.stream_to_tmp(r, filename, ext)
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
