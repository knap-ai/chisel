import json
import os
from typing import Any, Dict, List

from PIL import Image

from chisel.api.base_api_provider import (
    APIResult, BaseAPIProvider
)


class HF(BaseAPIProvider):
    api_key_name: str = "HF_API_KEY"

    def __init__(self) -> None:
        super().__init__()
        self.api_key = os.environ.get(self.api_key_name)

    def _process_results(self, response) -> List[Any]:
        api_result = APIResult()

        print("HEADER: ", response.headers['Content-Type'])
        if response.headers['Content-Type'] == 'image/jpeg':
            data = response.content
            full_path = self.storage.write_to_tmp(data, ext=".png")
            return Image.open(full_path)
        else:
            data = json.loads(response.content.decode("utf-8"))
            self.storage.write_to_tmp(data, ext=".txt")
            return data


class HFInference(HF):
    def __init__(self) -> None:
        super().__init__()

    def run(self, inp: Any, params: Dict[str, str] = None) -> Any:
        model_id = params.get("model_id", None)
        txt_to_img = params.get("txt_to_img", False)

        api_url = f"https://api-inference.huggingface.co/models/{model_id}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            'Content-Type': 'application/json',
        }

        response = self._post_with_retry(api_url, headers, inp)
        return self._process_results(response)
