import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import openai

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

        results = response.get('data', None)
        if results is None:
            return api_result

        for i, result in enumerate(results):
            url = result.get("url", None)
            if url is None:
                continue
            local_filename = self._download_img_from_url(url, ext=".png")
            api_result.add(local_filename, remote_url=url)
        return api_result


class HFInference(HF):
    def __init__(self) -> None:
        super().__init__()
        self.params: Dict[str, Any] = {
            "width": 512,
            "samples": 1,
        }

    def _query(self, payload):
        data = json.dumps(payload)
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    def run(self, inp: Any, params: Dict[str, str] = None) -> Any:
        if not isinstance(inp, str):
            raise ValueError("Expected inp to be a str (prompt).")

        model_id = params.get("model_id", None)
        API_URL = f"https://api-inference.huggingface.co/models/{model_id}"

        print(self.api_key)

        headers = {"Authorization": f"Bearer {self.api_key}"}

        data = query("Can you please let us know more details about your ")
        return self._process_results(response)
