from pathlib import Path
from typing import Any, Dict, List, Optional

import openai

from chisel.api.base_api_provider import APIResult, BaseAPIProvider
from chisel.util.env_handler import EnvHandler


class OpenAI(BaseAPIProvider):
    api_key_name: str = "CHISEL_API_KEY_OPEN_AI"

    def __init__(self) -> None:
        super().__init__()
        if not EnvHandler.contains(self.api_key_name):
            raise Exception(
                f"{self.api_key_name} not set. Please set the env variable "
                + "before using this class"
            )
        openai.api_key = EnvHandler.get(self.api_key_name)

    def _process_results(self, response) -> List[Any]:
        api_result = APIResult()

        results = response.get("data", None)
        if results is None:
            return api_result

        for i, result in enumerate(results):
            url = result.get("url", None)
            if url is None:
                continue
            local_filename = self._download_img_from_url(url, ext=".png")
            api_result.add(local_filename, remote_url=url)
        return api_result


class OpenAITxtToImg(OpenAI):
    def __init__(self) -> None:
        super().__init__()
        self.params: Dict[str, Any] = {
            "width": 512,
            "samples": 1,
        }

    def run(self, inp: str, params: Optional[Dict[str, str]] = None) -> Any:
        if not isinstance(inp, str):
            raise ValueError("Expected inp to be a str (prompt).")

        width = self.params.get("width", 512)
        response = openai.Image.create(
            prompt=inp, n=self.params.get("samples", 1), size=f"{width}x{width}"
        )
        return self._process_results(response)


class OpenAIImgToImg(OpenAI):
    def __init__(self) -> None:
        super().__init__()
        self.params: Dict[str, Any] = {
            "width": 512,
            "samples": 1,
        }

    def run(self, inp: str, params: Optional[Dict[str, str]] = None) -> Any:
        if not isinstance(inp, str) and not isinstance(inp, Path):
            raise ValueError("Expected inp to be a str or Path (path to img).")

        width = self.params.get("width", 512)
        response = openai.Image.create_variation(
            image=open(str(inp), "rb"),
            n=self.params.get("samples", 1),
            size=f"{width}x{width}",
        )
        return self._process_results(response)


class OpenAIImgEdit(OpenAI):
    def __init__(self) -> None:
        super().__init__()
        self.params: Dict[str, Any] = {
            "width": 512,
            "samples": 1,
        }

    def run(self, inp: Any, params: Optional[Dict[str, str]] = None) -> Any:
        if not isinstance(inp, list):
            raise ValueError("Expected inp to be a list of three: [prompt, img, mask]")

        width = self.params.get("width", 512)
        response = openai.Image.create_edit(
            image=open(inp[1], "rb"),
            mask=open(inp[2], "rb"),
            prompt=inp[0],
            n=self.params.get("samples", 1),
            size=f"{width}x{width}",
        )
        return self._process_results(response)
