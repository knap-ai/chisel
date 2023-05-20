from chisel.api import OpenAITxtToImg, StabilityAITxtToImg, StableDiffusionAPITxtToImg
from chisel.api.base_api_provider import BaseAPIProvider
from chisel.data_types import Text, Image
from chisel.ops.base_chisel import BaseChisel
from chisel.ops.provider import Provider


class TxtToImg(BaseChisel):
    def __init__(self, provider: Provider) -> None:
        super().__init__(provider)

    def _get_api(self, provider: Provider) -> BaseAPIProvider:
        if provider == Provider.OPENAI:
            return OpenAITxtToImg()
        elif provider == Provider.STABILITY_AI:
            return StabilityAITxtToImg()
        elif provider == Provider.STABLE_DIFFUSION_API:
            return StableDiffusionAPITxtToImg()
        else:
            raise ValueError(f"Invalid provider: {provider}")

    def __call__(self, txt: Text) -> Image:
        return self.api.run(txt)
