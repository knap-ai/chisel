import chisel.api
from chisel.api.base_api_provider import BaseAPIProvider
from chisel.data_types import Image
from chisel.ops.base_chisel import BaseChisel
from chisel.ops.provider import Provider


class ImgToImg(BaseChisel):
    def __init__(self, provider: Provider) -> None:
        super().__init__(provider)

    def _get_api(self, provider: Provider) -> BaseAPIProvider:
        if provider == Provider.OPENAI:
            return OpenAIImgToImg()
        if provider == Provider.DREAMBOOTH:
            return StabilityAIImgToImg()
        if provider == Provider.STABLEDIFFUSIONAPI:
            return StableDiffusionAPIImgToImg()

    def __call__(self, img: Image) -> Image:
        return self.api(img)
