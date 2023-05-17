from enum import Enum
from typing import Any, Set

from chisel.data_types import Image, Text


class ModelType(Enum):
    TXT2IMG = ("txt_2_img", Text, Image)
    IMG2IMG = ("img_2_img", Image, Image)
    IMG_EDIT = ("img_edit", Image, Image)
    SUPER_RES = ("super_res", Image, Image)
    VARIATION = ("variation", Image, Image)

    def __init__(self, model_type: str, in_cls: Any, out_cls: Any) -> None:
        self.model_type = model_type
        self.in_cls = in_cls
        self.out_cls = out_cls
