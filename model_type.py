from typing import Any, Set


class ModelType(object):
    model_types: Set[str] = {
        "txt_to_img", "img_to_img", "inpainting", "super_resolution"
    }
    # model_type_to_input_type: Dict[str, type] = {
    #     "txt_to_img": set(str),
    #     "img_to_img": set(NDArray, List[str]),
    #     "super_res": set(NDArray, List[str])
    # }

    def __init__(self, model_type: Any):
        ModelType.validate(model_type)
        self.model_type = model_type

    @classmethod
    def validate(cls, model_type_str: str) -> None:
        if model_type_str not in cls.model_types:
            raise ValueError(f"{model_type_str} not a valid model type. Must be " +
                             f"one of {cls.model_types}")

    # def validate_model_input(self, inp: Any) -> None:
    #     expected_input_types = self.model_type_to_input_type[self.model_type]
    #     if inp not in :
    #         raise ValueError("input doesn't match expected input type. Should be" +
    #                          f"one of {expected_input_types}")

    def __eq__(self, other: str) -> bool:
        if isinstance(other, str):
            return other == self.model_type
        elif isinstance(other, object) and hasattr(other, self.model_type):
            if other.model_type == self.model_type:
                return True
            return False
        else:
            raise ValueError(f"ModelType can't be compared to {other}")
