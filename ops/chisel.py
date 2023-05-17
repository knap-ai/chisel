from abc import abstractmethod, ABCMeta
from typing import Any, Dict

from chisel.api.base_api_provider import BaseAPIProvider
from chisel.ops.provider import Provider


class BaseChisel(metaclass=ABCMeta):
    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        self.api = self._get_api(provider)

    @abstractmethod
    def _get_api(self, provider: Provider) -> BaseAPIProvider:
        return None

    def get_params(self) -> Dict[str, Any]:
        return self.api.get_params()

    def set_params(self) -> Dict[str, Any]:
        return self.api.set_params()

    @abstractmethod
    def __call__(self, *args):
        return None
