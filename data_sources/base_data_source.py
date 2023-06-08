from abc import abstractmethod, ABCMeta

from chisel.storage.local_fs import LocalFS


class BaseDataSource(metaclass=ABCMeta):
    def __init__(self):
        self.storage = LocalFS("~/chisel")
