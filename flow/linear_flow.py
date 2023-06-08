from typing import List, Union

from chisel.ops.base_chisel import BaseChisel


class LinearFlow(object):
    def __init__(self, ops: List[BaseChisel] = []):
        self.ops: List[BaseChisel] = ops

    def add(self, ops: Union[BaseChisel, List[BaseChisel]]) -> None:
        if isinstance(ops, BaseChisel):
            self.ops.append(ops)
        elif isinstance(ops, List[BaseChisel]):
            self.ops += ops
        else:
            raise ValueError("ops should be either a BaseChisel or a Union " +
                             "of BaseChisels.")

    def __call__(self, *args):
        x = args
        for op in self.ops:
            x = op(x)
            # TODO: add logging/storage of results at each step.
        return x
