from __future__ import annotations
from dataclasses import dataclass
from typing import Union
from fpy.syntax import Ast


class Ir:
    pass


class IrLabel(Ir):
    def __init__(self, node: Ast, label: str):
        super().__init__()
        self.name = f"{node.id}.{label}"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, value):
        return isinstance(value, IrLabel) and value.name == self.name

    def __repr__(self):
        return f"IrLabel({self.name})"


@dataclass(frozen=True, unsafe_hash=True)
class IrGoto(Ir):
    label: IrLabel


@dataclass(frozen=True, unsafe_hash=True)
class IrIf(Ir):
    goto_if_false_label: IrLabel


@dataclass(frozen=True, unsafe_hash=True)
class IrPushLabelOffset(Ir):
    """pushes the label's offset from the start of the file to the stack, as a StackSizeType"""
    label: IrLabel