from __future__ import annotations
import inspect
import itertools
import typing
from typing import Callable, Iterable, get_args, get_origin

from dataclasses import fields

from fpy.syntax import Ast
from fpy.state import CompileState
from fpy.types import UNION_TYPES

from fpy.bytecode.directives import Directive
from fpy.ir import Ir


# Cache for visitor method mappings, keyed by visitor class
_visitor_cache: dict[type, dict[type, str]] = {}


class _StopDescent:
    """Sentinel returned by a visit_* method to prevent the framework from
    descending into the visited node's children."""
    __slots__ = ()
    def __repr__(self):
        return "STOP_DESCENT"

STOP_DESCENT = _StopDescent()


class Visitor:
    """visits each class, calling a custom visit function, if one is defined, for each
    node type"""

    def __init__(self):
        self.visitors: dict[type[Ast], Callable] = {}
        """dict of node type to handler function"""
        self.build_visitor_dict()

    def build_visitor_dict(self):
        cls = type(self)
        # Check if this class's visitor mapping is already cached
        if cls in _visitor_cache:
            # Use cached mapping (maps node type -> method name)
            for node_type, method_name in _visitor_cache[cls].items():
                self.visitors[node_type] = getattr(self, method_name)
            return

        # Build the mapping and cache it
        class_cache: dict[type, str] = {}
        for name, func in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("visit") or name == "visit_default":
                # not a visitor, or the default visit func
                continue
            signature = inspect.signature(func)
            params = list(signature.parameters.values())
            assert len(params) == 3
            assert params[1].annotation is not None
            annotations = typing.get_type_hints(func)
            param_type = annotations[params[1].name]

            origin = get_origin(param_type)
            if origin in UNION_TYPES:
                # It's a Union type, so get its arguments.
                for t in get_args(param_type):
                    class_cache[t] = name
                    self.visitors[t] = getattr(self, name)
            else:
                # It's not a Union, so it's a regular type
                class_cache[param_type] = name
                self.visitors[param_type] = getattr(self, name)

        _visitor_cache[cls] = class_cache

    def _visit(self, node: Ast, state: CompileState):
        visit_func = self.visitors.get(type(node), self.visit_default)
        return visit_func(node, state)

    def visit_default(self, node: Ast, state: CompileState):
        pass

    def run(self, start: Ast, state: CompileState):
        """runs the visitor, starting at the given node, descending depth-first"""

        def _descend(node: Ast):
            if not isinstance(node, Ast):
                return
            children = []
            for field in fields(node):
                field_val = getattr(node, field.name)
                if isinstance(field_val, list):
                    # also handle the one case where we have a list of tuples
                    if len(field_val) > 0 and isinstance(field_val[0], tuple):
                        field_val = itertools.chain.from_iterable(field_val)
                    children.extend(field_val)
                else:
                    children.append(field_val)

            for child in children:
                if not isinstance(child, Ast):
                    continue
                _descend(child)
                if len(state.errors) != 0:
                    break
                self._visit(child, state)
                if len(state.errors) != 0:
                    break

        _descend(start)
        self._visit(start, state)


class TopDownVisitor(Visitor):
    """Like Visitor, but visits parent before children (top-down / pre-order)."""

    def run(self, start: Ast, state: CompileState):
        """runs the visitor, starting at the given node, descending breadth-first"""

        def _descend(node: Ast):
            if not isinstance(node, Ast):
                return
            children = []
            for field in fields(node):
                field_val = getattr(node, field.name)
                if isinstance(field_val, list):
                    # also handle the one case where we have a list of tuples
                    if len(field_val) > 0 and isinstance(field_val[0], tuple):
                        field_val = itertools.chain.from_iterable(field_val)
                    children.extend(field_val)
                else:
                    children.append(field_val)

            for child in children:
                if not isinstance(child, Ast):
                    continue
                result = self._visit(child, state)
                if len(state.errors) != 0:
                    break
                if result is not STOP_DESCENT:
                    _descend(child)
                if len(state.errors) != 0:
                    break

        result = self._visit(start, state)
        if result is not STOP_DESCENT:
            _descend(start)


class Transformer(Visitor):

    class Delete:
        pass

    def run(self, start: Ast, state: CompileState):

        def _descend(node):
            if not isinstance(node, Ast):
                return
            for field in fields(node):
                field_val = getattr(node, field.name)
                if isinstance(field_val, list):
                    # child is a list, iterate over each member of the list
                    # use a copy so we can remove as we traverse, also so
                    # we don't visit things that we added

                    #
                    idx = -1
                    for child in field_val[:]:
                        idx += 1
                        if not isinstance(child, Ast):
                            continue
                        _descend(child)
                        if len(state.errors) != 0:
                            break
                        transformed = self._visit(child, state)
                        if len(state.errors) != 0:
                            break
                        if isinstance(transformed, Iterable):
                            assert all(
                                isinstance(n, Ast) for n in transformed
                            ), transformed
                            # func split one node into many
                            # remove the original child and add the new ones
                            # insert them in the place where the child used to be, in the right order
                            field_val.remove(child)
                            for new_child_idx, new_child in enumerate(transformed):
                                field_val.insert(idx + new_child_idx, new_child)
                            # make sure that we maintain insertion order by updating the idx
                            # accounting for our removal of an original node
                            # if we don't do this, then if we were to insert into list after this based on idx,
                            # the positions could be swapped around
                            idx += len(transformed) - 1
                        elif isinstance(transformed, Ast):
                            field_val.remove(child)
                            field_val.insert(idx, transformed)
                        elif transformed is Transformer.Delete:
                            # just delete it
                            field_val.remove(child)
                        else:
                            assert transformed is None, transformed
                            # don't do anything, didn't return anything
                    if len(state.errors) != 0:
                        # need a second check here to get out of the enclosing loop
                        break
                    # don't need to update the field, it was a ptr to a list so should
                    # already be updated
                else:
                    _descend(field_val)
                    if len(state.errors) != 0:
                        break
                    transformed = self._visit(field_val, state)
                    if len(state.errors) != 0:
                        break
                    if isinstance(transformed, Ast):
                        setattr(node, field.name, transformed)
                    elif transformed is Transformer.Delete:
                        # just delete it
                        setattr(node, field.name, None)
                    else:
                        # cannot return a list if the original attr wasn't a list
                        assert transformed is None, transformed
                        # don't do anything, didn't return anything

        _descend(start)
        self._visit(start, state)


# Cache for emitter method mappings, keyed by emitter class
_emitter_cache: dict[type, dict[type, str]] = {}


class Emitter:
    # Default: not in a function (top-level code)
    # Subclasses override this to indicate function body context
    in_function = False

    def __init__(self):
        self.emitters: dict[type[Ast], Callable] = {}
        """dict of node type to handler function"""
        self.build_emitter_dict()

    def build_emitter_dict(self):
        cls = type(self)
        # Check if this class's emitter mapping is already cached
        if cls in _emitter_cache:
            # Use cached mapping (maps node type -> method name)
            for node_type, method_name in _emitter_cache[cls].items():
                self.emitters[node_type] = getattr(self, method_name)
            return

        # Build the mapping and cache it
        class_cache: dict[type, str] = {}
        for name, func in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("emit_"):
                # not an emitter
                continue
            signature = inspect.signature(func)
            params = list(signature.parameters.values())
            assert len(params) == 3
            assert params[1].annotation is not None
            annotations = typing.get_type_hints(func)
            param_type = annotations[params[1].name]

            origin = get_origin(param_type)
            if origin in UNION_TYPES:
                # It's a Union type, so get its arguments.
                for t in get_args(param_type):
                    class_cache[t] = name
                    self.emitters[t] = getattr(self, name)
            else:
                # It's not a Union, so it's a regular type
                class_cache[param_type] = name
                self.emitters[param_type] = getattr(self, name)

        _emitter_cache[cls] = class_cache

    def emit(self, node: Ast, state: CompileState) -> list[Directive | Ir]:
        return self.emitters[type(node)](node, state)
