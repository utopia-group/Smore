from typing import Optional, List, Tuple

from lib.lang.production import Production, IdProduction
from lib.lang.symbol import Symbol

"""
class for node, which is the element in program
"""


class Node:
    def __init__(self, _id: int, name: str, sym: Optional[Symbol]):
        self.id: int = _id
        self.name: str | int = name
        self.sym: Optional[Symbol] = sym

    def duplicate(self, _id: int) -> 'Node':
        node = Node(_id, self.name, self.sym)
        return node

    def __repr__(self):
        return str(self.name)


class NonterminalNode(Node):
    def __init__(self, _id: int, name: str, sym: Symbol, prod: Production):
        super().__init__(_id, name, sym)
        self.prod: Production = prod

    def duplicate(self, _id: int) -> 'NonterminalNode':
        node = NonterminalNode(_id, self.name, self.sym, self.prod)
        return node


class TerminalNode(Node):
    def __init__(self, _id: int, name: str | int | float, sym: Optional[Symbol], prod: Optional[IdProduction] = None):
        super().__init__(_id, name, sym)
        self.id_prod: IdProduction = prod

    def duplicate(self, _id: int) -> 'TerminalNode':
        node = TerminalNode(_id, self.name, self.sym, self.id_prod)
        return node


class NoneNode(TerminalNode):
    def __init__(self, _id: int):
        super().__init__(_id, '', None)

    def duplicate(self, _id: int) -> 'NoneNode':
        node = NoneNode(_id)
        return node


class VariableNode(Node):
    def __init__(self, _id: int, name: str, sym: Symbol):
        super().__init__(_id, name, sym)

    def duplicate(self, _id: int) -> 'VariableNode':
        node = VariableNode(_id, self.name, self.sym)
        return node


class HoleNode(Node):
    def __init__(self, _id: int, _type: str, hole_id: int, name: str = 'hole', sym: Optional[Symbol] = None):
        super().__init__(_id, name, sym)
        self.type: str = _type
        self.hole_id: int = hole_id

    def duplicate(self, _id: int) -> 'HoleNode':
        node = HoleNode(_id, self.type, self.hole_id, self.name, self.sym)
        return node

    def __repr__(self):
        return '{{??{}: {}}}'.format(self.id, self.type)


class ContextNode(TerminalNode):
    def __init__(self, _id: int, name: str | int | float, value: Tuple[List[str], List[str]], sym: Optional[Symbol] = None):
        super().__init__(_id, name, sym)
        self.value: Tuple[List[str], List[str]] = value

    def duplicate(self, _id: int) -> 'ContextNode':
        new_value = (self.value[0].copy(), self.value[1].copy())
        node = ContextNode(_id, self.name, new_value, self.sym)

        return node

    def __repr__(self):
        return repr(self.value)