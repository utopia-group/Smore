from typing import List


class Symbol:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name


class TerminalSymbol(Symbol):
    def __init__(self, name: str, values: List[str | int | float]):
        super().__init__(name)
        self.values = values


class NonterminalSymbol(Symbol):
    def __init__(self, name: str, is_recursive=False):
        super().__init__(name)
        self.is_recursive: bool = is_recursive
