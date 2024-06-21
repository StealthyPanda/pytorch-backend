from dataclasses import dataclass
from typing import Any, Literal



@dataclass
class Node:
    line : int = None
    charpos : int = None
    def __repr__(self) -> str:
        # return f'{self.line}:' + self.__str__()
        return self.__str__()
    def __str__(self) -> str:
        return ''
    def __hash__(self) -> int:
        return self.__repr__().__hash__()
    def __eq__(self, value: object) -> bool:
        if type(self) != type(value): return False
        return self.__hash__() == value.__hash__()
    def __ne__(self, value: object) -> bool:
        if type(self) != type(value): return True
        return self.__hash__() != value.__hash__()


@dataclass(repr = False)
class Term(Node):
    value : str | float | int = None
    def __str__(self) -> str:
        return f'T({self.value})'


class Dimension(Node):
    num : int = 0
    
    def __init__(self) -> None:
        self.id = Dimension.num
        Dimension.num += 1
        
    def __str__(self) -> str:
        return f'_{self.id}'
    
    def __eq__(self, value: object) -> bool:
        if type(value) != Dimension: return False
        return self.id == value.id
    
    def __ne__(self, value: object) -> bool:
        if type(value) != Dimension: return True
        return self.id != value.id


class ShapeLength(Node):
    num : int = 0
    
    def __init__(self) -> None:
        self.id = ShapeLength.num
        ShapeLength.num += 1
        
    def __str__(self) -> str:
        return f'SL({self.id})'
    
    def __eq__(self, value: object) -> bool:
        if type(value) != ShapeLength: return False
        return self.id == value.id
    
    def __ne__(self, value: object) -> bool:
        if type(value) != ShapeLength: return True
        return self.id != value.id




@dataclass(repr = False)
class Tuple(Node):
    vals : list[Node] = None
    bracks : Literal['round', 'square', 'none'] = None
    
    def __getitem__(self, index : int) -> Node:
        return self.vals[index]
    
    def __str__(self) -> str:
        return f'tup({", ".join(list(map(lambda x: str(x), self.vals)))})'
    


@dataclass(repr = False, eq = False)
class Shape(Node):
    dims : list[int | Node] = None
    length : ShapeLength | int = None
    
    def __len__(self) -> int:
        return len(self.dims)
    
    def __getitem__(self, k : int) -> Node:
        return self.dims[k]
    
    def __setitem__(self, k : int, value : Node) -> None:
        self.dims[k] = value
    
    def __str__(self) -> str:
        return f'Shape({self.length}, {self.dims})'
    
    # def __eq__(self, value: object) -> bool:
    #     if type(self) != type(value): return False
    #     return self.dims == value.dims
    
    # def __ne__(self, value: object) -> bool:
    #     if type(self) != type(value): return True
    #     return self.dims != value.dims
    
    # def __repr__(self) -> str:
    #     if self.dims is not None:
    #         return '{' + ', '.join(list(map(lambda x:str(x), self.dims))) +'}'
    #     else: return '_empty_'


@dataclass(repr = False)
class Expr(Node):
    value : Node = None
    shape : Shape = None


@dataclass(repr = False, eq = False)
class Var(Node):
    name : str = None
    
    def __eq__(self, value : object) -> bool:
        if not (isinstance(value, Var) or (type(value) == str)): return False
        if type(value) == str:
            return self.name == value
        return self.name == value.name
    
    #TODO: reset this if it breaks anything:
    def __str__(self) -> str:
        # return self.name
        return self.name
    
    def __repr__(self) -> str:
        return f'Var({self.name})'
    
    def __hash__(self) -> int:
        return self.name.__hash__()


class Symbol(Var):
    def __str__(self) -> str:
        return self.name
    def __repr__(self) -> str:
        return f'Symbol({self.name})'

class Arg(Var):
    def __str__(self) -> str:
        return self.name
    def __repr__(self) -> str:
        return f'Arg({self.name})'




@dataclass(repr=False, eq=False)
class Number(Node):
    value : float | int = None
    def __eq__(self, value: object) -> bool:
        if not type(value) in [Number, int, float]: return False
        if type(value) == Number:
            return self.value == value.value
        else:
            return self.value == value
    def __str__(self) -> str:
        return f'N({self.value})'
    def __hash__(self) -> int:
        return self.__repr__().__hash__()



@dataclass(repr=False, eq=False)
class Op(Expr):
    left : Expr = None
    right : Expr = None
    def __str__(self) -> str:
        return f'Op({self.value}, left={self.left}, right={self.right})'


@dataclass(repr = False, eq=False)
class Statement:
    line : int = None
    charpos : int = None
    
    def __repr__(self) -> str:
        return (f'{self.line}:' + self.__str__())


@dataclass(repr = False, eq=False)
class Return(Statement):
    value : (
        Expr | Var |
        Number | Tuple
    ) = None
    
    def __str__(self) -> str:
        return f'Return({self.value})'

@dataclass(repr = False, eq = False)
class Let(Statement):
    flow : Node = None
    idts : list[Node] = None
    init : Node = None
    
    def __str__(self) -> str:
        return f'Let({self.flow}, {self.idts})'


@dataclass(repr=False, eq = False)
class Assignment(Statement):
    left : Expr = None
    right : Expr = None
    
    def __str__(self) -> str:
        return f'A({self.left} := {self.right})'







@dataclass
class Body:
    statements : list[Statement] = None
    line : int = None
    charpos : int = None
    def __str__(self) -> str:
        return f'Body({self.statements})'


@dataclass
class Build(Node):
    name : str = None
    flow : Node = None
    body : Body = None


@dataclass
class FlowProto(Node):
    name : str = None
    symbols : list[Symbol] = None
    args : list[Arg] = None


@dataclass(repr = False)
class FlowDef(Node):
    name : str = None
    proto : FlowProto = None
    body : Body = None
    retstmt : Return = None
    subftable : dict = None

    def __str__(self) -> str:
        return f'FlowDef({self.name}, symbols={self.proto.symbols}, args={self.proto.args}, statements={len(self.body.statements)})'

@dataclass(repr=False, eq = False)
class ShapeSpec(Statement):
    var : Node = None
    shape : Shape = None
    flow : FlowDef = None
    
    def __str__(self) -> str:
        return f'SS({self.var}=>{self.shape})'



@dataclass
class Program:
    name : str = None
    flows : list[FlowDef] = None
    builds : list[Build] = None
    file : str = None
    
    def getflow(self, name : str) -> FlowDef:
        for each in self.flows:
            if each.name == name: return each



@dataclass(repr = False)
class Call(Expr):
    name : str = None
    args : Tuple | list[Node] = None
    flow : FlowDef = None
    
    def __str__(self) -> str:
        return f'Call( {self.flow.name if self.flow is not None else ""} {self.name}({self.args}))'



@dataclass
class Slice(Node):
    left : Node = None
    right : Node = None
    step : Node = None



class InferenceChain:
    def __init__(self) -> None:
        self.chain = []
    
    def __getitem__(self, k : int) -> Statement: return self.chain[k]
    def __setitem__(self, k : int, v : Statement): self.chain[k] = v
    
    def __len__(self) -> int: return len(self.chain)
    def __contains__(self, val : Statement) -> bool: return val in self.chain 
    
    def push(self, val : Statement): self.chain.append(val)
    def pop(self, val : Statement) -> Statement: self.chain.append(val)
    


class CodeError(Exception):
    def __init__(
            self, *args: object, line : int = None, charpos : int = None,
            ichain : InferenceChain = None, #context : Context = None
        ) -> None:
        super().__init__(*args)
        self.line = line
        self.charpos = charpos
        self.ichain = ichain
        # self.context = context
    
    def __str__(self) -> str:
        return (
            (f'On line number {self.line}:\n' if self.line is not None else '') +  super().__str__()
        )


class ShapeClash(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)

class DimClash(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)

class MatMulShape(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)

class InferenceError(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)

class UnknownAttribute(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)

class InvalidAttribute(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)

class InvalidTranspose(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)


class InvalidAssignment(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)

class UnknownVar(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)

class UnknownSubFlow(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)

class UnknownFlow(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)


class InvalidSyntax(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)


class InvalidIdentifier(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)

class InvalidStatement(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)

class InvalidShape(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)

class ReturnError(CodeError):
    def __init__(self, *args: object, line: int = None, charpos: int = None, ichain: InferenceChain = None) -> None:
        super().__init__(*args, line=line, charpos=charpos, ichain=ichain)
