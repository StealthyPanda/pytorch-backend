from nodes import *


def is_instance_of(obj : object, cls):
    return (
        type(obj).__name__ == cls.__name__ and 
        all(hasattr(obj, attr) for attr in dir(cls) if not attr.startswith('__'))
    )


tab = '    '
class Renderable:
    def __init__(self, *children, info : str = None, indentlvl : int = 0) -> None:
        self.children = children
        self.lvl = indentlvl
        self.info = info if info is not None else ''
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return (
            '\n'.join(list(map(lambda x: (tab * (self.lvl)) + str(x) , self.children)))
        )

class Info(Renderable):
    def __init__(self, *children, indentlvl : int = 0) -> None:
        super().__init__(*children, info=None, indentlvl=indentlvl)
    
    def __str__(self) -> str:
        self.children = list(map(lambda x: f'# {str(x)}' , self.children))
        return super().__str__()


class PTExpr(Renderable):
    def __init__(self, left : Renderable, op : str, right : Renderable) -> None:
        super().__init__(left, op, right, indentlvl=0)

    def __str__(self) -> str:
        left, op, right = self.children
        if op == '^': op = '**'
        return f'({str(left)} {(op)} {str(right)})'



class PTAssignment(Renderable):
    def __init__(self, left : Renderable, right : Renderable, indentlvl: int = 0) -> None:
        super().__init__(left, right, indentlvl=indentlvl)
    
    def __str__(self) -> str:
        left, right = self.children
        return (tab * self.lvl) + f'{str(left)} = {str(right)}'


def dmarc(n : int = 100) -> str:
    return '#'+('-'*n)

class PTModule(Renderable):
    def __init__(self, *children, name : str = None, info: str = None, indentlvl: int = 0) -> None:
        super().__init__(*children, info=info, indentlvl=indentlvl)
        self.name = name
        self.params = []
    
    def __str__(self) -> str:
        self.children = list(self.children)
        if not self.children: self.children = ['pass']
        
        for i, each in enumerate(self.children):
            if isinstance(each, Renderable):
                each.lvl = self.lvl + 1
            else:
                self.children[i] = (tab * (self.lvl + 1)) + str(each)
        self.children.insert(0, f'class {self.name}(torch.nn.Module):')
        self.children.insert(0, self.info)
        self.children.insert(0, '\n')
        self.children.insert(0, dmarc())
        self.children.append('\n')
        self.children.append(dmarc())
        
        return super().__str__()
        

class PTParam(Renderable):
    def __init__(self, shape : Shape) -> None:
        super().__init__(info=None, indentlvl=0)
        self.shape = shape.dims
    
    def __str__(self) -> str:
        # return f'torch.nn.Parameter({", ".join(list(map(lambda x: str(x), self.shape)))}, bias = False)'
        return f'torch.nn.Parameter(torch.randn({self.shape}))'

class PTVar(Renderable):
    def __init__(self, variable : Var | str, module : PTModule) -> None:
        super().__init__(info=None, indentlvl=0)
        self.name = variable
        self.module = module
    
    def __str__(self) -> str:
        if is_instance_of(self.name, Arg) : return f'self.{self.name.name}'
        if self.name in self.module.params: return f'self.{self.name.name}'
        if is_instance_of(self.name, Var) or is_instance_of(self.name, Symbol): return self.name.name
        return self.name


class PTInit(Renderable):
    def __init__(self, *children, info: str = None, indentlvl : int = 0) -> None:
        super().__init__(*children, info=info, indentlvl=indentlvl)
        self.children = ['super().__init__()', *self.children]
        
    
    def __str__(self) -> str:        
        for i, each in enumerate(self.children):
            if isinstance(each, Renderable): each.lvl = self.lvl
            else: self.children[i] = ('    ' * (self.lvl)) + each
        self.children.insert(0, f'def __init__(self):')
        
        if self.info:
            self.children.insert(
                0, self.info
            )
        
        self.children.insert(0, '\n')
        return super().__str__()


class PTForward(Renderable):
    def __init__(self, *children, inputs : list[Var | str], info: str = None, indentlvl : int = 0) -> None:
        super().__init__(*children, info=info, indentlvl=indentlvl)
        self.inputs = inputs
        
    
    def __str__(self) -> str:
        self.children = list(self.children)   
        if not self.children: self.children = ['pass']
        for i, each in enumerate(self.children):
            if isinstance(each, Renderable): each.lvl = self.lvl
            else: self.children[i] = ('    ' * (self.lvl)) + each
        self.children.insert(
            0, 
            f'def forward(self, {", ".join(list(map(lambda x: str(x) , self.inputs)))}):'
        )
        if self.info:
            self.children.insert(
                0, self.info
            )
        
        self.children.insert(0, '\n')
        return super().__str__()


class PTCall(Renderable):
    def __init__(self, name : str, inputs : list[PTExpr], module : PTModule) -> None:
        super().__init__(info=None, indentlvl=0)
        self.name = PTVar(name, module)
        self.inputs = inputs
    
    def __str__(self) -> str:
        return f'{self.name}({", ".join(list(map(lambda x: str(x) , self.inputs)))})'


class PTReturn(Renderable):
    def __init__(self, value : PTExpr, info: str = None, indentlvl: int = 0) -> None:
        super().__init__(info=info, indentlvl=indentlvl)
        self.value = value
    
    def __str__(self) -> str:
        self.children = [f'return ({str(self.value)})']
        return super().__str__()






preamble = (
'''
# File autogenerated by ptbackend plugin.
# Simply import models from this module to integrate in your pyTorch workflows.

import torch
''' + '\n\n\n'
)



attrs = {
    'max' : lambda x, module, *args: PTCall('torch.max', [x], module),
    'min' : lambda x, module, *args: PTCall('torch.min', [x], module),
    'T' : lambda x, module, *args: PTCall('torch.transpose', [x, *args], module),
    # 'min' : lambda x: PTCall('torch.min', [x]),
}



def make_expression(expr : Expr, module : PTModule) -> PTExpr:
    if is_instance_of(expr, Op):
        if expr.value in ['+', '-', '/', '*', '^', '@']:
            return PTExpr(make_expression(expr.left, module), expr.value, make_expression(expr.right, module))
        elif expr.value == '.':
            name = expr.right
            if is_instance_of(expr.right, Call):
                name = expr.right.name
                return attrs[name](expr.left, module, *expr.right.args)
            return attrs[name](expr.left, module)
            
    elif is_instance_of(expr, Call):
        return PTCall(expr.name, expr.args, module)
    elif type(expr) in [str, int, float]:
        return expr
    else:
        return PTVar(expr, module)


def make_module(
        name : str, flow : FlowDef, 
        context : dict[str | Var, Shape], allmodules : list[PTModule],
        program : Program
    ) -> PTModule:
    mod = PTModule(name=name, info=Info(f'Module corresponding to flow / sub-flow `{flow.name}`'))
    
    # print(f'Building {name} ({flow.name})')
    
    symbols = flow.proto.symbols
    params = flow.proto.args
    
    init = [PTAssignment(PTVar(x, mod), PTParam(context[x])) for x in params] if params else []
    forward = []
    
    
    for stmt in flow.body.statements:
        # print(type(stmt), is_instance_of(stmt, Assignment), stmt)
        if is_instance_of(stmt, Let):
            mod.params.append(stmt.idts[0])
            init.append(PTAssignment(PTVar(stmt.idts[0], mod), f'{stmt.flow.name}()'))
            # print('here', type(program.getflow(stmt.flow)))
            
            already_made = False
            for each in allmodules:
                if each.name == stmt.flow: already_made = True
            
            if not already_made:
                allmodules.append(
                    make_module(
                        stmt.flow, program.getflow(stmt.flow), 
                        context[stmt.idts[0]], allmodules, program
                    )
                )
            
        elif is_instance_of(stmt, Assignment):
            forward.append(PTAssignment(PTVar(stmt.left, mod), make_expression(stmt.right, mod)))
        elif is_instance_of(stmt, Return):
            forward.append(PTReturn(make_expression(stmt.value, mod)))
    
    init = PTInit(*init, info=Info('Initializing parameters'))
    forward = PTForward(*forward, inputs = symbols, info=Info('Main logic of the flow'))
    mod.children = [init, forward]
    return mod















if __name__ == '__main__':
    a = PTModule('linear', Info('Simple linear class'))
    a.children = [
        PTInit(
            PTAssignment(PTVar(Symbol(name='x')), PTParam(Shape(dims=[10, 784]))),
            PTAssignment(PTVar(Arg(name='x2')), PTParam(Shape(dims=[10, 784]))),
        )
    ]
    
    with open('./buffer.py', 'w') as file:
        file.write(preamble + str(a))
    
