from nodes import *
from ptbackend import *



def main(program : Program, context : dict[Var | str, dict | Shape]) -> str:
    allmods = []
    
    build = program.builds[-1]
    
    mod = make_module(build.name, build.flow, context, allmods, program)
    
    allmods.append(mod)
    
    allmods = [str(x) for x in allmods]
    
    return (preamble + '\n'.join(allmods)), program, context




