from minizinc import Instance, Model, Solver

def solve_jindosh_riddle():
    model = Model('./jindosh_riddle.mzn')
    gecode = Solver.lookup("gecode")
    instance = Instance(gecode, model)
    result = instance.solve()
    
    output = str(result)
    return output

result = solve_jindosh_riddle()
print(result)