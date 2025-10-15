# from minizinc import Instance, Model, Solver

# nqueens = Model("./nqueens.mzn")

# gecode = Solver.lookup("gecode")

# instance = Instance(gecode, nqueens)

# instance["n"] = 4
# result = instance.solve()

# print(result["q"])


import minizinc
import asyncio

def solve_cryptarithmetic():
    model_code = """
include "all_different.mzn";

var 0..9: G;
var 0..9: E;
var 0..9: T;
var 0..9: B;
var 0..9: Y;
var 0..9: A;
var 0..9: R;

constraint all_different([G, E, T, B, Y, A, R]);

constraint G != 0;
constraint B != 0;

constraint (100*G + 10*E + T) * (10*B + Y) = (10000*B + 1000*E + 100*A + 10*R + E);

solve satisfy;

output [
  "GET x BY = BEARE:\\n",
  "G = ", show(G), "\\n",
  "E = ", show(E), "\\n", 
  "T = ", show(T), "\\n",
  "B = ", show(B), "\\n",
  "Y = ", show(Y), "\\n",
  "A = ", show(A), "\\n",
  "R = ", show(R), "\\n",
];
"""

    async def solve():
        model = minizinc.Model()
        model.add_string(model_code)
        
        gecode = minizinc.Solver.lookup("gecode")
        instance = minizinc.Instance(gecode, model)
        
        result = await instance.solve_async()
        
        if result.solution is not None:
            print("GET x BY = BEARE:")
            print(f"G = {result.solution.G}")
            print(f"E = {result.solution.E}")
            print(f"T = {result.solution.T}")
            print(f"B = {result.solution.B}")
            print(f"Y = {result.solution.Y}")
            print(f"A = {result.solution.A}")
            print(f"R = {result.solution.R}")
            
            print(f"\nПроверка:")
            G, E, T, B, Y, A, R = result.solution.G, result.solution.E, result.solution.T, result.solution.B, result.solution.Y, result.solution.A, result.solution.R
            get_value = 100*G + 10*E + T
            by_value = 10*B + Y
            beare_value = 10000*B + 1000*E + 100*A + 10*R + E
            print(f"{get_value} x {by_value} = {beare_value}")
            print(f"Результат умножения: {get_value * by_value}")
            print(f"Равенство выполняется: {get_value * by_value == beare_value}")
        else:
            print("Решение не найдено")

    return asyncio.run(solve())

solve_cryptarithmetic()