from sympy import *
from sympy.physics.quantum import Dagger
init_printing(use_unicode=True)

# Need the following to fold the conjugates. See
# https://stackoverflow.com/questions/48754975/simplification-of-derivative-of-square-using-sympy

from sympy.core.rules import Transform
fold_conjugates = Transform(lambda f: 2*re(f.args[0]),
                            lambda f: isinstance(f, Add) and len(f.args) == 2 and f.args[1] == f.args[0].conjugate())

fold_conjugates_2 = Transform(lambda f: f.args[0] + 2*re(f.args[1]),
                            lambda f: isinstance(f, Add) and len(f.args) == 3 and f.args[2] == f.args[1].conjugate())

I, Q, U, V, l, m, n = symbols('I Q U V l m n', real=True)
Jxxi, Jxyi, Jyxi, Jyyi = symbols("Jxxi Jxyi Jyxi Jyyi")
Ji = Matrix([[Jxxi, Jxyi], [Jyxi, Jyyi]])
Jxxj, Jxyj, Jyxj, Jyyj = symbols("Jxxj Jxyj Jyxj Jyyj")
Jj = Matrix([[Jxxj, Jxyj], [Jyxj, Jyyj]])

Jxx, Jxy, Jyx, Jyy = symbols("Jxx Jxy Jyx Jyy")
J = Matrix([[Jxx, Jxy], [Jyx, Jyy]])

Vxxij, Vxyij, Vyxij, Vyyij = symbols("Vxxij Vxyij Vyxij Vyyij")

Vij = Matrix([[Vxxij, Vxyij], [Vyxij, Vyyij]])

Cij = J * Vij * Dagger(J)

pprint(Cij)

pprint(collect(Cij, Vxxij))

#print(Cij.det())

