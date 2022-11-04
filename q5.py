"""Solve Q5 with sympy"""
import sympy
# define symbols
p, n, psi, L_p, L_psi = sympy.symbols("p n psi L_p L_psi", real=True)

# sub in p=psi^4
p = psi**4

# likelihood of p: binomial with N=1
L_p: sympy.Symbol = p**n * (1-p)**(1-n)
print(L_p)

# likelihood transformed so now it's in terms of psi
L_psi: sympy.Symbol = L_p * sympy.diff(p, psi)

# take derivative
dll_dpsi = sympy.diff(sympy.ln(L_psi), psi)

print(dll_dpsi.simplify())
# calculate I(psi)
I_psi: sympy.Symbol = L_psi.subs({n:0}) * dll_dpsi.subs({n:0})**2 +\
    L_psi.subs({n:1}) * dll_dpsi.subs({n:1})**2

print(sympy.latex(I_psi.simplify()))

