import sympy as sp
import math

def solve_quartic_eq(constants):
    m, n, o, z, r, a  = sp.symbols('m n o z r a', constant=True)

    u = sp.symbols('u')
    eq =(r - (m+a**2 + n*(1-u**2)/(1+u**2) - o*2*u/(1+u**2) ) )**2/(4*a**2) + z**2 - m - n*(1-u**2)/(1+u**2) + o*2*u/(1+u**2)

    m_val = constants[0]
    n_val = constants[1]
    o_val = constants[2]
    z_val = constants[3]
    r_val = constants[4]
    a_val = constants[5]

    eq_subs = eq.subs([(m, m_val), (n, n_val), (o, o_val), (z, z_val), (r, r_val), (a, a_val)])

    simplified_eq = sp.simplify(eq_subs)

    solutions = sp.solve(simplified_eq, u)
    
    sol_1=sp.re(solutions[0])
    sol_2=sp.re(solutions[1])
    sol_3=sp.re(solutions[2])
    sol_4=sp.re(solutions[3])

    return sol_1,sol_2,sol_3,sol_4

def quartic_eq(theta,m,n,o,z,r,a):
    return (r-(m+a**2+n*math.cos(theta)-o*math.sin(theta)))**2/(4*a**2) + z**2 -m-n*math.cos(theta)+o*math.sin(theta)

def solve_quadratic_eq(constants):
    r,a1,f1,f2,k1 = sp.symbols('r a1 f1 f2 k1')

    u=sp.symbols('u')

    eq= r - 2*a1*(f1*(1-u**2)/(1+u**2) - f2*(2*u)/(1+u**2) ) - k1

    eq_subs = eq.subs([(r,constants[0]),(a1,constants[1]),(f1,constants[2]),(f2,constants[3]),(k1,constants[4])])

    simplified_eq = sp.simplify(eq_subs)
    solutions = sp.solve(simplified_eq, u)

    sol1 = sp.re(solutions[0])
    sol2 = sp.re(solutions[1])

    return sol1,sol2


