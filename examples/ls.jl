#
using LinearSolve

"""
 plug into OrdinaryDiffEq -
    - write all functionality of DefaultLinSolve, 
      any of the stuff from DiffEqBase/linear_nonlinear.jl
    - wrap Krylov.jl, IterativeSolvers.jl
"""

A = rand(5, 5)
b = rand(5)
prob = LinearProblem(A, b)

# Factorization
A * solve(prob, LUFactorization() ) ≈ b
A * solve(prob, QRFactorization() ) ≈ b
A * solve(prob, SVDFactorization()) ≈ b
A * solve(prob, KrylovJL(A, b))     ≈ b

