#
using OrdinaryDiffEq, Test, Random
Random.seed!(123)
using OrdinaryDiffEq, DiffEqOperators, LinearAlgebra

"""
GOALS

1. have defaultlinsolve use predefined ldiv! if available on that type
    - allow for `Diagonal`, `LinearAlgebra.I`

2. add misc shit in each Krylov iteration call: https://github.com/SciML/DiffEqBase.jl/issues/366

3. Make SEM.jl operators AbstractDiffEqOperators

4. how to provide inverse to mass_matrix

5. https://github.com/SciML/OrdinaryDiffEq.jl/issues/1430

6. where is mass_matrix being inverted??? how to set preconditioner, 
"""

#=============================================#
"""
 OrdinaryDiffEq/test/interface/linear_nonlinear_tests.jl
"""
n = 10
A = 0.01*rand(n,n)
A = Tridiagonal(-ones(n-1),2ones(n),-ones(n-1))
rn = (du, u, p, t) -> begin
    mul!(du, A, u)
end
u0 = rand(n)
prob = ODEProblem(ODEFunction(rn, jac_prototype=JacVecOperator{Float64}(rn, u0; autodiff=false)), u0, (0, 10.))
#---------------------------------------------
""" GMRES """
sol = solve(prob, QNDF(autodiff=false, linsolve=LinSolveGMRES()));
@show sol.retcode
#---------------------------------------------
""" CG """
#sol = solve(prob, QNDF(autodiff=false, linsolve=LinSolveCG()));
#@show sol.retcode
#=============================================#
"""
 OrdinaryDiffEq/test/interface/mass_matrix_tests.jl
"""

#=============================================#
return
