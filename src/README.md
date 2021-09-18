## SEM.jl

Partial Differential Equation solver based on the spectral elemenet method.


 TODO
 -pick a better name for package

 -use a packaged iterative solver (IterativeSolvers.jl)
       -overwrite A*x, A'x, <x,y>
   
 -replace ndgrid with broadcast arrays or something

 -use NNlib's optimized gather scatter

 -could help extend code to unstructued grids (big win!!)

 -mesh: x(nx,ny,E). implement ABu for this

 -parametrize to Float64, Float32

 -linsolve function is redundant since \ (backslash) already has an adjoint

 -use StaticArrays.jl, LazyArrays.jl for performance

 -profile code, use nonallocating functions everywhere to imrpove efficiency

 -check out gridap.jl, moose.jl, climacore.jl, trixi.jl

 - obtain CFL values, and do adaptive time-stepping

