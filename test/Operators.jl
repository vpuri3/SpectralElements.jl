#
using SpectralElements.Spectral

nr = 8
ns = 12

u = rand(nr,ns) |> Field
v = rand(nr,ns) |> Field

# ComposeOperator

# InverseOperator

# DiagonalOp
d = rand(nr,ns) |> Field
D = DiagonalOp(d)

mul!(v,D,u)
@test v ≈ d .* u

# TensorProd2DOp
Ar = rand(nr,nr)
Bs = rand(ns,ns)
T = TensorProd2DOp(Ar,Bs)

mul!(v,T,u)
@test v ≈ Field(Ar * u.array * Bs')
mul!(v,T,u)
@test v ≈ Field(Ar * u.array * Bs')

