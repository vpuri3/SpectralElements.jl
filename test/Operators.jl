#
using SpectralElements.Spectral

nr = 8
ns = 12

u = rand(nr,ns) |> Field
v = rand(nr,ns) |> Field

## IdentityOp

## CompositeOperator

## ComposeOperator

## InverseOperator

## DiagonalOp
d = rand(nr,ns) |> Field
D = DiagonalOp(d)

@test mul!(v,D,u) ≈ d .* u

## TensorProd2DOp

Ar = rand(nr,nr)
Bs = rand(ns,ns)
T = TensorProd2DOp(Ar,Bs)

@test mul!(v,T,u) ≈ Field(Ar * u.array * Bs')
@test mul!(v,T,u) ≈ Field(Ar * u.array * Bs')

