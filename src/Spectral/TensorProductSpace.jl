#
export TPPSpace
""" Gauss Lobatto Legendre spectral space """
struct TPPSpace{T,vecT,fldT,matT,F} <: AbstractSpectralSpace{T,2}
    z::vecT
    w::vecT

    r::TPPField{T,2,fldT}  # reference coordinates
    s::TPPField{T,2,fldT}

    x::TPPField{T,2,fldT}  # spatial coordinates
    y::TPPField{T,2,fldT}

    mass  ::TPPDiagOp{2,T,fldT} # mass   matrix
    deriv ::TPPOp{2,T,matT}     # deriv  matrix
    interp::TPPOp{2,T,matT}   # interp matrix

    deform::F
end

function TPPSpace(n::Int = 8, T=Float64,
                  deform = (r,s) -> (copy(r), copy(s))
                 )
    z,w = FastGaussQuadrature.gausslobatto(n)
    z = T.(z)
    w = T.(w)

    o = ones(T,n)
    r = z * o' |> TPPField
    s = o * z' |> TPPField

    x,y = deform(r,s)

    B = w * w'      |> TPPDiagOp
    D = derivMat(z) |> TPPOp

    return TPPSpace(z,w,r,s,x,y,B,D, deform)
end
