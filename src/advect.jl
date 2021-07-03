#
#--------------------------------------#
export advect
#--------------------------------------#
"""
 for v,u,T in H¹₀(Ω)

 (v,(u⃗⋅∇)T) = (v,ux*∂xT + uy*∂yT)\n
            = v' *B*(ux*∂xT + uy*∂yT)\n

 implemented as

 R'R * QQ' * B * (ux*∂xT + uy*∂yT)

 (u⃗⋅∇)T = ux*∂xT + uy*∂yT
        = [ux uy] * [Dx] T
                    [Dx]

 ux,uy, ∇T are interpolated to
 a higher polynomial order grid
 for dealiasing

"""
function advect(T  ::AbstractArray
               ,ux ::AbstractArray
               ,uy ::AbstractArray
               ,msh::Mesh)

    @unpack B = msh

    Tx,Ty = grad(T,msh)

    Cu   = @. ux * Tx + uy * Ty
    Cu .*= B

#   Cu = gatherScatter(Cu,msh)
#   Cu = mask(Cu,M)

    return Cu
end
#--------------------------------------#
function advect(T   ::AbstractArray
               ,ux  ::AbstractArray
               ,uy  ::AbstractArray
               ,msh1::Mesh 
               ,msh2::Mesh) # dealias

    Jr = interpMat(msh2.zr,msh1.zr)
    Js = interpMat(msh2.zs,msh1.zs)

    Tx,Ty = grad(T,msh)

    JTx = ABu(Js,Jr,Tx)
    JTy = ABu(Js,Jr,Ty)
    Jux = ABu(Js,Jr,ux)
    Juy = ABu(Js,Jr,uy)

    JCu   = @. Jux*JTx + Juy*JTy
    JCu .*= msh2.B
    Cu    = ABu(Js',Jr',JCu)

    return Cu
end
#--------------------------------------#
