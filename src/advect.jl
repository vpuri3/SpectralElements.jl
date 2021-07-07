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
 a grid with higher polynomial order
 for dealiasing (over-integration)

"""
function advect(T  ::Array
               ,ux ::Array
               ,uy ::Array
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
function advect(T   ::Array
               ,ux  ::Array
               ,uy  ::Array
               ,mshV::Mesh 
               ,mshD::Mesh  # dealias
               ,Jr,Js)

    Tx,Ty = grad(T,mshV)

    JTx = ABu(Js,Jr,Tx)
    JTy = ABu(Js,Jr,Ty)
    Jux = ABu(Js,Jr,ux)
    Juy = ABu(Js,Jr,uy)

    JCu   = @. Jux*JTx + Juy*JTy
    JCu .*= mshD.B
    Cu    = ABu(Js',Jr',JCu)

    return Cu
end
#--------------------------------------#
function advect(T   ::Array
               ,ux  ::Array
               ,uy  ::Array
               ,mshV::Mesh 
               ,mshD::Mesh) # dealias
  
    Jr = interpMat(mshD.zr,mshV.zr)
    Js = interpMat(mshD.zs,mshV.zs)

    Cu = advect(T,ux,uy,mshV,mshD,Jr,Js)

    return Cu
end
#--------------------------------------#
