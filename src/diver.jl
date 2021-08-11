#
#--------------------------------------#
export diver
#--------------------------------------#
"""
 for q ∈ H¹₀(Ω)  (test  pressure)
 for u⃗ ∈ H¹₀(Ω)² (trial velocity)

 (q,∇⋅u⃗) = (J*q)' * Bᵥ * [Dx Dy][ux]
                                [uy]

 Bᵥ is mass matrix on velocity grid
 J  is the interpolation matrix
 from pressure grid to velocity grid.

"""
function diver(ux  ::Array
              ,uy  ::Array
              ,mshV::Mesh
              ,Jr,Js)

    uxdx,_    = grad(ux,msh)
    _   ,uydy = grad(uy,msh)

    div = uxdx + uydy

    Bdiv  = mass(div,msh)
    JBdiv = ABu(Js',Jr',Bdiv)

    return JBdiv # pressure grid
end
#--------------------------------------#
export diverᵀ
#--------------------------------------#
"""
 for p ∈ H¹₀(Ω)  (trial pressure)
 for v⃗ ∈ H¹₀(Ω)² (test  velocity)

 -(v⃗,∇p) = -([vx] .* [px])
            ([vy]    [py])

 integrating by parts,

 -(∇v⃗,p) = ([Dx Dy] * [vx])' Bᵥ * J * p
           (          [vy])

         + surface terms (vanish)

         = [vx vy] *  [Dx'] Bᵥ * J * p
                      [Dy']

"""
function diverᵀ(pr  ::Array
               ,mshV::Mesh
               ,Jr,Js)
  
    Jp  = ABu(Js,Jr,pr) # pr -> vel gird
    BJp = mass(Jp,mshV)
    
    qx,qy = gradᵀ(BJp,mshV)

    return qx,qy
end
#--------------------------------------#
export pressureOp
#--------------------------------------#
"""
 Schur complement + operator splitting
 operation for stokes solve

 EE = -DD AA^-1 * DD'
"""
function pressureOp(q   ::Array
                   ,mshP::Mesh
                   ,Jr,Js)

    # DD'
    qx,qy = diverᵀ(q,mshV,Jr,Js)

    qx = gatherScatter(qx,mshV)
    qy = gatherScatter(qy,mshV)

    # Ainv approximation
    qx = qx .* mshV.Bi
    qy = qy .* mshV.Bi
    
    qx = gatherScatter(qx,mshV)
    qy = gatherScatter(qy,mshV)

    qx = mask(qx,mshV)
    qy = mask(qy,mshV)

    # DD
    Eq = diver(qx,qy,mshV,Jr,Js)
    Eq = gatherScatter(Eq,mshV)

  return -Eq
end
#--------------------------------------#
