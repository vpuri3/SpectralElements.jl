#
#--------------------------------------#
export hlmz
#--------------------------------------#
"""
 for v,u in H^1 of Omega

 (v,(-ν∇² + k)u)\n
          = ν * a(v,u)
          + k *  (v,u)
"""
function hlmz(u::Array
             ,ν,k,msh::Mesh)

    Hu   = ν .* lapl(u,msh)
    Hu   = Hu .+ k .* mass(u,msh)

return Hu
end
#--------------------------------------#
# dealiased version
function hlmz(u::Array
             ,ν,k
             ,msh1::Mesh
             ,msh2::Mesh)

    Hu = hlmz(u,ν,k,msh1)

return Hu
end
#--------------------------------------#
#
