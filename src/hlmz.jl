#
#--------------------------------------#
export hlmz
#--------------------------------------#
"""
 for v,u in H^1 of Omega

 (v,(-ν∇² + k) u )
"""
function hlmz(u::AbstractArray
             ,ν::AbstractArray
             ,k::AbstractArray
             ,msh::Mesh)

    Hu  = ν .* lapl(u,msh)
    Hu += k .* mass(u,msh)

return Hu
end
#--------------------------------------#
# dealiased version
function hlmz(u::AbstractArray
             ,ν::AbstractArray
             ,k::AbstractArray
             ,msh1::Mesh
             ,msh2::Mesh)

    Hu  = ν .* lapl(u,msh1,msh2)
    Hu += k .* mass(u,msh1,msh2)

return Hu
end
#--------------------------------------#
#
