#
#--------------------------------------#
export grad
#--------------------------------------#
"""
 Compute gradient of u∈H¹(Ω).

 Continuity isn't enforced across
 element boundaries for gradients

 [Dx] * u = [rx sx] * [Dr] * u
 [Dy]     = [ry sy]   [Ds]

"""
function grad(u  ::Array
             ,msh::Mesh)

    @unpack Dr,Ds,rx,ry,sx,sy = msh

    ux,uy = grad(u,Dr,Ds,rx,ry,sx,sy)

    return ux,uy
end
#--------------------------------------#
function grad(u,Dr,Ds,rx,ry,sx,sy)

  ur = ABu([],Dr,u)
  us = ABu(Ds,[],u)
  
  ux = @. rx * ur + sx * us
  uy = @. ry * ur + sy * us
  
  return ux,uy
end
#--------------------------------------#
export gradᵀ
#--------------------------------------#
"""
 Gradient transpose operator

 [Dx' Dy'] = [Dr' Ds'] [rx ry]
                       [sx sy]
"""
function gradᵀ(u  ::Array
              ,msh::Mesh)

    @unpack Dr,Ds,rx,ry,sx,sy = msh

    ux,uy = gradᵀ(u,Dr,Ds,rx,ry,sx,sy)

    return ux,uy
end
#--------------------------------------#
function gradᵀ(u,Dr,Ds,rx,ry,sx,sy)

  ux = ABu(Dr',[] ,rx .* u) +
       ABu([] ,Ds',sx .* u)

  uy = ABu(Dr',[] ,ry .* u) +
       ABu([] ,Ds',sy .* u)

  return ux,uy
end
#--------------------------------------#
