#
#--------------------------------------#
export mask
#--------------------------------------#
"""
 Mu = (R'*R)*u

 masks dirichlet boundary points
"""
function mask(u::Array
             ,M::Array)

if(length(M)==0) Mu = copy(u)
else             Mu = @. M*u
end

return Mu
end
#--------------------------------------#
