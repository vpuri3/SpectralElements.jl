#
using UnPack, BenchmarkTools

mutable struct Field{T}
    u ::Array{T,2} # value
    u1::Array{T,2} # histories
    u2::Array{T,2}
    u3::Array{T,2}
end

#----------------------------------#
function updateHist!(u::AbstractArray)

    for i=length(u):-1:2
        u[i] = u[i-1]
    end
    u[1] = 999# u[2] ### garbage
    return
end
#----------------------------------#
function updateHist!(fld::Field)
    @unpack u,u1,u2,u3 = fld

    u3[:,:] .= u2
    u2[:,:] .= u1
    u1[:,:] .= u 

    return
end
#----------------------------------#
a = ones(4,4)
f = Field(3*a,2*a,1*a,0*a)
t = [3.,2.,1.,0.]

println("t: $t")
@time updateHist!(t)
println("t: $t")

println("f: $(f.u[1]), $(f.u1[1]), $(f.u2[1]), $(f.u3[1])")
@time updateHist!(f)
println("f: $(f.u[1]), $(f.u1[1]), $(f.u2[1]), $(f.u3[1])")

#
