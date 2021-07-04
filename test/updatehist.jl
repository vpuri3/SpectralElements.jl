#
using UnPack, BenchmarkTools

mutable struct Field{T}
    u ::Array{T,2} # value
    u1::Array{T,2} # histories
    u2::Array{T,2}
    u3::Array{T,2}
end

#----------------------------------#
mutable struct Gield{T}
    u ::Array{T}
    uh::Vector{Array}
end

#----------------------------------#
function updateHist!(g::Gield)

    for i=length(g.uh):-1:2
        g.uh[i] .= g.uh[i-1]
    end

    g.uh[1] .= g.u
    return
end
#----------------------------------#
function updateHist!(u::AbstractArray)

    for i=length(u):-1:2
        u[i] .= u[i-1]
    end
    return
end
#----------------------------------#
function updateHist!(fld::Field)
    @unpack u,u1,u2,u3 = fld

    u3[:] .= u2[:]
    u2[:] .= u1[:]
    u1[:] .= u[:]

    return
end
#----------------------------------#
n = 10
a = ones(n,n)
b = Array[ (3-i).*a for i in 1:3]
f = Field(3*a,2*a,1*a,0*a)
g = Gield(3*a,b)
t = [3.,2.,1.,0.]

#println("t: $t")
#@btime updateHist!(t)
#println("t: $t")
println("#==================================#")
println("f: $(f.u[1]), $(f.u1[1]), $(f.u2[1]), $(f.u3[1])")
@btime updateHist!(f)
println("f: $(f.u[1]), $(f.u1[1]), $(f.u2[1]), $(f.u3[1])")
println("#==================================#")
println("g: $(g.u[1]), $(g.uh[1][1]), $(g.uh[2][1]), $(g.uh[3][1])")
@btime updateHist!(g)
println("g: $(g.u[1]), $(g.uh[1][1]), $(g.uh[2][1]), $(g.uh[3][1])")
println("#==================================#")
#
