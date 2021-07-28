#
#--------------------------------------#
export meshplt, meshplt!
#--------------------------------------#
function meshplt(x,y,u;a=45,b=60,c=:grays,kwargs...)
    p = plot(x,y,u,legend=false,c=c,camera=(a,b),kwargs...)
    p = plot!(x',y',u',legend=false,c=c,camera=(a,b),kwargs...)
    return p
end
function meshplt!(x,y,u;a=45,b=60,c=:grays,kwargs...)
    p = plot!(x,y,u,legend=false,c=c,camera=(a,b),kwargs...)
    p = plot!(x',y',u',legend=false,c=c,camera=(a,b),kwargs...)
    return p
end
#--------------------------------------#
function meshplt(u,msh::Mesh;a=45,b=60,kwargs...)
    p = meshplt(msh.x,msh.y,u;a=a,b=b,kwargs...)
    return p
end
#--------------------------------------#
function meshplt(f::Field,msh::Mesh;a=45,b=60)
    p = meshplt(f.u,msh;a=a,b=b)
    return p
end
#--------------------------------------#
