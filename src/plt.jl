#
#--------------------------------------#
export meshplt
#--------------------------------------#
function meshplt(x,y,u;a=45,b=60)
    p = plot(x,y,u,legend=false,c=:grays,camera=(a,b))
    p = plot!(x',y',u',legend=false,c=:grays,camera=(a,b))
    return p
end
#--------------------------------------#
function meshplt(u,msh::Mesh;a=45,b=60)
    p = meshplt(msh.x,msh.y,u;a=a,b=b)
    return p
end
#--------------------------------------#
function meshplt(f::Field,msh::Mesh;a=45,b=60)
    p = meshplt(f.u,msh;a=a,b=b)
    return p
end
#--------------------------------------#
