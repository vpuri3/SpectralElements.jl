#
export mesh
#
#
#
function mesh(x,y,u,a=45,b=60)
    p = plot(x,y,u,legend=false,c=:grays,camera=(a,b));
    p = plot!(x',y',u',legend=false,c=:grays,camera=(a,b));
    return p
end

