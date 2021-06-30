#
#--------------------------------------#
export semmesh
#--------------------------------------#
#
# creates 1D sem mesh of poly. order n-1
# and e elements [-1,1].
#
function semmesh(E,n)

z0,w0 = FastGaussQuadrature.gausslobatto(n); # [-1,1]

z0 = 0.5 .* (z0.+1); # [0,1]
w0 = 0.5 .*  w0;

ze = linspace(-1,1,E+1); # element mesh
dz = diff(ze);

z = kron(dz,z0) + kron(ze[1:end-1],ones(n,1));

w = kron(dz,w0);

z = reshape(z,E*n);
w = reshape(w,E*n);

return z,w 
end
#--------------------------------------#
export semreshape
#--------------------------------------#
function semreshape(u,nr,ns,Ex,Ey)

    v = zeros(nr,ns,Ex*Ey)

    Ix = 1:nr
    Iy = 1:ns

    for i=1:Ex
        for j=1:Ey

            ie = j + (i-1)*Ey

            ix = Ix .+ (i-1)*nr
            iy = Iy .+ (j-1)*ns

            v[:,:,ie] = @view u[ix,iy]
        end
    end

    return v
end
#--------------------------------------#
