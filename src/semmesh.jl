#
export semmesh
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

