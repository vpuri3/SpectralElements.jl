#
include("NDgrid.jl")

###
# Lagrange polynomial matrices
###

"""
    Compute the Lagrange interpolation matrix from xi to xo.
    lagrange_poly_interp_mat(xₒ,xᵢ)
"""
function lagrange_poly_interp_mat(xo,xi)

    no = length(xo)
    ni = length(xi)

    a = ones(1,ni)
    for i=1:ni
        for j=1:(i-1)  a[i]=a[i]*(xi[i]-xi[j]); end
        for j=(i+1):ni a[i]=a[i]*(xi[i]-xi[j]); end
    end
    a = 1 ./ a # Barycentric weights

    J = zeros(no,ni)
    s = ones(1,ni)
    t = ones(1,ni)
    for i=1:no
        x = xo[i]
        for j=2:ni
            s[j]      = s[j-1]    * (x-xi[j-1]   )
            t[ni+1-j] = t[ni+2-j] * (x-xi[ni+2-j])
        end
        J[i,:] = a .* s .* t
    end

    return J
end

"""
 Compute derivative matrix for lagrange
 interpolants on points [x]
"""
function lagrange_poly_deriv_mat(x)
    
    n = length(x)

    a = ones(1,n)
    for i=1:n
        for j=1:(i-1) a[i]=a[i]*(x[i]-x[j]) end
        for j=(i+1):n a[i]=a[i]*(x[i]-x[j]) end
    end
    a = 1 ./ a # Barycentric weights

    # diagonal elements
    D = x .- x'
    for i=1:n D[i,i] = 1. end
    D = 1 ./ D
    for i=1:n
        D[i,i] = 0.
        D[i,i] = sum(D[i,:])
    end

    # off-diagonal elements
    for j=1:n for i=1:n
        if(i!=j) D[i,j] = a[j] / (a[i]*(x[i]-x[j])) end
    end end

    return D
end

###
# Lagrange polynomial function spaces
###

function LagrangePolySpace1D(domain::AbstractDomain{<:Number,1},
                             n = 16;
                             quadrature = gausslobatto,
                             T = Float64,
                            )
    z, w = quadrature(n)

    D  = lagrange_poly_deriv_mat(z)

    z = T.(z) |> Field
    w = T.(w) |> Field

    grid = z
    mass = DiagonalOp(w)
    grad = D |> MatrixOp

    gather_scatter(n, domain)

    space = SpectralSpace(grid, mass, grad, gather_scatter)
    space = domain.mapping === nothing ? space : deform(space)
end

GaussLobattoLegendre1D(args...; kwargs...) = LagrangePolySpace1D(args...; quadrature=gausslobatto, kwargs...)
GaussLegendre1D(args...; kwargs...) = LagrangePolySpace1D(args...; quadrature=gausslegendre, kwargs...)
GaussChebychev1D(args...; kwargs...) = LagrangePolySpace1D(args...; quadrature=gausschebyshev, kwargs...)

function LagrangePolySpace2D(domain::AbstractDomain{<:Number,2} nr, ns;
                             quadrature = gausslobatto,
                             T = Float64,
                            )
    zr,wr = quadrature(nr)
    zs,ws = quadrature(ns)
    
    zr,wr = T.(zr), T.(wr)
    zs,ws = T.(zs), T.(ws)
    
    x, y = ndgrid(zr,zs)
    grid = (x, y)
    
    Dr = lagrange_poly_deriv_mat(zr)
    Ds = lagrange_poly_deriv_mat(zs)
    
    Ir = Matrix(I, nr, nr) |> sparse
    Is = Matrix(I, ns, ns) |> sparse

    DrOp = TensorProductOp2D(Dr,Is)
    DsOp = TensorProductOp2D(Ir,Ds)
    
    mass  = w * w' |> Field |> DiagonalOp

    grad = [DrOp
            DsOp]

    space = SpectralSpace(grid, mass, grad, gather_scatter)
    space = domain.mapping === nothing ? space : deform(space)
end

GaussLobattoLegendre2D(args...; kwargs...) = LagrangePolySpace2D(args...; quadrature=gausslobatto, kwargs...)
GaussLegendre2D(args...; kwargs...) = LagrangePolySpace2D(args...; quadrature=gausslegendre, kwargs...)
GaussChebychev2D(args...; kwargs...) = LagrangePolySpace2D(args...; quadrature=gausschebyshev, kwargs...)
#
