#
#----------------------------------------------------------------------
export Mesh
#----------------------------------------------------------------------
#
mutable struct Mesh
    nr::Int
    ns::Int
    Ex::Int
    Ey::Int

    deform::Function
    ifperiodic::Array{Bool,1}

    zr::Array{Float64,1} # interpolation points
    zs::Array{Float64,1}
    wr::Array{Float64,1} # interpolation weights
    ws::Array{Float64,1}
    Dr::Array{Float64,2} # differentiation matrix
    Ds::Array{Float64,2}

    x::Array{Float64,2} # grid
    y::Array{Float64,2}

    QQtx::Array{Float64,2} # gather scatter op (loc -> loc)
    QQty::Array{Float64,2}
    mult::Array{Float64,2} # weights for inner product

    Jac ::Array{Float64,2} # jacobian
    Jaci::Array{Float64,2}

    rx::Array{Float64,2} # dr/dx
    ry::Array{Float64,2}
    sx::Array{Float64,2}
    sy::Array{Float64,2}

    B ::Array{Float64,2} # mass matrix
    Bi::Array{Float64,2}

    G11::Array{Float64,2} # lapl
    G12::Array{Float64,2}
    G22::Array{Float64,2}

end

function Mesh(nr::Int,ns::Int,Ex::Int,Ey::Int
             ,deform::Function,ifperiodic::Array{Bool,1})

    zr,wr = FastGaussQuadrature.gausslobatto(nr)
    zs,ws = FastGaussQuadrature.gausslobatto(ns)

    Dr  = derivMat(zr)
    Ds  = derivMat(zs)

    # mappings
    # Q: global -> local op, Q': local -> global
    Qx = semq(Ex,nr,ifperiodic[1])
    Qy = semq(Ex,ns,ifperiodic[2])
    QQtx = Qx*Qx'
    QQty = Qy*Qy'
    mult = ones(nr*Ex,ns*Ey)
    mult = gatherScatter(mult,QQtx,QQty)
    mult = @. 1 / mult

    xe,_ = semmesh(Ex,nr)
    ye,_ = semmesh(Ey,ns)
    x,y  = ndgrid(xe,ye)

    # mesh deformation
    x,y = deform(x,y)

    # jacobian
    Jac,Jaci,rx,ry,sx,sy = jac(x,y,Dr,Ds)

    # diagonal mass matrix
    wx = kron(ones(Ex,1),wr)
    wy = kron(ones(Ey,1),ws)

    B  = Jac .* (wx*wy')
    Bi = 1   ./ B

    # Lapl solve
    G11 = @. B * (rx * rx + ry * ry)
    G12 = @. B * (rx * sx + ry * sy)
    G22 = @. B * (sx * sx + sy * sy)

    return Mesh(nr,ns,Ex,Ey
               ,deform,ifperiodic
               ,zr,zs,wr,ws,Dr,Ds,x,y
               ,QQtx,QQty,mult
               ,Jac,Jaci
               ,rx,ry,sx,sy
               ,B,Bi
               ,G11,G12,G22)
end
#----------------------------------------------------------------------
export Field
#----------------------------------------------------------------------
#
mutable struct Field
    u::Array{Float64,2} # value
    M::Array{Float64,2} # mask
#   msh::Mesh           # underlying mesh
end

function Field(expr::Function,bc::Array{Char,1},msh::Mesh)

    @unpack x,y,nr,ns,Ex,Ey,ifperiodic = msh

    u = @. expr(x,y)
    
    Ix = sparse(I,Ex*nr,Ex*nr);
    Iy = sparse(I,Ey*ns,Ey*ns);

    xIter = collect(1:(Ex*nr))
    yIter = collect(1:(Ey*ns))

    if(bc[1]=='D') xIter = xIter[2:end]   end
    if(bc[2]=='D') xIter = xIter[1:end-1] end
    if(bc[3]=='D') xIter = yIter[2:end]   end
    if(bc[4]=='D') xIter = yIter[1:end-1] end

    if(ifperiodic[1]) xIter = collect(1:(Ex*nr)); end
    if(ifperiodic[2]) yIter = collect(1:(Ey*ns)); end

    Rx = Ix[xIter,:];
    Ry = Iy[yIter,:];

    M = diag(Rx'*Rx) * diag(Ry'*Ry)'
    M = Array(M)

    return Field(u,M)
end

(+)(f::Field) = f
(-)(f::Field) = Field(-f.u,f.M)

(+)(f0::Field,f1::Field) = Field(f0.u+f1.u,max.(f0.M,f1.M))
(-)(f0::Field,f1::Field) = Field(f0.u-f1.u,max.(f0.M,f1.M))

(*)(λ::Number,f::Field) = Field(f.u .* λ,f.M)
(*)(f::Field,λ::Number) = λ * f
(*)(f0::Field,f1::Field) = Field(f0.u .* f1.u, f0.M .* f1.M)

(/)(f0::Field,f1::Field) = Field(f0.u ./ f1.u, f0.M .* f1.M)
(\)(f0::Field,f1::Field) = Field(f0.u .\ f1.u, f0.M .* f1.M)

(^)(λ::Number,f::Field) = Field(λ .^ f.u,f.M)
(^)(f::Field,λ::Number) = Field(f.u .^ λ,f.M)
(^)(f0::Field,f1::Field) = Field(f0.u .^ f1.u, f0.M .* f1.M)

#----------------------------------------------------------------------
#=
mutable struct simulation

#   ifvl = 0    # evolve  vel field per NS eqn
#   ifad = 1    # advect  vel, sclr
#   ifpr = 0    # project vel onto a div-free subspace
#   ifps = 1    # evolve sclr per advection diffusion eqn

    m1::mesh    # velocity mesh
    m2::mesh    # pressure mesh
    md::mesh    # dealiasing mesh

    # mesh interpolation operators
    J1d::Array
    J2d::Array
    J21::Array

end
=#

