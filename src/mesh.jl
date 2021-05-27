#
#----------------------------------------------------------------------
export Mesh
#----------------------------------------------------------------------
#
mutable struct Mesh{T,N}
    nr::Int
    ns::Int
    Ex::Int
    Ey::Int

    deform::Function
    ifperiodic::Array{Bool,1}

    zr::Array{T,1} # interpolation points
    zs::Array{T,1}
    wr::Array{T,1} # interpolation weights
    ws::Array{T,1}
    Dr::Array{T,2} # differentiation matrix
    Ds::Array{T,2}

    x::Array{T,N} # grid
    y::Array{T,N}

    QQtx::Array{T,2} # gather scatter op (loc -> loc)
    QQty::Array{T,2}
    mult::Array{T,N} # weights for inner product

    Jac ::Array{T,N} # jacobian
    Jaci::Array{T,N}

    rx::Array{T,N} # dr/dx
    ry::Array{T,N}
    sx::Array{T,N}
    sy::Array{T,N}

    B ::Array{T,N} # mass matrix
    Bi::Array{T,N}

    G11::Array{T,N} # lapl
    G12::Array{T,N}
    G22::Array{T,N}

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

    return Mesh{Float64,2}(nr,ns,Ex,Ey
                          ,deform,ifperiodic
                          ,zr,zs,wr,ws,Dr,Ds,x,y
                          ,QQtx,QQty,mult
                          ,Jac,Jaci
                          ,rx,ry,sx,sy
                          ,B,Bi
                          ,G11,G12,G22)
end
#----------------------------------------------------------------------
export generate_mask
#----------------------------------------------------------------------
#
function generate_mask(bc::Array{Char,1},msh::Mesh)

    @unpack nr,ns,Ex,Ey,ifperiodic = msh

    Ix = sparse(I,Ex*nr,Ex*nr)
    Iy = sparse(I,Ey*ns,Ey*ns)

    xIter = collect(1:(Ex*nr))
    yIter = collect(1:(Ey*ns))

    if(bc[1]=='D') xIter = xIter[2:end]   end
    if(bc[2]=='D') xIter = xIter[1:end-1] end
    if(bc[3]=='D') yIter = yIter[2:end]   end
    if(bc[4]=='D') yIter = yIter[1:end-1] end

    if(ifperiodic[1]) xIter = collect(1:(Ex*nr)); end
    if(ifperiodic[2]) yIter = collect(1:(Ey*ns)); end

    Rx = Ix[xIter,:]
    Ry = Iy[yIter,:]

    M = diag(Rx'*Rx) * diag(Ry'*Ry)'
    M = Array(M)
    return M
end
#----------------------------------------------------------------------

#=
#----------------------------------------------------------------------
export Field
#----------------------------------------------------------------------
#
mutable struct Field{T,N}
    u::Array{T,N} # value
    M::Array{T,N} # mask
end

function Field(expr::Function,bc::Array{Char,1},msh::Mesh)
    u = @. expr(msh.x,msh.y)
    M = generate_mask(bc,msh)
    return Field{Float64,2}(u,M)
end

function Field(u::Array{Float64,2},bc::Array{Char,1},msh::Mesh)
    M = generate_mask(bc,msh)
    return Field{Float64,2}(u,M)
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

# todo: function application, arithmatic ops with fields and Arrays (like B)

#u = Field( (x,y) -> sin(π*x)*sin(π*y),bc,m1)
#plt = meshplt(u,m1)
#display(plt)

=#

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
