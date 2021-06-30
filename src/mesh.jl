#
#----------------------------------------------------------------------
export Mesh
#----------------------------------------------------------------------
struct Mesh{T,N}
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
    # Q : glo -> loc (scatter)
    # Q': loc -> glo (gather)
    Qx = semq(Ex,nr,ifperiodic[1])
    Qy = semq(Ex,ns,ifperiodic[2])
    QQtx = Qx*Qx'
    QQty = Qy*Qy'

    (nxl,nxg) = size(Qx)
    (nyl,nyg) = size(Qy)

    loc2glo = Array(1:nxg*nyg)
    loc2glo = reshape(loc2glo,nxg,nyg)
    loc2glo = ABu(Qy,Qx,loc2glo) # shape of local with global indices

    # loc2glo = semreshape(loc2glo,nr,ns,Ex,Ey)

    # inner product weights
    # (u,v) = sum(u .* v .* mult)
    mult = ones(nr*Ex,ns*Ey)
    mult = gatherScatter(mult,QQtx,QQty)
    mult = @. 1 / mult
    
    # mult = semreshape(mult,nr,ns,Ex,Ey)

    xe,_ = semmesh(Ex,nr)
    ye,_ = semmesh(Ey,ns)
    x,y  = ndgrid(xe,ye)

    # x = semreshape(x   ,nr,ns,Ex,Ey)
    # y = semreshape(y   ,nr,ns,Ex,Ey)


    # deform Ω = [-1,1]²
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
export generateMask
#----------------------------------------------------------------------
"""
 bc = ['D','N','D','D'] === BC at [xmin,xmax,ymin,ymax]

 'D': Hom. Dirichlet = zeros ∂Ω data\n
 'N': Hom. Neumann   = keeps ∂Ω data

 A periodic mesh overwrites 'D' to 'N' in direction of periodicity.

 To achieve inhomogeneous Dirichlet condition, apply the formulation
 u = ub + uh, where uh is homogeneous part, and ub is an arbitrary
 smooth function on Ω. Then, solve for uh

"""
function generateMask(bc::Array{Char,1},msh::Mesh)

    @unpack nr,ns,Ex,Ey,ifperiodic = msh

    Ix = sparse(I,Ex*nr,Ex*nr)
    Iy = sparse(I,Ey*ns,Ey*ns)

    ix = collect(1:(Ex*nr))
    iy = collect(1:(Ey*ns))

    if(bc[1]=='D') ix = ix[2:end]   end
    if(bc[2]=='D') ix = ix[1:end-1] end
    if(bc[3]=='D') iy = iy[2:end]   end
    if(bc[4]=='D') iy = iy[1:end-1] end

    if(ifperiodic[1]) ix = collect(1:(Ex*nr)); end
    if(ifperiodic[2]) iy = collect(1:(Ey*ns)); end

    Rx = Ix[ix,:]
    Ry = Iy[iy,:]

    M = diag(Rx'*Rx) * diag(Ry'*Ry)'
    M = Array(M) .== 1

    # M = semreshape(M,nr,ns,Ex,Ey)
    return M
end
#----------------------------------------------------------------------
export Field
#----------------------------------------------------------------------
struct Field{T,N}
    u ::Array{T,N} # value
    u1::Array{T,N} # histories
    u2::Array{T,N}
    u3::Array{T,N}
    ub::Array{T,N} # boundary data
    M ::Array{T,N} # BC mask

    mshRef::Ref{Mesh{T,N}} # underlying mesh
end

function Field(bc::Array{Char,1},msh::Mesh{T,N}) where {T,N}
    u  = zero(msh.x)
    u1 = copy(u)
    u2 = copy(u)
    u3 = copy(u)
    ub = copy(u)
    M  = generateMask(bc,msh)
    return Field{T,N}(u,u1,u2,u3,ub,M,Ref(msh))
end
#----------------------------------------------------------------------
export updateHist!
#--------------------------------------#
function updateHist!(fld::Field)
    @unpack u,u1,u2,u3 = fld

    u3[:,:] .= u2
    u2[:,:] .= u1
    u1[:,:] .= u 

    return
end

function updateHist!(u::AbstractArray)

    for i=length(u):-1:2
        u[i] = u[i-1]
    end
    u[1] = u[2]
    return
end
#--------------------------------------#
