#
#----------------------------------------------------------------------
"""
 Data structure to hold mesh information, and operators.

 Q is the global -> local scatter (copying) operator,
 consequently, Q' is local -> global gather (summing) operator,
 and QQ' is the local -> local gather-scatter operator

 R is the restriction operation, that removes

 M = R'R is the Dirichlet boundary condition mask, i.e. it zeros out
 data of ∂Ω_D  

"""
struct Mesh{T}
    nr::Int
    ns::Int
    Ex::Int
    Ey::Int

    zr::Array{T,1} # interpolation points
    zs::Array{T,1}

    QQtx::Array{T,2} # gather scatter op (loc -> loc)
    QQty::Array{T,2}
    mult::Array{T} # weights for inner product
    l2g ::Array{T} # local to global mapping

end
#--------------------------------------#
function Mesh(nr::Int,ns::Int,Ex::Int,Ey::Int
             ,ifperiodic=[false,false]
             ,deform=fixU)

    # mappings
    # Q : glo -> loc (scatter)
    # Q': loc -> glo (gather)
    Qx = semq(Ex,nr,ifperiodic[1])
    Qy = semq(Ex,ns,ifperiodic[2])
    QQtx = Qx*Qx'
    QQty = Qy*Qy'

    (nxl,nxg) = size(Qx)
    (nyl,nyg) = size(Qy)

    l2g = Array(1:nxg*nyg)
    l2g = reshape(l2g,nxg,nyg)
#   l2g = ABu(Qy,Qx,l2g) # shape of local with global indices


    # inner product weights
    # (u,v) = sum(u .* v .* mult)
    mult = ones(nr*Ex,ns*Ey)
    mult = gatherScatter(mult,QQtx,QQty)
    mult = @. 1 / mult

    return Mesh{Float64}(nr,ns,Ex,Ey
                        ,deform,ifperiodic
                        ,zr,zs,wr,ws,Dr,Ds,x,y
                        ,QQtx,QQty,mult,l2g
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
struct Field{T,K}
    u ::Array{T}     # value
    ub::Array{T}     # boundary data
    M ::Array{T}     # BC mask
end
#----------------------------------------------------------------------
