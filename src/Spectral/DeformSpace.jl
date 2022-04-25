#
"""
Deform domain, compute Jacobian of transformation, and its inverse

x = x(r,s), y = y(r,s)

J = [xr xs],  Jinv = [rx ry]
    [yr ys]          [sx sy]

[Dx] * u = [rx sx] * [Dr] * u
[Dy]     = [ry sy]   [Ds]

⟹
[1 0] = [rx sx] *  [Dr] [x y]
[0 1]   [ry sy]    [Ds]

⟹
                 -1
[rx sx] = [xr yr]
[ry sy]   [xs ys]

"""
struct DeformedSpace{T,D,Tspace<:AbstractSpace{T,D},
                     Tgrid,Tjacmat,Tjacimat,Tjac,Tjaci} <: AbstractSpace{T,D}
    space::Tspace
    grid::Tgrid # (x1, ..., xD,)
    dXdR::Tjacmat
    dRdX::Tjacimat
    J::Tjac
    Ji::Tjaci
end

function deform(space::AbstractSpace{<:Number,D}, mapping = nothing) where{D}
    if mappping === nothing
        J    = IdentityOp{D}()
        Jmat = Diagonal([J for i=1:D])
        return DeformedSpace(space, grid(space), Jmat, Jmat, J, J)
    end

    R = grid(space)
    X = mapping(R...)

    gradR = gradOp(space)

    dXdR = gradR.(X)
    dXdR = hcat(dXdR)'
    dXdR = DiagonalOp.(dXdR)

    J  = det(dXdR)
    Ji = 1 / J

    DeformedSpace(space, X, dXdR, dRdX, J, Ji)
end

#function deform(space::AbstractSpace{<:Number,2}, mapping = (r,s) -> (r,s))
#    r, s = grid(space)
#    x, y = mapping(r, s)
#
#    grad = gradOp(space)
#
#    dxdr, dxds = grad * x
#    dydr, dyds = grad * y
#
#    # Jacobian matrix
#    dXdR = DiagonalOp.([dxdr dxds
#                        dydr dyds])
#
#    J  = det(dXdR) # dxdr * dyds - dxds * dydr
#    Ji = 1 / J
#
#    drdx =  (Ji * dyds)
#    drdy = -(Ji * dxds)
#    dsdx = -(Ji * dydr)
#    dsdy =  (Ji * dxdr)
#
#    dRdX = DiagonalOp.[drdx dsdx
#                       drdy dsdy]
#
#    DeformedSpace(space, (x,y), dXdR, dRdX, J, Ji)
#end

function grid(space::DeformedSpace)
    space.grid
end

"""
[Dx] * u = [rx sx] * [Dr] * u
[Dy]     = [ry sy]   [Ds]
"""
function gradOp(space::DeformedSpace)
    gradR = gradOp(space.space)
    dRdX  = space.dRdX
    gradX = dRdX * gradR

    return first(gradX)
end

function massOp(space::DeformedSpace)
    M = massOp(space.space)
    J = space.J

    M * J
end

function laplaceOp(space::DeformedSpace)
    gradR = gradOp(space.space)

    mass = MassOp(space)
    dRdX = jacOp(space) # space.dRdX

    M = Diagonal([mass, mass])
    G = dRdX' * M * dRdX |> Symmetric

    laplOp = @. gradR' .∘ G .∘ gradR

    return first(laplOp)
end
#
