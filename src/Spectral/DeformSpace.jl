#
"""
Deform domain, compute Jacobian of transformation, and its inverse

given

x = x(r,s), y = y(r,s)

compute

dXdR = [xr xs],
       [yr ys] 

dRdX = [rx ry],
       [sx sy]

J = det(dXdR),

Jinv = det(dRdX)

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

    """
    dXdR = [dx1/dr1 ... dx1/drD]
           [...     ...     ...]
           [dxD/dr1 ... dxD/drD]
    """
    dXdR = begin
        dXdR = gradR.(X) |> hcat
        dXdR = DiagonalOp.(dXdR)
        dXdR = dXdR'
    end

    J = if D == 1
        dXdR[1]
    elseif D == 2
        xr = dXdR[1]
        yr = dXdR[2]
        xs = dXdR[3]
        ys = dXdR[4]

        xr * ys - xs * yr
    elseif D == 3
        xr = dXdR[1]; yr = dXdR[2]; zr = dXdR[3]
        xs = dXdR[4]; ys = dXdR[5]; zs = dXdR[6]
        xt = dXdR[7]; yt = dXdR[8]; zt = dXdR[9]

        J = xr * (ys * zt - zs * yt) -
            xs * (yr * zt - zr * yt) +
            xt * (yr * zs - zr * ys)
    else
        det(dXdR) # errors
    end

    Ji = inv(D)

    dRdX = if D == 1
        fill(Ji, 1, 1)
    elseif D == 2
        xr = dXdR[1]
        yr = dXdR[2]
        xs = dXdR[3]
        ys = dXdR[4]

        rx =  (Ji * ys)
        ry = -(Ji * xs)
        sx = -(Ji * yr)
        sy =  (Ji * xr)

        dRdX = [rx ry 
                sx sy]
    elseif D == 3 # cramer's rule
        inv(dXdR)
    else
        inv(dXdR) # errors
    end

    DeformedSpace(space, X, dXdR, dRdX, J, Ji)
end

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

function laplaceOp(space::DeformedSpace{<:Number, D}) where{D}
    gradR = gradOp(space.space)

    mass = MassOp(space)
    dRdX = jacOp(space) # space.dRdX

    M = Diagonal([mass for i=1:D])
    G = dRdX' * M * dRdX |> Symmetric

    laplOp = gradR' .∘ G .∘ gradR

    return first(laplOp)
end
#
