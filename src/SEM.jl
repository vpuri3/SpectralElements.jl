#
module SEM

using FastGaussQuadrature

export interpMat
export derivMat
export ndgrid

# import ...

include("interpMat.jl")
include("derivMat.jl")
include("ndgrid.jl")

end
