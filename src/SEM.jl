#
module SEM

using FastGaussQuadrature

export interpMat
export derivMat

# import ...

include("interpMat.jl")
include("derivMat.jl")

end
