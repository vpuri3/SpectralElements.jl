#
using SpectralElements.Spectral

nr = 8
ns = 12

u = rand(nr,ns) |> Field

@inferred similar(u)
@inferred +u
@inferred -u
@inferred 2u
@inferred 2 .+ u
@inferred 2 .- u
@inferred u .+ 2
@inferred u .- 2

