using SpectralElements
using Test

@testset "SpectralElements.jl" begin

   using SpectralElements.Spectral
   u = Spectral.TPPField(rand(10,10))
   v = u'
   D = Spectral.TPPDiagOp(u)
   v = u .+ u
   @test size(v.u) == (10,10) # broadcasting works ok

   space = GLL2D(8)
end

