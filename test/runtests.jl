using SEM
using Test

@testset "SEM.jl" begin

   using SEM.Spectral
   u = Spectral.TPPField(rand(10,10))
   v = u'
   D = Spectral.TPPDiagOp(u)
   v = u .+ u
   @test size(v.u) == (10,10)
end

