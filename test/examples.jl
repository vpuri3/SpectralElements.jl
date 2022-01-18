#
dir = "../examples"
files = [
         "p2d.jl",
        ]
for file in files
    @testset "$file" begin include("$dir/$file") end
end
