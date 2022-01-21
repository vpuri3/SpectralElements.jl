#
import ..activate_env
dir = "../examples"
activate_env(dir)
files = [
         "p2d.jl",
        ]
for file in files
    @testset "$file" begin include("$dir/$file") end
end
