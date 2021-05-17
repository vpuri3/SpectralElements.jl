using Flux, DiffEqFlux, DifferentialEquations
using Statistics, LinearAlgebra, Plots
import Flux.Data: DataLoader

function random_point_in_sphere(dim, min_radius, max_radius)
    distance = (max_radius - min_radius) .* (rand(1) .^ (1.0 / dim)) .+ min_radius
    direction = randn(dim)
    unit_direction = direction ./ norm(direction)
    return distance .* unit_direction
end

function concentric_sphere(dim, inner_radius_range, outer_radius_range,
                           num_samples_inner, num_samples_outer; batch_size = 64)
    data = []
    labels = []
    for _ in 1:num_samples_inner
        push!(data, reshape(random_point_in_sphere(dim, inner_radius_range...), :, 1))
        push!(labels, ones(1, 1))
    end
    for _ in 1:num_samples_outer
        push!(data, reshape(random_point_in_sphere(dim, outer_radius_range...), :, 1))
        push!(labels, -ones(1, 1))
    end
    data = cat(data..., dims=2)
    labels = cat(labels..., dims=2)
    return DataLoader(data |> gpu, labels |> gpu; batchsize=batch_size, shuffle=true,
                      partial=false)
end

diffeqarray_to_array(x) = reshape(gpu(x), size(x)[1:2])

function construct_model(out_dim, input_dim, hidden_dim, augment_dim)
    input_dim = input_dim + augment_dim
    node = NeuralODE(Chain(Dense(input_dim, hidden_dim, relu),
                           Dense(hidden_dim, hidden_dim, relu),
                           Dense(hidden_dim, input_dim)) |> gpu,
                     (0.f0, 1.f0), Tsit5(), save_everystep = false,
                     reltol = 1e-3, abstol = 1e-3, save_start = false) |> gpu
    node = augment_dim == 0 ? node : AugmentedNDELayer(node, augment_dim)
    return Chain((x, p=node.p) -> node(x, p),
                 diffeqarray_to_array,
                 Dense(input_dim, out_dim) |> gpu), node.p |> gpu
end

function plot_contour(model, npoints = 300)
    grid_points = zeros(2, npoints ^ 2)
    idx = 1
    x = range(-4.0, 4.0, length = npoints)
    y = range(-4.0, 4.0, length = npoints)
    for x1 in x, x2 in y
        grid_points[:, idx] .= [x1, x2]
        idx += 1
    end
    sol = reshape(model(grid_points |> gpu), npoints, npoints) |> cpu
    
    return contour(x, y, sol, fill = true, linewidth=0.0)
end

loss_node(x, y) = mean((model(x) .- y) .^ 2)

println("Generating Dataset")

dataloader = concentric_sphere(2, (0.0, 2.0), (3.0, 4.0), 2000, 2000; batch_size = 256)

cb = function()
    global iter += 1
    if iter % 10 == 0
        println("Iteration $iter || Loss = $(loss_node(dataloader.data[1], dataloader.data[2]))")
    end
end

model, parameters = construct_model(1, 2, 64, 0)
opt = ADAM(0.005)
iter = 0

println("Training Neural ODE")

for _ in 1:10
    Flux.train!(loss_node, Flux.params([parameters, model]), dataloader, opt, cb = cb)
end

plt_node = plot_contour(model)

model, parameters = construct_model(1, 2, 64, 1)
opt = ADAM(0.005)
iter = 0

println()
println("Training Augmented Neural ODE")

for _ in 1:10
    Flux.train!(loss_node, Flux.params([parameters, model]), dataloader, opt, cb = cb)
end

plt_anode = plot_contour(model)
