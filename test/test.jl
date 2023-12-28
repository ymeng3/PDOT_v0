import Random 
import Plots
include("../src/PDOT.jl")

ran_seed = 123
Random.seed!(ran_seed)
rng = Random.MersenneTwister(ran_seed)

m, n = 100, 100

C = rand(rng, m, n) * 10
p = abs.(randn(rng, m))
q = abs.(randn(rng, n))
p .= p ./ sum(p)
q .= q ./ sum(q)

problem = OptimalTransportProblem(C, p, q) 

params = PrimalDualOptimizerParameters(
    2 * 60 * 60,  # 2 hours
    1e-4,
    ConstantStepsizeParams(),
)

kkt_stats_res, iter, time_basic, time_full, converged, final_transport_matrix = optimize(problem, params)

# Calculate primal infeasibility
sum_rows = sum(final_transport_matrix, dims=2)  # Sum over columns to get total transport from each source
sum_cols = sum(final_transport_matrix, dims=1)  # Sum over rows to get total transport to each target

source_diff = maximum(abs.(sum_rows[:] .- p))  # Max absolute difference from source distribution
target_diff = maximum(abs.(sum_cols[:] .- q))  # Max absolute difference from target distribution

primal_infeasibility = max(source_diff, target_diff)

println("Primal Infeasibility: $primal_infeasibility")

kkt_plt = Plots.plot()
Plots.plot!(
    1:length(kkt_stats_res), 
    kkt_stats_res, 
    xlabel = "Iterations", 
    ylabel = "KKT Residual", 
    yscale = :log10,
    label = false,
)

Plots.savefig(kkt_plt, joinpath("./test.png"))
