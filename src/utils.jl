
mutable struct OptimalTransportProblem 
    cost_matrix::Matrix{Float64}
    source_distribution::Vector{Float64}
    target_distribution::Vector{Float64}   
end 

struct ConstantStepsizeParams end

# mutable struct PrimalDualOptimizerParameters 
#     initial_primal_solution::Matrix{Float64}
#     initial_dual_source_solution::Vector{Float64}
#     initial_dual_target_solution::Vector{Float64}
#     step_size_policy_params::ConstantStepsizeParams
#     iteration_limit::Int64 
#     kkt_tolerance::Float64
# end 

mutable struct PrimalDualOptimizerParameters 
    time_limit::Int 
    kkt_tolerance::Float64
    step_size_policy_params::ConstantStepsizeParams
end 

mutable struct SolutionAverage
    sum_primal_solution::Matrix{Float64}
    sum_primal_row_sum::Vector{Float64}
    sum_primal_col_sum::Vector{Float64}
    sum_dual_source_solution::Vector{Float64}
    sum_dual_target_solution::Vector{Float64}
    sum_primal_gradient::Matrix{Float64}
    sum_weight::Float64
end

@enum SolutionStatus STATUS_OPTIMAL STATUS_ITERATION_LIMIT

mutable struct PrimalDualSolverState 
    current_primal_solution::Matrix{Float64}
    current_primal_row_sum::Vector{Float64}
    current_primal_col_sum::Vector{Float64}
    old_primal_row_sum::Vector{Float64}
    old_primal_col_sum::Vector{Float64}
    dual_source_solution::Vector{Float64}
    dual_target_solution::Vector{Float64}
    primal_gradient::Matrix{Float64}
    solution_avg::SolutionAverage
    step_size_primal::Float64
    step_size_dual::Float64
end 


function initialize_solver_state(
    problem::OptimalTransportProblem,
    initial_primal_solution::Matrix{Float64},
    initial_dual_source_solution::Vector{Float64},
    initial_dual_target_solution::Vector{Float64},
    step_size_policy_params::ConstantStepsizeParams,
)
    m, n = size(initial_primal_solution)

    if step_size_policy_params isa ConstantStepsizeParams
        step_size_primal, step_size_dual = 1/sqrt(m+n), 1/sqrt(m+n)
    end

    primal_gradient = (problem.cost_matrix .- initial_dual_source_solution) .- initial_dual_target_solution'

    return PrimalDualSolverState(
        initial_primal_solution,
        sum(initial_primal_solution, dims=2),
        sum(initial_primal_solution, dims=1),
        zeros(m),
        zeros(n),
        initial_dual_source_solution,
        initial_dual_target_solution,
        primal_gradient,
        initial_solution_avg(n,m),
        step_size_primal,
        step_size_dual,
    )
end

mutable struct OTCache
    norm_cost_matrix::Float64
    norm_right_hand_side::Float64
end

mutable struct KKTResidual
    primal_residual::Float64
    dual_residual::Float64
    primal_dual_gap::Float64
    kkt_residual::Float64
    rel_primal_residual::Float64
    rel_dual_residual::Float64
    rel_primal_dual_gap::Float64
end

function copy_kkt(res::KKTResidual)
    return res
end

function compute_kkt_residual(
    problem::OptimalTransportProblem,
    solver_state::PrimalDualSolverState,
    ot_cache::OTCache,
)
    # # last iterates
    # primal_residual = norm(solver_state.current_primal_row_sum - problem.source_distribution)^2 + norm(solver_state.current_primal_col_sum - problem.target_distribution)^2

    # dual_residual = norm(max.(0, -solver_state.primal_gradient))^2
    
    # primal_obj = dot(problem.cost_matrix, solver_state.current_primal_solution)
    # dual_obj = dot(solver_state.dual_source_solution, problem.source_distribution) + dot(solver_state.dual_target_solution, problem.target_distribution)

    # primal_dual_gap = (primal_obj - dual_obj)^2


    # average iterates
    primal_residual = norm(solver_state.solution_avg.sum_primal_row_sum ./ solver_state.solution_avg.sum_weight .- problem.source_distribution)^2 + norm(solver_state.solution_avg.sum_primal_col_sum ./ solver_state.solution_avg.sum_weight .- problem.target_distribution)^2

    dual_residual = norm(max.(0, -solver_state.solution_avg.sum_primal_gradient))^2 / solver_state.solution_avg.sum_weight^2
    
    primal_obj = dot(problem.cost_matrix, solver_state.solution_avg.sum_primal_solution) / solver_state.solution_avg.sum_weight
    dual_obj = dot(solver_state.solution_avg.sum_dual_source_solution, problem.source_distribution) / solver_state.solution_avg.sum_weight + dot(solver_state.solution_avg.sum_dual_target_solution, problem.target_distribution) / solver_state.solution_avg.sum_weight

    primal_dual_gap = (primal_obj - dual_obj)^2

    return KKTResidual(
        sqrt(primal_residual),
        sqrt(dual_residual),
        sqrt(primal_dual_gap),
        sqrt(primal_residual + dual_residual + primal_dual_gap),
        sqrt(primal_residual) / (1 + ot_cache.norm_right_hand_side),
        sqrt(dual_residual) / (1 + ot_cache.norm_cost_matrix),
        sqrt(primal_dual_gap) / (1 + abs(primal_obj) + abs(dual_obj)),
    )
end



function initialize_solution_avg(
    primal_size::Int64,
    dual_size::Int64,
)
    return SolutionAverage(
        zeros(dual_size,primal_size),
        zeros(dual_size),
        zeros(primal_size),
        zeros(dual_size),
        zeros(primal_size),
        zeros(dual_size,primal_size),
        0.0,
    )
end

function reset_solution_avg!(
    solver_state::PrimalDualSolverState,
)
    solver_state.solution_avg.sum_primal_solution .= 0.0
    solver_state.solution_avg.sum_primal_row_sum .= 0.0
    solver_state.solution_avg.sum_primal_col_sum .= 0.0
    solver_state.solution_avg.sum_dual_source_solution .= 0.0
    solver_state.solution_avg.sum_dual_target_solution .= 0.0
    solver_state.solution_avg.sum_primal_gradient .= 0.0
    solver_state.solution_avg.sum_weight = 0.0
end

mutable struct BufferFeas
    solution::Matrix{Float64}
    err_r::Vector{Float64}
    err_c::Vector{Float64}
end

function rounding!(
    problem::OptimalTransportProblem,
    solver_state::PrimalDualSolverState, 
    buffer_feas::BufferFeas,
)
    buffer_feas.solution .= solver_state.solution_avg.sum_primal_solution ./ solver_state.solution_avg.sum_weight
    buffer_feas.solution .= LinearAlgebra.Diagonal(min.(problem.source_distribution./(solver_state.solution_avg.sum_primal_row_sum ./ solver_state.solution_avg.sum_weight),1)) * buffer_feas.solution

    buffer_feas.solution .= buffer_feas.solution * LinearAlgebra.Diagonal(min.(problem.target_distribution./sum(buffer_feas.solution, dims=1),1))

    buffer_feas.err_r .= problem.source_distribution .- vec(sum(buffer_feas.solution, dims=2))
    buffer_feas.err_r ./= norm(buffer_feas.err_r, 1)
    buffer_feas.err_c .= problem.target_distribution .- vec(sum(buffer_feas.solution, dims=1))

    buffer_feas.solution .+= buffer_feas.err_r * buffer_feas.err_c'
end

function compute_kkt_residual(
    problem::OptimalTransportProblem,
    solver_state::PrimalDualSolverState,
    buffer_feas::BufferFeas,
    ot_cache::OTCache,
)
    
    primal_residual = norm(vec(sum(buffer_feas.solution, dims=2)) .- problem.source_distribution)^2 + norm(vec(sum(buffer_feas.solution, dims=1)) .- problem.target_distribution)^2

    dual_residual = norm(max.(0, -solver_state.solution_avg.sum_primal_gradient))^2 / solver_state.solution_avg.sum_weight^2
    
    primal_obj = dot(problem.cost_matrix, buffer_feas.solution)
    dual_obj = dot(solver_state.solution_avg.sum_dual_source_solution, problem.source_distribution) / solver_state.solution_avg.sum_weight + dot(solver_state.solution_avg.sum_dual_target_solution, problem.target_distribution) / solver_state.solution_avg.sum_weight

    primal_dual_gap = (primal_obj - dual_obj)^2

    return KKTResidual(
        sqrt(primal_residual),
        sqrt(dual_residual),
        sqrt(primal_dual_gap),
        sqrt(primal_residual + dual_residual + primal_dual_gap),
        sqrt(primal_residual) / (1 + ot_cache.norm_right_hand_side),
        sqrt(dual_residual) / (1 + ot_cache.norm_cost_matrix),
        sqrt(primal_dual_gap) / (1 + abs(primal_obj) + abs(dual_obj)),
    )
end