 

function projection_primal!(primal_iterate::Matrix{Float64})
    @. primal_iterate = max(0, primal_iterate)
end

function take_step!(
    step_params::ConstantStepsizeParams,
    problem::OptimalTransportProblem,
    solver_state::PrimalDualSolverState,
)
    solver_state.old_primal_col_sum = copy(solver_state.current_primal_col_sum)
    solver_state.old_primal_row_sum = copy(solver_state.current_primal_row_sum)

    # PDHG
    solver_state.current_primal_solution .-= solver_state.step_size_primal .* solver_state.primal_gradient
    projection_primal!(solver_state.current_primal_solution)

    solver_state.current_primal_row_sum .= vec(sum(solver_state.current_primal_solution, dims=2))
    solver_state.current_primal_col_sum .= vec(sum(solver_state.current_primal_solution, dims=1))

    solver_state.dual_source_solution .+= solver_state.step_size_dual.*(problem.source_distribution - 2 * solver_state.current_primal_row_sum + solver_state.old_primal_row_sum)
    solver_state.dual_target_solution .+= solver_state.step_size_dual.*(problem.target_distribution - 2 * solver_state.current_primal_col_sum + solver_state.old_primal_col_sum)

    solver_state.primal_gradient = problem.cost_matrix .- solver_state.dual_target_solution' .- solver_state.dual_source_solution

    # update average
    solver_state.solution_avg.sum_primal_solution .+= solver_state.current_primal_solution
    solver_state.solution_avg.sum_primal_row_sum .+= solver_state.current_primal_row_sum
    solver_state.solution_avg.sum_primal_col_sum .+= solver_state.current_primal_col_sum
    
    solver_state.solution_avg.sum_dual_source_solution .+= solver_state.dual_source_solution
    solver_state.solution_avg.sum_dual_target_solution .+= solver_state.dual_target_solution
    solver_state.solution_avg.sum_primal_gradient .+= solver_state.primal_gradient
    solver_state.solution_avg.sum_weight += 1.0
end

function optimize(
    problem::OptimalTransportProblem,
    params::PrimalDualOptimizerParameters,
)

    m, n = size(problem.cost_matrix)

    ot_cache = OTCache(
        norm(problem.cost_matrix),
        norm([problem.source_distribution;problem.target_distribution]),
    )

    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    println("(m, n) = ($(m), $(n)), time_limit = $(params.time_limit) seconds, kkt_tolerance = $(params.kkt_tolerance)")
    println()


    if params.step_size_policy_params isa ConstantStepsizeParams
        step_size_primal, step_size_dual = 1/sqrt(m+n), 1/sqrt(m+n)
    end

    solver_state = PrimalDualSolverState(
        zeros(m,n),
        zeros(m),
        zeros(n),
        zeros(m),
        zeros(n),
        zeros(m),
        zeros(n),
        problem.cost_matrix,
        initialize_solution_avg(n,m),
        step_size_primal,
        step_size_dual,
    )

    buffer_feas = BufferFeas(
        zeros(m,n),
        zeros(m),
        zeros(n),
    )

    # list of all kkt object throughout runtime 
    kkt_stats = KKTResidual[]
    # list of all kkt values throughout runtime 
    kkt_stats_res = Float64[]
    # list of all kkt values every 40 steps 
    kkt_every_40_steps = Float64[]
    # kkt from previous evaluation point 
    last_kkt = KKTResidual(
        Inf,
        Inf,
        Inf,
        Inf,
        Inf,
        Inf,
        Inf,
    )
    
    kkt_stats_feas = KKTResidual[]
    kkt_stats_feas_res = Float64[]
    
    evaluation_frequency = 64
    restart_threshold = 0.5
    restart_artificial = 0.25
    print_frequency = 1024
    check_feas = false 

    start_time = time()
    time_basic = 0.0

    iter = 0
    iter_epoch = 0

    converged = false 



    while true

        iter += 1
        iter_epoch += 1

        
        
        if mod(iter, evaluation_frequency) == 0
            curr_kkt = compute_kkt_residual(problem, solver_state, ot_cache)
            push!(kkt_stats, curr_kkt)
            push!(kkt_stats_res, curr_kkt.kkt_residual)
           

            if check_feas
                rounding!(problem, solver_state, buffer_feas)
                feas_kkt = compute_kkt_residual(problem, solver_state, buffer_feas, ot_cache)
                push!(kkt_stats_feas, feas_kkt)
                push!(kkt_stats_feas_res, feas_kkt.kkt_residual)
                
                if max(feas_kkt.primal_residual, feas_kkt.dual_residual, feas_kkt.primal_dual_gap) < params.kkt_tolerance # || max(feas_kkt.rel_primal_residual, feas_kkt.rel_dual_residual, feas_kkt.rel_primal_dual_gap) < params.kkt_tolerance
                    println("Optimal solution is found after $(iter) iterations.")
                    converged = true
                    break
                end
            else
                if max(curr_kkt.primal_residual, curr_kkt.dual_residual, curr_kkt.primal_dual_gap) < params.kkt_tolerance # || max(curr_kkt.rel_primal_residual, curr_kkt.rel_dual_residual, curr_kkt.rel_primal_dual_gap) < params.kkt_tolerance
                    println("Optimal solution is found after $(iter) iterations.")
                    converged = true
                    break
                end
            end

            if curr_kkt.kkt_residual <= restart_threshold * last_kkt.kkt_residual# || iter_epoch > restart_artificial * iter
                solver_state.current_primal_solution .= solver_state.solution_avg.sum_primal_solution ./ solver_state.solution_avg.sum_weight
                solver_state.current_primal_row_sum .= solver_state.solution_avg.sum_primal_row_sum ./ solver_state.solution_avg.sum_weight
                solver_state.current_primal_col_sum .= solver_state.solution_avg.sum_primal_col_sum ./ solver_state.solution_avg.sum_weight
                solver_state.old_primal_row_sum .= 0.0
                solver_state.old_primal_col_sum .= 0.0
                solver_state.dual_source_solution .= solver_state.solution_avg.sum_dual_source_solution ./ solver_state.solution_avg.sum_weight
                solver_state.dual_target_solution .= solver_state.solution_avg.sum_dual_target_solution ./ solver_state.solution_avg.sum_weight
                solver_state.primal_gradient .= solver_state.solution_avg.sum_primal_gradient ./ solver_state.solution_avg.sum_weight
                reset_solution_avg!(solver_state)

                println("Restart!")

                last_kkt = copy_kkt(curr_kkt)
                iter_epoch = 0
            end
        end

        if mod(iter, print_frequency) == 0 
            if check_feas
                println("iter = $(iter), kkt = $(kkt_stats_feas_res[end])")
            else
                println("iter = $(iter), kkt = $(kkt_stats_res[end])")
            end
        end

        if time() - start_time > params.time_limit
            println("Time limit reached after $(iter) iterations. Stopping optimization.")
            break
        end

        time_basic_checkpoint = time()
        take_step!(params.step_size_policy_params, problem, solver_state)
        time_basic += time() - time_basic_checkpoint
    end

    time_full = time() - start_time

    println("#####################################")
    if check_feas
        println("primal residual: $(kkt_stats_feas[end].primal_residual)")
        println("dual residual: $(kkt_stats_feas[end].dual_residual)")
        println("primal-dual gap: $(kkt_stats_feas[end].primal_dual_gap)")
        # println("relative primal residual: $(kkt_stats_feas[end].rel_primal_residual)")
        # println("relative dual residual: $(kkt_stats_feas[end].rel_dual_residual)")
        # println("relative primal-dual gap: $(kkt_stats_feas[end].rel_primal_dual_gap)")
        kkt_output = kkt_stats_feas_res
    else
        println("primal residual: $(kkt_stats[end].primal_residual)")
        println("dual residual: $(kkt_stats[end].dual_residual)")
        println("primal-dual gap: $(kkt_stats[end].primal_dual_gap)")
        # println("relative primal residual: $(kkt_stats[end].rel_primal_residual)")
        # println("relative dual residual: $(kkt_stats[end].rel_dual_residual)")
        # println("relative primal-dual gap: $(kkt_stats[end].rel_primal_dual_gap)")
        kkt_output = kkt_stats_res
    end

    # println("relative primal residual: $(kkt_stats[end].rel_primal_residual)")
    # println("relative dual residual: $(kkt_stats[end].rel_dual_residual)")
    # println("relative primal-dual gap: $(kkt_stats[end].rel_primal_dual_gap)")
    
    println()

    println("iter: $(iter)")
    println("time_basic: $(time_basic)")
    println("time_full: $(time_full)")
    println("#####################################")



    
    if check_feas
        return kkt_output, iter, time_basic, time_full, converged, solver_state.current_primal_solution, kkt_stats_feas[end].primal_residual, kkt_stats_feas[end].dual_residual, kkt_stats_feas[end].primal_dual_gap
    else
        return kkt_output, iter, time_basic, time_full, converged, solver_state.current_primal_solution, kkt_stats[end].primal_residual, kkt_stats[end].dual_residual, kkt_stats[end].primal_dual_gap
    end
end






















