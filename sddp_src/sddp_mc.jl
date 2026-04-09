"""
This code solves the SDDP model for the hydro-thermal example using Markov discretized training data.
It uses stage T = 24, number of Markov states = 20, and SA as the dicretization method.
It also saves the simulation results into the `output/simulation/` folder.
"""

using SDDP, HiGHS
using DataFrames, CSV
using Distributions
using LinearAlgebra

include("save_results_sddp.jl")
include("read_data_sddp.jl")

const T = 24
#const scaling_factor = 1e-3


function analyze_coefficients(subproblem, t)
    # Analyze objective coefficients
    println("\nObjective coefficients:")
    
    # Deficit costs
    deficit_coeffs = [deficit_.OBJ[d] for d in 1:4]
    println("Deficit costs: ", extrema(deficit_coeffs), 
            " ratio: ", maximum(deficit_coeffs)/minimum(deficit_coeffs))
    
    # Thermal costs
    thermal_coeffs = vcat([thermal[p].OBJ for p in 1:4]...)
    println("Thermal costs: ", extrema(thermal_coeffs),
            " ratio: ", maximum(thermal_coeffs)/minimum(thermal_coeffs))
    
    # Exchange costs
    println("Exchange costs: ", extrema(exchange_cost),
            " ratio: ", maximum(exchange_cost)/minimum(exchange_cost))
    
    println("\nBounds:")
    # Storage bounds
    println("Storage bounds: ", extrema(hydro_.UB[1:4]))
    
    # Hydro generation bounds
    println("Hydro gen bounds: ", extrema(hydro_.UB[9:12]))
    
    # Thermal bounds
    thermal_bounds = vcat([thermal[p].UB for p in 1:4]...)
    println("Thermal bounds: ", extrema(thermal_bounds))
    
    # Deficit bounds (varies by time period)
    deficit_bounds = [demand[t%12+1,r] * deficit_.DEPTH[d] 
                     for r in 1:4, d in 1:4]
    println("Deficit bounds: ", extrema(deficit_bounds))
    
    # Exchange bounds
    println("Exchange bounds: ", extrema(exchange_ub))
    
    println("\nRHS values:")
    # Demand values
    println("Demand values: ", extrema(demand[t%12+1,:]))
    
    # Inflow values
    inflow_values = extrema(Markov_states[t])
    println("Inflow values: ", inflow_values)
end

model = SDDP.MarkovianPolicyGraph(
    transition_matrices = transition_matrix,
    sense = :Min,
    lower_bound = 0.0,
    optimizer = HiGHS.Optimizer
) do subproblem, node
    # Unpack the stage and Markov index
    t, markov_state = node
    
    # State variables
    @variable(subproblem, 0 <= stored[i=1:4] <= hydro_.UB[i], SDDP.State, 
                initial_value = stored_initial[i])
        
    # Control variables
    @variable(subproblem, 0 <= spill[i=1:4])
    @variable(subproblem, 0 <= hydro[i=1:4] <= hydro_.UB[i+8])
    @variable(subproblem, 0 <= deficit[r=1:4, d=1:4] <= demand[t%12 == 0 ? 12 : t%12, r] * deficit_.DEPTH[d])
    
    # Exchange variables - create with bounds separately
    @variable(subproblem, 0 <= exchange[i=1:5,j=1:5] <= exchange_ub[i,j])
    
    # Thermal variables
    @variable(subproblem, thermal_gen[p=1:4,u=1:size(thermal[p],1)], 
        lower_bound = thermal[p].LB[u],
        upper_bound = thermal[p].UB[u])
    
    # Random variables
    @variable(subproblem, inflow[1:4])

    # Get unique scenarios from current state's Markov matrix
    noise_terms = collect(eachrow(Markov_states[t]))

    # Parameterize the model
    SDDP.parameterize(subproblem, noise_terms) do ω
        for j in 1:4
            JuMP.fix(inflow[j], ω[j])
        end
    end

    # Objective function
    @stageobjective(subproblem,
        0.001 * sum(spill) +
        sum(deficit_.OBJ[d] * deficit[r,d] for r in 1:4, d in 1:4) +
        sum(exchange_cost[i,j] * exchange[i,j] for i in 1:5, j in 1:5) +
        sum(thermal[p].OBJ[u] * thermal_gen[p,u] for p in 1:4, u in 1:size(thermal[p],1))
    )

    # Constraints
    # Water balance
    @constraint(subproblem, [i=1:4], 
                stored[i].out == stored[i].in - hydro[i] - spill[i] + inflow[i])

    # Power balance
    @constraint(subproblem, [r=1:4],
        sum(thermal_gen[r,u] for u in 1:size(thermal[r],1)) +
        sum(deficit[r,d] for d in 1:4) +
        hydro[r] -
        sum(exchange[r,j] for j in 1:5) +
        sum(exchange[j,r] for j in 1:5) ==
        demand[t%12 == 0 ? 12 : t%12, r])

    # Exchange balance
    @constraint(subproblem,
        sum(exchange[j,5] for j in 1:5) -
        sum(exchange[5,j] for j in 1:5) == 0)

    # Initial conditions
    if t == 1
        @constraint(subproblem, [i=1:4], stored[i].in == stored_initial[i])
        analyze_coefficients(subproblem, t)
    end
end

SDDP.train(model, 
    iteration_limit=40,
    print_level=1,
    log_frequency=1,
    stopping_rules = [
        SDDP.BoundStalling(5, 1e-4)  # Stop if bound doesn't improve by 1e-4 in 5 iterations
    ],
    risk_measure = SDDP.Expectation(),
    #dashboard = true
)

# Simulate policy
simulations = SDDP.simulate(
    model, 
    100,  # number of replications
    [:stored, :spill, :hydro, :deficit, :exchange, :thermal_gen, :inflow]  # variables to record
)

# Save both individual and average results
save_simulation_results(simulations, "output/simulations/sddp_train/")
