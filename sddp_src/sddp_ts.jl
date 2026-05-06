using SDDP
using HiGHS
using JuMP
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Plots
using Random
using Printf
using Statistics


include("save_results_sddp.jl")


# Include the exact same scenario generator you use in A2C.jl
# (Assuming shared_scenarios.jl has the `generate_eval_scenarios` function)
include("../shared_scenarios.jl")

const N_EVAL = 100
const EVAL_SEED = 204 # Must match the seed in A2C.jl


gamma = Matrix(CSV.read("../data/gamma.csv", DataFrame))[:, 2:end]

# Read sigma matrices
sigma = Vector{Matrix{Float64}}(undef, 12)
for i in 0:11
    df = CSV.read("../data/sigma_$i.csv", DataFrame)
    sigma[i+1] = Matrix(df[:, 2:end])  # Remove first column if it's index
end

exp_mu = Matrix(CSV.read("../data/exp_mu.csv", DataFrame))[:, 2:end]

hydro_ = CSV.read("../data/hydro.csv", DataFrame)
demand = Matrix(CSV.read("../data/demand.csv", DataFrame))[:, 2:end]
deficit_ = CSV.read("../data/deficit.csv", DataFrame)
exchange_ub = Matrix(CSV.read("../data/exchange.csv", DataFrame))[:, 2:end]
exchange_cost = Matrix(CSV.read("../data/exchange_cost.csv", DataFrame))[:, 2:end]

# Read thermal data
thermal = Vector{DataFrame}(undef, 4)
for i in 0:3
    thermal[i+1] = CSV.read("../data/thermal_$i.csv", DataFrame)[:, 2:end]
end

stored_initial = hydro_.INITIAL[1:4]
inflow_initial = round.(hydro_.INITIAL[5:8], digits=0)

# Create noise sampler
# TODO
function create_sampler(t, sigma, gamma, exp_mu)
    # Pre-calculate the standard deviations for the current month
    # sigma is a covariance matrix, so the standard deviations are the sqrt of the diagonal
    std_devs = sqrt.(diag(sigma[mod1(t, 12)]))

    function sample_noise(rng::AbstractRNG)
        # 1. Define the multivariate normal distribution (mean = 0)
        d = MvNormal(zeros(4), sigma[mod1(t, 12)])

        # 2. Draw the raw statistical noise
        raw_noise = rand(rng, d)

        # 3. CLAMP THE RAW NOISE: bound it strictly to ± 3 standard deviations
        clamped_noise = clamp.(raw_noise, -3 .* std_devs, 3 .* std_devs)

        # 4. Exponentiate the safely bounded noise
        noise = exp.(clamped_noise)

        # 5. Calculate the PAR(1) affine terms
        coef = -noise .* gamma[mod1(t, 12), :] .* exp_mu[mod1(t, 12), :] ./ exp_mu[mod1(t - 1, 12), :]

        rhs = noise .* (1 .- gamma[mod1(t, 12), :]) .* exp_mu[mod1(t, 12), :]

        return vcat(coef, rhs)
    end

    return sample_noise
end

T = 24

# for bounding the inflow variables to avoid infinite space
# this is set to be at least 10x the initial inflow values
MAX_INFLOW = [1000000.0, 1000000.0, 1000000.0, 1000000.0]
# TODO

model = SDDP.LinearPolicyGraph(
    stages=T,
    sense=:Min,
    lower_bound=0.0,
    optimizer=HiGHS.Optimizer
) do sp, t

    # State variables
    @variables(sp, begin
        stored[i=1:4], (SDDP.State, initial_value=stored_initial[i],
            lower_bound=0, upper_bound=hydro_[i, "UB"])
        inflow[i=1:4], (SDDP.State, initial_value=inflow_initial[i],
            lower_bound=0, upper_bound=MAX_INFLOW[i])
    end)

    # Control variables  
    @variable(sp, spill[i=1:4] >= 0)

    # Hydro generation with indexed bounds
    @variable(sp, hydro_gen[i=1:4] >= 0)
    for i in 1:4
        set_upper_bound(hydro_gen[i], hydro_[i+8, "UB"])
    end

    # Deficit with indexed bounds
    @variable(sp, deficit[i=1:4, j=1:4] >= 0)
    for i in 1:4, j in 1:4
        set_upper_bound(deficit[i, j],
            demand[t % 12 == 0 ? 12 : t % 12, i] * deficit_.DEPTH[j])
    end

    # Thermal generation with indexed bounds
    @variable(sp, thermal_gen[i=1:4, j=1:length(thermal[i].LB)] >= 0)
    for i in 1:4, j in 1:length(thermal[i].LB)
        set_lower_bound(thermal_gen[i, j], thermal[i].LB[j])
        set_upper_bound(thermal_gen[i, j], thermal[i].UB[j])
    end

    # Exchange with indexed bounds
    @variable(sp, exchange[i=1:5, j=1:5] >= 0)
    for i in 1:5, j in 1:5
        set_upper_bound(exchange[i, j], exchange_ub[i, j])
    end

    # Constraints
    for i in 1:4
        # Hydro balance
        @constraint(sp, stored[i].out + spill[i] + hydro_gen[i] - stored[i].in == inflow[i].out)

        # Power balance
        @constraint(sp,
            sum(thermal_gen[i, :]) + sum(deficit[i, :]) + hydro_gen[i] -
            sum(exchange[i, :]) + sum(exchange[:, i]) ==
            demand[t % 12 == 0 ? 12 : t % 12, i]
        )
    end

    # Exchange balance
    @constraint(sp,
        sum(exchange[:, 5]) - sum(exchange[5, :]) == 0
    )

    # Markov chain for inflows
    if t == 1  # First stage — single deterministic scenario so Historical sampler covers it
        @constraint(sp, base_t1[i=1:4], inflow[i].out == 0.0)
        SDDP.parameterize(sp, [inflow_initial], [1.0]) do ω
            for i in 1:4
                set_normalized_rhs(base_t1[i], ω[i])
            end
        end
    else  # Other stages
        # Create base constraints
        @constraint(sp, base[i=1:4], inflow[i].out + inflow[i].in == 0)

        # Generate support points and probabilities
        rng = Random.default_rng()
        n_samples = 100
        support_points = Vector{Vector{Float64}}(undef, n_samples)
        probabilities = fill(1.0 / n_samples, n_samples)

        sampler = create_sampler(t, sigma, gamma, exp_mu)
        for i in 1:n_samples
            support_points[i] = sampler(rng)
        end

        support_points = [round.(point, digits=4) for point in support_points]

        # Apply uncertainty
        SDDP.parameterize(sp, support_points, probabilities) do ω
            for i in 1:4
                set_normalized_coefficient(base[i], inflow[i].in, ω[i])
                set_normalized_rhs(base[i], ω[i+4])
            end
        end
    end

    # Objective
    @expression(sp, thermal_cost,
        sum(thermal[i].OBJ[j] * thermal_gen[i, j]
            for i = 1:4, j = 1:length(thermal[i].LB)))

    @expression(sp, deficit_cost,
        sum(deficit[i, j] * deficit_[j, "OBJ"]
            for i = 1:4, j = 1:4))

    @expression(sp, exchange_cost_sum,
        sum(exchange[i, j] * exchange_cost[i, j]
            for i = 1:5, j = 1:5))

    @expression(sp, spill_cost,
        0.001 * sum(spill))

    @stageobjective(sp, thermal_cost + deficit_cost + exchange_cost_sum + spill_cost)

end

sddp_train_time = @elapsed SDDP.train(model,
    iteration_limit=1000,
    print_level=1,
    log_frequency=1,
    stopping_rules=[
        SDDP.BoundStalling(20, 1e-4)  # Stop if bound doesn't improve by 1e-4 in 20 iterations
    ],
    risk_measure=SDDP.Expectation(),
    #dashboard = true
)

# Reformat parameters for the scenario generator to match A2C.jl
# Ensure they are exactly 12 rows (months) x 4 columns (regions)
gamma_mat = size(gamma, 1) == 12 ? gamma : Matrix(gamma')
exp_mu_mat = size(exp_mu, 1) == 12 ? exp_mu : Matrix(exp_mu')
sigma_mats = sigma

# Generate the identical evaluation paths
scenarios_inflow, scenarios_omega = generate_eval_scenarios(
    N_EVAL, 24, gamma_mat, sigma_mats, exp_mu_mat, inflow_initial;
    seed=EVAL_SEED
)

# Map the scenarios into SDDP's "Historical" format
# (Stage 1 is deterministic, so noise starts at t=2)
historical_sampler = SDDP.Historical(
    [
    vcat(
        [(1, inflow_initial)],
        [(t + 1, scenarios_omega[s][t]) for t in 1:23]
    )
    for s in 1:N_EVAL
]
)

# Run the simulation using the synchronized paths
simulations = SDDP.simulate(
    model,
    N_EVAL,
    [
        :thermal_cost, :deficit_cost, :spill_cost,
        :stored, :spill, :hydro_gen, :deficit, :exchange, :thermal_gen, :inflow
    ];
    sampling_scheme=historical_sampler
)

costs = [sum(stage[:stage_objective] for stage in sim) for sim in simulations]
true_mean_cost = sum(costs) / length(costs)
std_cost = std(costs)
println("Std Dev of Simulation Cost: ", std_cost)
println()
println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
@printf "SDDP training time: %.2f seconds\n" sddp_train_time
@printf "SDDP training time: %.2f minutes\n" sddp_train_time / 60
println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
println("Final Lower Bound: ", SDDP.calculate_bound(model))
println("True Expected Simulation Cost: ", true_mean_cost)

save_results_sddp(simulations, "../output/simulations/sddp_train/ts/"; thermal=thermal, save_individual=true)