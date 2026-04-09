using SDDP
using HiGHS
using JuMP
using Distributions
using LinearAlgebra
using Plots
using Random
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "data.jl"))
include(joinpath(@__DIR__, "..", "shared_scenarios.jl"))

# ── Map data.jl variables to local names ──────────────────────────────────────

# PAR(1) parameters (same as A2C.jl)
const gamma_mat = reduce(vcat, [γ[m]' for m in 1:12])  # 12×4
const sigma_mats = [Σ[m] for m in 1:12]   # 12 × 4×4
const exp_mu_mat = reduce(vcat, [EXP_MEANS[m]' for m in 1:12])  # 12×4

# Initial conditions
const stored_initial_sddp = storedEnergy_initial                  # 4-vector
const inflow_initial_sddp = round.(INITIAL_INFLOWS, digits=0)    # 4-vector

# Bounds
const stored_ub_sddp = storedEnergy_ub   # 4-vector, MWmonth
const hydro_gen_ub_sddp = hydro_ub         # 4-vector, MW

# Thermal (per-subsystem vectors)
const th_lb = [vec(thermal_lb[i]) for i in 1:4]
const th_ub = [vec(thermal_ub[i]) for i in 1:4]
const th_obj = [vec(thermal_obj[i]) for i in 1:4]

# Demand: 12×4 matrix (month × subsystem)
const demand_sddp = reduce(vcat, demand)          # 12×4

# Deficit
const deficit_obj_sddp = deficit_obj   # [1142.8, 2465.4, 5152.46, 5845.54]
const deficit_ub_sddp = deficit_ub    # [0.05, 0.05, 0.1, 0.8]

# Exchange upper bounds: 5×5 matrix
const exchange_ub_sddp = reduce(vcat, exchange_ub)  # 5×5
# (exchange cost is zero in data.jl, so not included in objective)

# ── PAR(1) sampler (same formula as shared_scenarios.jl / A2C.jl) ─────────────

function create_sampler(t::Int)
    function sample_noise(rng::AbstractRNG)
        month = mod1(t, 12)
        prev_month = mod1(t - 1, 12)
        noise = exp.(rand(rng, MvNormal(zeros(4), sigma_mats[month])))
        coef = -noise .* gamma_mat[month, :] .*
               exp_mu_mat[month, :] ./ (exp_mu_mat[prev_month, :] .+ 1e-8)
        rhs = noise .* (1.0 .- gamma_mat[month, :]) .* exp_mu_mat[month, :]
        return vcat(clamp.(coef, -1e6, 1e6), round.(rhs, digits=4))
    end
    return sample_noise
end

# ── Model ─────────────────────────────────────────────────────────────────────

const T = 24

model = SDDP.LinearPolicyGraph(
    stages=T,
    sense=:Min,
    lower_bound=0.0,
    optimizer=HiGHS.Optimizer
) do sp, t

    # State variables
    @variables(sp, begin
        stored[i=1:4], (SDDP.State, initial_value=stored_initial_sddp[i],
            lower_bound=0, upper_bound=stored_ub_sddp[i])
        inflow[i=1:4], (SDDP.State, initial_value=inflow_initial_sddp[i],
            lower_bound=0)
    end)

    # Control variables
    @variable(sp, spill[i=1:4] >= 0)

    @variable(sp, hydro_gen[i=1:4] >= 0)
    for i in 1:4
        set_upper_bound(hydro_gen[i], hydro_gen_ub_sddp[i])
    end

    @variable(sp, deficit[i=1:4, j=1:4] >= 0)
    for i in 1:4, j in 1:4
        set_upper_bound(deficit[i, j],
            demand_sddp[t % 12 == 0 ? 12 : t % 12, i] * deficit_ub_sddp[j])
    end

    @variable(sp, thermal_gen[i=1:4, j=1:length(th_lb[i])] >= 0)
    for i in 1:4, j in 1:length(th_lb[i])
        set_lower_bound(thermal_gen[i, j], th_lb[i][j])
        set_upper_bound(thermal_gen[i, j], th_ub[i][j])
    end

    @variable(sp, exchange[i=1:5, j=1:5] >= 0)
    for i in 1:5, j in 1:5
        set_upper_bound(exchange[i, j], exchange_ub_sddp[i, j])
    end

    # Constraints
    for i in 1:4
        @constraint(sp, stored[i].out + spill[i] + hydro_gen[i] - stored[i].in == inflow[i].out)

        @constraint(sp,
            sum(thermal_gen[i, :]) + sum(deficit[i, :]) + hydro_gen[i] -
            sum(exchange[i, :]) + sum(exchange[:, i]) ==
            demand_sddp[t % 12 == 0 ? 12 : t % 12, i]
        )
    end

    @constraint(sp, sum(exchange[:, 5]) - sum(exchange[5, :]) == 0)

    # Stochastic inflow via PAR(1)
    if t == 1
        for i in 1:4
            @constraint(sp, inflow[i].in == inflow_initial_sddp[i])
            @constraint(sp, stored[i].in == stored_initial_sddp[i])
        end
    else
        @constraint(sp, base[i=1:4], inflow[i].out + inflow[i].in == 0)

        rng_build = Random.default_rng()
        n_samples = 100
        sampler = create_sampler(t)
        support_pts = [clamp.(round.(sampler(rng_build), digits=4), -1e6, 1e6)
                       for _ in 1:n_samples]
        probs = fill(1.0 / n_samples, n_samples)

        SDDP.parameterize(sp, support_pts, probs) do ω
            for i in 1:4
                set_normalized_coefficient(base[i], inflow[i].in, ω[i])
                set_normalized_rhs(base[i], ω[i+4])
            end
        end
    end

    # Objective (exchange cost = 0 in data.jl)
    @expression(sp, thermal_cost,
        sum(th_obj[i][j] * thermal_gen[i, j]
            for i in 1:4, j in 1:length(th_lb[i])))

    @expression(sp, deficit_cost,
        sum(deficit[i, j] * deficit_obj_sddp[j]
            for i in 1:4, j in 1:4))

    @expression(sp, spill_cost, 0.001 * sum(spill))

    @stageobjective(sp, thermal_cost + deficit_cost + spill_cost)
end

# ── Training ───────────────────────────────────────────────────────────────────

SDDP.train(model,
    iteration_limit=100,
    print_level=1,
    log_frequency=1,
    stopping_rules=[SDDP.BoundStalling(5, 1e-4)],
    risk_measure=SDDP.Expectation(),
)

# ── Evaluation on shared scenarios ────────────────────────────────────────────

const N_EVAL = 100    # number of evaluation scenarios
const EVAL_SEED = 100    # fixed seed — same as A2C evaluate()

scenarios_inflow, scenarios_omega = generate_eval_scenarios(
    N_EVAL, T, gamma_mat, sigma_mats, exp_mu_mat, inflow_initial_sddp;
    seed=EVAL_SEED
)

# SDDP.Historical expects: Vector of scenarios, each a Vector of stage=>noise pairs
# Stage 1 is deterministic; stages 2..T carry noise (omega_seq has T-1 entries)
historical = SDDP.Historical(
    [
        [(t + 1, scenarios_omega[s][t]) for t in 1:(T - 1)]
        for s in 1:N_EVAL
    ]
)

simulations = SDDP.simulate(model, N_EVAL,
    [:thermal_cost, :deficit_cost, :spill_cost];
    sampling_scheme = historical)

# Extract total cost per simulation
sim_costs = [
    sum(sim[t][:thermal_cost] + sim[t][:deficit_cost] + sim[t][:spill_cost]
        for t in 1:length(sim))
    for sim in simulations
]

@printf "\nSDDP Evaluation (%d scenarios, T=%d):\n" N_EVAL T
@printf "  Mean total cost : %8.2f M R\$\n" mean(sim_costs) / 1e6
@printf "  Std             : %8.2f M R\$\n" std(sim_costs) / 1e6