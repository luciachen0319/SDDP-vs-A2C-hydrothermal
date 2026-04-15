"""
A2C for Hydrothermal Scheduling — Julia implementation
Mirrors sddp_ts.jl:
  - Same 4 subsystems: SE, S, NE, N
  - Same PAR(1) inflow model (γ, Σ, exp_mu from data.jl)
  - Thermal: economic dispatch across 95 units (exact per-unit cost)
  - Exchange: learned approximation with anti-cycling penalty
  - Deficit: single-level per subsystem (simplified for training stability)
  - Spill: capped at 10% of storage per month
"""

using Flux
using Distributions
using LinearAlgebra
using Random
using Statistics
using Printf
using Plots

include("data.jl")
include("shared_scenarios.jl")

# ─────────────────────────────────────────────────────────────────
# 1. CONSTANTS AND DATA PREPARATION
# ─────────────────────────────────────────────────────────────────

const N_SYSTEMS = 4
const T_HORIZON = 24   # set to 120 for full long-term run

# PAR(1) inflow model parameters from data.jl
exp_mu = reduce(vcat, [EXP_MEANS[m]' for m in 1:12])   # 12×4
gamma_mat = reduce(vcat, [γ[m]' for m in 1:12])   # 12×4
sigma_mats = [Σ[m] for m in 1:12]                           # 12 × (4×4)

# Hydro bounds
stored_initial = storedEnergy_initial
inflow_initial = round.(INITIAL_INFLOWS, digits=0)
stored_ub = storedEnergy_ub
hydro_gen_ub = hydro_ub

# Thermal: per-unit data for economic dispatch
const N_UNITS = [length(thermal_ub[i]) for i in 1:4]   # [43,17,33,2]
const N_THERMAL_TOTAL = sum(N_UNITS)                             # 95

thermal_unit_ub = vcat([vec(thermal_ub[i]) for i in 1:4]...)  # 95-vector
thermal_unit_lb = vcat([vec(thermal_lb[i]) for i in 1:4]...)  # 95-vector
thermal_unit_cost = vcat([vec(thermal_obj[i]) for i in 1:4]...)  # 95-vector

thermal_idx = [sum(N_UNITS[1:i-1])+1:sum(N_UNITS[1:i]) for i in 1:4]
thermal_agg_ub = [sum(thermal_unit_ub[thermal_idx[i]]) for i in 1:4]
thermal_agg_lb = [sum(thermal_unit_lb[thermal_idx[i]]) for i in 1:4]
thermal_sort_idx = [thermal_idx[i][sortperm(thermal_unit_cost[thermal_idx[i]])]
                    for i in 1:4]

# Demand: 12×4
demand_mat = reduce(vcat, demand)

# Deficit (single level per subsystem — simplified for training stability)
deficit_obj_vec = deficit_obj   # [1142.8, 2465.4, 5152.46, 5845.54]
deficit_ub_frac = deficit_ub    # [0.05, 0.05, 0.1, 0.8] used as subsystem caps

# Exchange
exchange_ub_mat = reduce(vcat, exchange_ub)   # 5×5 upper bounds
exchange_cost_mat = [0.0 0.001 0.001 0.001 0.0005;   # from exchange_cost.csv
    0.001 0.0 0.001 0.001 0.0005;
    0.001 0.001 0.0 0.001 0.0005;
    0.001 0.001 0.001 0.0 0.0005;
    0.0005 0.0005 0.0005 0.0005 0.0]

const SPILL_COST = 0.001f0

# ─────────────────────────────────────────────────────────────────
# 2. INFLOW SAMPLER  (mirrors create_sampler in sddp_ts.jl)
# ─────────────────────────────────────────────────────────────────

function sample_inflow(t::Int, prev_inflow::Vector{Float64}, rng::AbstractRNG)
    month = mod1(t, 12)
    prev_month = mod1(t - 1, 12)

    # clamp raw noise to ±3 std devs before exponentiating
    # mirrors updated sddp_ts.jl create_sampler
    std_devs = sqrt.(diag(sigma_mats[month]))
    d = MvNormal(zeros(4), sigma_mats[month])
    raw_noise = rand(rng, d)
    clamped_noise = clamp.(raw_noise, -3 .* std_devs, 3 .* std_devs)
    noise = exp.(clamped_noise)

    coef = -noise .* gamma_mat[month, :] .*
           exp_mu[month, :] ./ (exp_mu[prev_month, :] .+ 1e-8)
    rhs = noise .* (1 .- gamma_mat[month, :]) .* exp_mu[month, :]

    # round to 4 digits — matches SDDP's support_points rounding
    coef = round.(coef, digits=4)
    rhs = round.(rhs, digits=4)

    return clamp.(rhs .- coef .* prev_inflow, 0.0, Inf)
end

# ─────────────────────────────────────────────────────────────────
# 3. ENVIRONMENT
# ─────────────────────────────────────────────────────────────────

mutable struct HydrothermalEnv
    t::Int
    stored::Vector{Float64}
    inflow::Vector{Float64}
    rng::AbstractRNG
    horizon::Int
end

function HydrothermalEnv(horizon=T_HORIZON; seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    HydrothermalEnv(0, copy(stored_initial), copy(inflow_initial), rng, horizon)
end

obs_dim() = 1 + N_SYSTEMS + N_SYSTEMS + N_SYSTEMS  # Month + Storage + Inflow + Demand = 13

# Action: hydro(4) + spill(4) + thermal(95) + deficit(16) + exchange(25) = 144
# matches SDDP decision variables exactly (section 5.1.3)
const N_DEFICIT = N_SYSTEMS * 4   # 4 subsystems × 4 depth levels = 16
act_dim() = N_SYSTEMS +           # hydro:    4
            N_SYSTEMS +           # spill:    4
            N_THERMAL_TOTAL +     # thermal: 95
            N_DEFICIT +           # deficit: 16
            25                    # exchange: 25

# 2. Update the get_obs function:
function get_obs(env::HydrothermalEnv)
    month = mod1(env.t + 1, 12)

    month_norm = Float32[month/12.0]
    stored_norm = Float32.(env.stored ./ stored_ub)
    inflow_norm = Float32.(env.inflow ./ 100000.0)

    # NEW: Explicit Demand Normalization (Divided by ~30k max regional demand)
    demand_norm = Float32.(demand_mat[month, :] ./ 30000.0)

    return vcat(month_norm, stored_norm, inflow_norm, demand_norm)
end

function reset!(env::HydrothermalEnv)
    env.t = 0
    env.stored = copy(stored_initial)
    env.inflow = copy(inflow_initial)
    return get_obs(env)
end

function env_step!(env::HydrothermalEnv, raw_action::Vector{Float32})
    month = mod1(env.t + 1, 12)
    demand_t = demand_mat[month, :]
    a = sigmoid.(raw_action)

    # ── Index layout (144 actions, mirrors SDDP section 5.1.3) ────
    # [1:4]     hydro q_it
    # [5:8]     spill s_it
    # [9:103]   thermal g_uit (95 units)
    # [104:119] deficit df_ijt (4×4=16)
    # [120:144] exchange ex_ijt (5×5=25)

    # ── Step 1: Hydro — clip to bounds ────────────────────────────
    avail_water = env.stored .+ env.inflow
    hydro_gen = Float64.(a[1:4]) .* min.(hydro_gen_ub, avail_water)

    # ── Step 2: Spill — enforce hydrological balance ───────────────
    # v_it = v_{i,t-1} + a_it - q_it - s_it
    # mandatory spill to prevent overflow + optional network spill
    spill_mandatory = max.(avail_water .- hydro_gen .- stored_ub, 0.0)
    spill_optional = Float64.(a[5:8]) .* env.stored .* 0.1
    spill = spill_mandatory .+ spill_optional
    # hydrological balance satisfied exactly:
    new_stored = clamp.(avail_water .- hydro_gen .- spill, 0.0, stored_ub)

    # ── Step 3: Exchange — clip to bounds, enforce transshipment ───
    exchange = reshape(Float64.(a[120:144]), 5, 5) .* exchange_ub_mat
    for i in 1:5
        exchange[i, i] = 0.0
    end

    # transshipment balance: sum(ex_i5) == sum(ex_5i)
    inflow_5 = sum(exchange[1:4, 5])
    outflow_5 = sum(exchange[5, 1:4])
    imbalance = inflow_5 - outflow_5
    if outflow_5 > 1e-8
        exchange[5, 1:4] .+= imbalance .* exchange[5, 1:4] ./ outflow_5
    end
    exchange = clamp.(exchange, 0.0, exchange_ub_mat)

    # ── Step 4: Thermal — clip each unit to [lb, ub] ──────────────
    thermal_raw = Float64.(a[9:103])
    thermal_units = thermal_unit_lb .+ thermal_raw .* (thermal_unit_ub .- thermal_unit_lb)
    thermal_gen = [sum(thermal_units[thermal_idx[i]]) for i in 1:4]

    # ── Step 5: Deficit — clip to [0, d_it × β_j] ─────────────────
    df_raw = Float64.(a[104:119])
    deficit_mat = zeros(4, 4)
    for i in 1:4, j in 1:4
        cap = demand_t[i] * deficit_ub_frac[j]
        deficit_mat[i, j] = df_raw[(i-1)*4+j] * cap
    end

    # ── Step 6: Power balance projection ──────────────────────────
    # sum(g_uit) + sum(df_ijt) + q_it - sum(ex_ijt) + sum(ex_jit) = d_it
    # scale thermal and deficit proportionally to satisfy this exactly
    for i in 1:4
        net_import = sum(exchange[:, i]) - sum(exchange[i, :])
        total_gen = hydro_gen[i] + thermal_gen[i] +
                    sum(deficit_mat[i, :]) + net_import
        gap = demand_t[i] - total_gen

        if abs(gap) > 1e-6
            flex = thermal_gen[i] + sum(deficit_mat[i, :])
            if flex > 1e-8
                scale = clamp((flex + gap) / flex, 0.0, 2.0)
                # scale thermal units
                for u in thermal_idx[i]
                    thermal_units[u] = clamp(thermal_units[u] * scale,
                        thermal_unit_lb[u],
                        thermal_unit_ub[u])
                end
                thermal_gen[i] = sum(thermal_units[thermal_idx[i]])
                # scale deficit
                for j in 1:4
                    deficit_mat[i, j] = clamp(deficit_mat[i, j] * scale,
                        0.0,
                        demand_t[i] * deficit_ub_frac[j])
                end
            end
        end
    end

    # ── Step 7: Cost (same as SDDP @stageobjective) ────────────────
    thermal_cost = dot(thermal_unit_cost, thermal_units)
    deficit_cost = sum(deficit_mat[i, j] * deficit_obj_vec[j]
                       for i in 1:4, j in 1:4)
    exchange_cost_val = sum(exchange .* exchange_cost_mat)
    spill_cost_val = 0.001 * sum(spill)
    total_cost = thermal_cost + deficit_cost + exchange_cost_val + spill_cost_val

    reward = Float32(-total_cost / 1e6)

    # ── Advance state ──────────────────────────────────────────────
    env.stored = new_stored
    env.t += 1
    env.inflow = sample_inflow(env.t, env.inflow, env.rng)
    done = env.t >= env.horizon

    return get_obs(env), reward, done, total_cost
end


# ─────────────────────────────────────────────────────────────────
# 4. ACTOR-CRITIC NETWORK
# ─────────────────────────────────────────────────────────────────

const OBS_DIM = obs_dim()
const ACT_DIM = act_dim()

# Custom initialization function for the final actor layer
small_init(out, in) = 0.01f0 .* randn(Float32, out, in)

struct A2CAgent
    actor::Chain
    critic::Chain
    log_std::Vector{Float32}
end
Flux.@functor A2CAgent (actor, critic, log_std)

function A2CAgent()
    actor = Chain(
        Dense(OBS_DIM => 256, tanh),
        Dense(256 => 256, tanh),
        Dense(256 => ACT_DIM, init=small_init)
    )

    critic = Chain(
        Dense(OBS_DIM => 256, tanh),
        Dense(256 => 256, tanh),
        Dense(256 => 1)
    )

    A2CAgent(actor, critic, zeros(Float32, ACT_DIM))
end

# ─────────────────────────────────────────────────────────────────
# 5. LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────

const VF_COEF = 0.5f0
const ENT_COEF = 0.01f0
const LOG_2PIE = Float32(log(2π * ℯ))

function a2c_loss(agent, obs_buf, act_buf, returns, adv)
    # 1. Convert Vectors of Vectors into Matrices (features × batch_size)
    obs_mat = reduce(hcat, obs_buf)
    act_mat = reduce(hcat, act_buf)

    # 2. CRITIC LOSS
    # Critic outputs a 1×batch_size matrix, we need a flat vector
    values = vec(agent.critic(obs_mat))
    critic_loss = mean((returns .- values) .^ 2)

    # 3. ACTOR LOSS
    μ = agent.actor(obs_mat)
    σ = exp.(agent.log_std)

    # Calculate log probability manually to ensure Flux/Zygote doesn't crash
    variance = σ .^ 2
    log_probs = -0.5f0 .* (((act_mat .- μ) .^ 2) ./ variance .+ 2.0f0 .* log.(σ) .+ log(2.0f0 * π))

    # Sum log probabilities across the action dimension for each step
    sum_log_probs = vec(sum(log_probs, dims=1))

    # Policy gradient loss
    actor_loss = -mean(sum_log_probs .* adv)

    # 4. ENTROPY BONUS (encourages exploration)
    # Entropy of a Normal distribution
    entropy = mean(sum(log.(σ) .+ 0.5f0 .+ 0.5f0 * log(2.0f0 * Float32(π))))

    # Total loss combining all three components
    total_loss = actor_loss + 0.5f0 * critic_loss - 0.005f0 * entropy

    return total_loss
end

# ─────────────────────────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

const GAMMA = 0.99f0
const LR = 3f-4
const N_STEPS = 12      # one full seasonal cycle per update
const N_EPISODES = 5_000

function train_a2c(; seed=42, n_episodes=N_EPISODES)
    rng = MersenneTwister(seed)
    env = HydrothermalEnv(T_HORIZON; seed=seed)
    agent = A2CAgent()
    opt = Flux.setup(Adam(LR), agent)

    ep_costs = Float64[]

    for ep in 1:n_episodes
        obs = reset!(env)
        ep_cost = 0.0
        done = false

        obs_buf = Vector{Float32}[]
        act_buf = Vector{Float32}[]
        rew_buf = Float32[]
        val_buf = Float32[]
        done_buf = Bool[]

        while !done
            for _ in 1:N_STEPS
                done && break
                μ = agent.actor(obs)
                σ = exp.(agent.log_std)
                a = μ .+ σ .* randn(rng, Float32, ACT_DIM)
                v = agent.critic(obs)[1]

                next_obs, rew, done, cost = env_step!(env, a)
                ep_cost += cost

                push!(obs_buf, obs)
                push!(act_buf, a)
                push!(rew_buf, rew)
                push!(val_buf, v)
                push!(done_buf, done)
                obs = next_obs
            end

            bootstrap = done ? 0.0f0 : agent.critic(obs)[1]
            n = length(rew_buf)
            returns = Vector{Float32}(undef, n)
            R = bootstrap
            for i in n:-1:1
                R = rew_buf[i] + GAMMA * R * (1f0 - Float32(done_buf[i]))
                returns[i] = R
            end

            adv = returns .- val_buf
            adv = (adv .- mean(adv)) ./ (std(adv) .+ 1f-8)

            _, grads = Flux.withgradient(agent) do ag
                a2c_loss(ag, obs_buf, act_buf, returns, adv)
            end
            Flux.update!(opt, agent, grads[1])

            empty!(obs_buf)
            empty!(act_buf)
            empty!(rew_buf)
            empty!(val_buf)
            empty!(done_buf)
        end

        push!(ep_costs, ep_cost)
        if ep % 100 == 0
            avg = mean(ep_costs[max(1, end - 99):end]) / 1e6
            @printf "Ep %4d | avg cost (last 100): %8.2f M R\$\n" ep avg
        end
    end

    return agent, ep_costs
end

# ─────────────────────────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────────────────────────

function evaluate(agent::A2CAgent; n_episodes=20, seed=100)
    costs = Float64[]
    for ep in 1:n_episodes
        env = HydrothermalEnv(T_HORIZON; seed=seed + ep)
        obs = reset!(env)
        ep_cost = 0.0
        done = false
        while !done
            μ = agent.actor(obs)
            _, _, done, cost = env_step!(env, μ)
            ep_cost += cost
        end
        push!(costs, ep_cost)
    end
    @printf "\nEvaluation (%d episodes, deterministic policy):\n" n_episodes
    @printf "  Mean total cost : %8.2f M R\$\n" mean(costs) / 1e6
    @printf "  Std             : %8.2f M R\$\n" std(costs) / 1e6
    return costs
end

function evaluate_detailed(agent; n_episodes=5, seed=100)
    for ep in 1:n_episodes
        env = HydrothermalEnv(T_HORIZON; seed=seed + ep)
        obs = reset!(env)
        done = false
        monthly_hydro = Float64[]
        monthly_thermal = Float64[]
        monthly_deficit = Float64[]
        monthly_spill = Float64[]
        monthly_exchange = Float64[]
        monthly_thermal_cost = Float64[]
        monthly_deficit_cost = Float64[]

        while !done
            μ = agent.actor(obs)
            a = sigmoid.(μ)
            month = mod1(env.t + 1, 12)
            demand_t = demand_mat[month, :]

            # same index layout as env_step!
            avail_water = env.stored .+ env.inflow
            hydro = Float64.(a[1:4]) .* min.(hydro_gen_ub, avail_water)

            spill_mandatory = max.(avail_water .- hydro .- stored_ub, 0.0)
            spill_optional = Float64.(a[5:8]) .* env.stored .* 0.1
            spill = spill_mandatory .+ spill_optional

            thermal_raw = Float64.(a[9:8+N_THERMAL_TOTAL])
            thermal_units = thermal_unit_lb .+ thermal_raw .* (thermal_unit_ub .- thermal_unit_lb)
            thermal = [sum(thermal_units[thermal_idx[i]]) for i in 1:4]
            t_cost = dot(thermal_unit_cost, thermal_units)

            df_raw = Float64.(a[104:119])
            deficit_mat = zeros(4, 4)
            for i in 1:4, j in 1:4
                cap = demand_t[i] * deficit_ub_frac[j]
                deficit_mat[i, j] = df_raw[(i-1)*4+j] * cap
            end
            d_cost = sum(deficit_mat[i, j] * deficit_obj_vec[j] for i in 1:4, j in 1:4)

            exchange = reshape(Float64.(a[120:144]), 5, 5) .* exchange_ub_mat
            for i in 1:5
                exchange[i, i] = 0.0
            end

            push!(monthly_hydro, sum(hydro))
            push!(monthly_thermal, sum(thermal))
            push!(monthly_deficit, sum(deficit_mat))
            push!(monthly_spill, sum(spill))
            push!(monthly_exchange, sum(exchange))
            push!(monthly_thermal_cost, t_cost)
            push!(monthly_deficit_cost, d_cost)

            obs, _, done, _ = env_step!(env, μ)
        end

        @printf "\nEpisode %d:\n" ep
        @printf "  Avg hydro:         %8.1f MW\n" mean(monthly_hydro)
        @printf "  Avg thermal:       %8.1f MW\n" mean(monthly_thermal)
        @printf "  Avg deficit:       %8.1f MW\n" mean(monthly_deficit)
        @printf "  Avg spill:         %8.1f MW\n" mean(monthly_spill)
        @printf "  Avg exchange:      %8.1f MW\n" mean(monthly_exchange)
        @printf "  Avg thermal cost:  %8.2f M R\$\n" mean(monthly_thermal_cost) / 1e6
        @printf "  Avg deficit cost:  %8.2f M R\$\n" mean(monthly_deficit_cost) / 1e6
    end
end

function evaluate_on_scenarios(agent::A2CAgent,
    scenarios_inflow::Vector{Vector{Vector{Float64}}})
    costs = Float64[]
    T = length(scenarios_inflow[1])
    for scenario in scenarios_inflow
        env = HydrothermalEnv(T)
        env.inflow = copy(scenario[1])
        obs = get_obs(env)
        ep_cost = 0.0
        done = false
        step = 1
        while !done
            μ = agent.actor(obs)
            _, _, done, cost = env_step!(env, μ)
            ep_cost += cost
            step += 1
            if !done && step <= T
                env.inflow = copy(scenario[step])
                obs = get_obs(env)
            end
        end
        push!(costs, ep_cost)
    end
    @printf "\nA2C Evaluation on shared scenarios (%d scenarios, T=%d):\n" length(scenarios_inflow) T
    @printf "  Mean total cost : %8.2f M R\$\n" mean(costs) / 1e6
    @printf "  Std             : %8.2f M R\$\n" std(costs) / 1e6
    return costs
end

# ─────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────

println("Training A2C for Hydrothermal Scheduling (T=$T_HORIZON months)...")
agent, ep_costs = train_a2c(seed=42)

# learning curve
smooth_n = 50
smooth = [mean(ep_costs[max(1, i - smooth_n + 1):i]) for i in eachindex(ep_costs)]
p = plot(ep_costs ./ 1e6, alpha=0.3, label="Episode cost",
    xlabel="Episode", ylabel="Total Cost (M R\$)",
    title="A2C Learning Curve — Hydrothermal Scheduling")
plot!(p, smooth ./ 1e6, lw=2, label="$(smooth_n)-ep moving avg")
savefig(p, "a2c_learning_curve.png")
println("Learning curve saved → a2c_learning_curve.png")

evaluate_detailed(agent)

const N_EVAL = 100
const EVAL_SEED = 200
scenarios_inflow, _ = generate_eval_scenarios(
    N_EVAL, T_HORIZON, gamma_mat, sigma_mats, exp_mu,
    inflow_initial; seed=EVAL_SEED
)
shared_eval_costs = evaluate_on_scenarios(agent, scenarios_inflow)