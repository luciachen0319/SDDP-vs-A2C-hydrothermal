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
exchange_ub_mat = reduce(vcat, exchange_ub)   # 5×5

const SPILL_COST = 0.001f0

# ─────────────────────────────────────────────────────────────────
# 2. INFLOW SAMPLER  (mirrors create_sampler in sddp_ts.jl)
# ─────────────────────────────────────────────────────────────────

function sample_inflow(t::Int, prev_inflow::Vector{Float64}, rng::AbstractRNG)
    month = mod1(t, 12)
    prev_month = mod1(t - 1, 12)
    d = MvNormal(zeros(4), sigma_mats[month])
    noise = exp.(rand(rng, d))
    coef = -noise .* gamma_mat[month, :] .*
           exp_mu[month, :] ./ (exp_mu[prev_month, :] .+ 1e-8)
    rhs = noise .* (1 .- gamma_mat[month, :]) .* exp_mu[month, :]
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

obs_dim() = 2 * N_SYSTEMS + N_SYSTEMS   # stored(4) + inflow(4) + surplus(4) = 12
act_dim() = 3 * N_SYSTEMS + 5 * 5      # hydro(4) + thermal(4) + spill(4) + exchange(25) = 37

function get_obs(env::HydrothermalEnv)
    month = mod1(env.t + 1, 12)
    demand_t = demand_mat[month, :]

    stored_norm = env.stored ./ stored_ub
    inflow_norm = env.inflow ./ (exp_mu[month, :] .+ 1e-8)

    # surplus: positive = can export, negative = needs imports
    avail_hydro = min.(hydro_gen_ub, env.stored .+ env.inflow)
    surplus_norm = (avail_hydro .+ thermal_agg_ub .- demand_t) ./ (demand_t .+ 1e-8)

    Float32.(vcat(stored_norm, inflow_norm, surplus_norm))
end

function reset!(env::HydrothermalEnv)
    env.t = 0
    env.stored = copy(stored_initial)
    env.inflow = copy(inflow_initial)
    return get_obs(env)
end

# Economic dispatch: cheapest units first — same result as SDDP LP for given total
function economic_dispatch(i::Int, total_mw::Float64)
    idx = thermal_sort_idx[i]
    lbs = thermal_unit_lb[idx]
    ubs = thermal_unit_ub[idx]
    dispatch = copy(lbs)
    remaining = total_mw - sum(lbs)

    for j in eachindex(idx)
        remaining <= 0 && break
        extra = min(remaining, ubs[j] - lbs[j])
        dispatch[j] += extra
        remaining -= extra
    end

    result = zeros(length(thermal_idx[i]))
    result[sortperm(thermal_sort_idx[i] .- (thermal_idx[i].start - 1))] = dispatch
    return result
end

function env_step!(env::HydrothermalEnv, raw_action::Vector{Float32})
    month = mod1(env.t + 1, 12)
    demand_t = demand_mat[month, :]
    a = sigmoid.(raw_action)

    # ── Hydro ─────────────────────────────────────────────────────
    avail_water = env.stored .+ env.inflow
    hydro_gen = Float64.(a[1:4]) .* min.(hydro_gen_ub, avail_water)

    # ── Thermal: economic dispatch (exact per-unit cost) ───────────
    thermal_total = thermal_agg_lb .+
                    Float64.(a[5:8]) .* (thermal_agg_ub .- thermal_agg_lb)
    thermal_units = vcat([economic_dispatch(i, thermal_total[i]) for i in 1:4]...)
    thermal_gen = [sum(thermal_units[thermal_idx[i]]) for i in 1:4]
    thermal_cost = dot(thermal_unit_cost, thermal_units)

    # ── Spill (capped at 10% of storage) ──────────────────────────
    spill = Float64.(a[9:12]) .* env.stored .* 0.1

    # ── Exchange ───────────────────────────────────────────────────
    exchange = reshape(Float64.(a[13:37]), 5, 5) .* exchange_ub_mat

    for i in 1:5
        exchange[i, i] = 0.0
    end   # no self-exchange

    # transshipment balance at node 5
    inflow_5 = sum(exchange[1:4, 5])
    outflow_5 = sum(exchange[5, 1:4])
    imbalance = inflow_5 - outflow_5
    if outflow_5 > 1e-8
        exchange[5, 1:4] .+= imbalance .* exchange[5, 1:4] ./ outflow_5
    end
    exchange = clamp.(exchange, 0.0, exchange_ub_mat)

    # export cap: cannot export more than you generate
    for i in 1:4
        max_exportable = hydro_gen[i] + thermal_gen[i]
        total_export = sum(exchange[i, :])
        if total_export > max_exportable
            exchange[i, :] .*= max_exportable / (total_export + 1e-8)
        end
    end

    # ── Hydrological mass balance ──────────────────────────────────
    new_stored = clamp.(env.stored .+ env.inflow .- hydro_gen .- spill,
        0.0, stored_ub)

    # ── Energy balance: deficit (single-level per subsystem) ───────
    deficit = zeros(4)
    for i in 1:4
        net_export = sum(exchange[i, :]) - sum(exchange[:, i])
        shortfall = demand_t[i] - hydro_gen[i] - thermal_gen[i] + net_export
        deficit[i] = clamp(shortfall, 0.0, demand_t[i] * deficit_ub_frac[i])
    end

    # ── Cost ───────────────────────────────────────────────────────
    deficit_cost = dot(deficit_obj_vec, deficit)
    spill_cost_val = SPILL_COST * sum(spill)
    exchange_vol_penalty = 0.0001 * sum(exchange)   # discourages cycling
    total_cost = thermal_cost + deficit_cost + spill_cost_val + exchange_vol_penalty

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

struct A2CAgent
    actor::Chain
    critic::Chain
    log_std::Vector{Float32}
end
Flux.@functor A2CAgent (actor, critic, log_std)

function A2CAgent()
    actor = Chain(Dense(OBS_DIM => 64, tanh),
        Dense(64 => 64, tanh),
        Dense(64 => ACT_DIM))
    critic = Chain(Dense(OBS_DIM => 64, tanh),
        Dense(64 => 64, tanh),
        Dense(64 => 1))
    A2CAgent(actor, critic, zeros(Float32, ACT_DIM))
end

# ─────────────────────────────────────────────────────────────────
# 5. LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────

const VF_COEF = 0.5f0
const ENT_COEF = 0.01f0
const LOG_2PIE = Float32(log(2π * ℯ))

function a2c_loss(agent::A2CAgent,
    obs_buf::Vector{Vector{Float32}},
    act_buf::Vector{Vector{Float32}},
    returns::Vector{Float32},
    advantages::Vector{Float32})
    n = length(obs_buf)
    lnσ = agent.log_std
    σ = exp.(lnσ)
    actor_l = 0.0f0
    critic_l = 0.0f0
    entropy = 0.0f0

    for i in 1:n
        μ = agent.actor(obs_buf[i])
        v = agent.critic(obs_buf[i])[1]
        ε = (act_buf[i] .- μ) ./ σ
        lp = sum(-0.5f0 .* ε .^ 2 .- lnσ .- 0.5f0 * log(2f0 * Float32(π)))

        actor_l -= lp * advantages[i]
        critic_l += (returns[i] - v)^2
        entropy += sum(0.5f0 .* (LOG_2PIE .+ 2f0 .* lnσ))
    end

    return (actor_l + VF_COEF * critic_l - ENT_COEF * entropy) / n
end

# ─────────────────────────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

const GAMMA = 0.99f0
const LR = 3f-4
const N_STEPS = 12       # one full seasonal cycle per update
const N_EPISODES = 3_000

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

        while !done
            μ = agent.actor(obs)
            a = sigmoid.(μ)
            month = mod1(env.t + 1, 12)
            demand_t = demand_mat[month, :]
            avail = env.stored .+ env.inflow

            hydro = Float64.(a[1:4]) .* min.(hydro_gen_ub, avail)
            thermal_total = thermal_agg_lb .+ Float64.(a[5:8]) .* (thermal_agg_ub .- thermal_agg_lb)
            thermal_units = vcat([economic_dispatch(i, thermal_total[i]) for i in 1:4]...)
            thermal = [sum(thermal_units[thermal_idx[i]]) for i in 1:4]
            spill = Float64.(a[9:12]) .* env.stored .* 0.1
            exch = reshape(Float64.(a[13:37]), 5, 5) .* exchange_ub_mat
            deficit = clamp.(demand_t .- hydro .- thermal,
                0.0, demand_t .* deficit_ub_frac)

            push!(monthly_hydro, sum(hydro))
            push!(monthly_thermal, sum(thermal))
            push!(monthly_deficit, sum(deficit))
            push!(monthly_spill, sum(spill))
            push!(monthly_exchange, sum(exch))

            obs, _, done, _ = env_step!(env, μ)
        end

        @printf "\nEpisode %d:\n" ep
        @printf "  Avg hydro:    %8.1f MW\n" mean(monthly_hydro)
        @printf "  Avg thermal:  %8.1f MW\n" mean(monthly_thermal)
        @printf "  Avg deficit:  %8.1f MW\n" mean(monthly_deficit)
        @printf "  Avg spill:    %8.1f MW\n" mean(monthly_spill)
        @printf "  Avg exchange: %8.1f MW\n" mean(monthly_exchange)
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
const EVAL_SEED = 100
scenarios_inflow, _ = generate_eval_scenarios(
    N_EVAL, T_HORIZON, gamma_mat, sigma_mats, exp_mu,
    inflow_initial; seed=EVAL_SEED
)
shared_eval_costs = evaluate_on_scenarios(agent, scenarios_inflow)