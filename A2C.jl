"""
A2C for Hydrothermal Scheduling — Julia implementation
Mirrors sddp_ts.jl exactly:
  - Same 4 subsystems: SE, S, NE, N
  - Same PAR(1) inflow model 
  - Same cost structure: thermal + deficit + exchange + spill
  - Same data loading pattern
"""

using Flux
using Distributions
using LinearAlgebra
using Random
using Statistics
using Printf
using Plots

include("data.jl")  # Import the data

# ─────────────────────────────────────────────────────────────────
# 1. CONSTANTS AND DATA PREPARATION
# ─────────────────────────────────────────────────────────────────

const N_SYSTEMS = 4
const T_HORIZON = 120  # set to 120 for full long-term run

# exp_mu: 12×4 matrix of expected inflows per month per subsystem
exp_mu = reduce(vcat, [EXP_MEANS[m]' for m in 1:12])   # 12×4

# gamma_mat: PAR(1) autoregressive coefficients (12×4), from γ in data.jl
gamma_mat = reduce(vcat, [γ[m]' for m in 1:12])   # 12×4

# sigma_mats: PAR(1) noise covariance matrices (one 4×4 per month), from Σ in data.jl
sigma_mats = [Σ[m] for m in 1:12]

# Hydro bounds
stored_initial = storedEnergy_initial          # 4-vector, MWmonth
inflow_initial = round.(INITIAL_INFLOWS, digits=0)
stored_ub = storedEnergy_ub               # 4-vector
hydro_gen_ub = hydro_ub                      # 4-vector

# Thermal
N_UNITS = [43, 17, 33, 2]     # units per subsystem
N_THERMAL_TOTAL = 95           # total units
thermal_unit_ub/lb/cost        # 95-element flat vectors
thermal_idx                    # index ranges per subsystem

# Demand: 12×4 matrix (month × subsystem)
demand_mat = reduce(vcat, demand)               # 12×4

# Deficit
deficit_obj_vec = deficit_obj                   # [1142.8, 2465.4, 5152.46, 5845.54]
deficit_ub_frac = deficit_ub                    # [0.05, 0.05, 0.1, 0.8]

# Exchange upper bounds: 5×5 matrix
exchange_ub_mat = reduce(vcat, exchange_ub)     # 5×5

const SPILL_COST = 0.001f0

# ─────────────────────────────────────────────────────────────────
# 2. INFLOW SAMPLER  (mirrors create_sampler in sddp_ts.jl)
#    PAR(1) log-normal
# ─────────────────────────────────────────────────────────────────

function sample_inflow(t::Int, prev_inflow::Vector{Float64}, rng::AbstractRNG)
    # The formula: log(inflow[t]) = γ[month] × log(inflow[t-1]) + noise[t]

    month = mod1(t, 12)
    prev_month = mod1(t - 1, 12)

    d = MvNormal(zeros(4), sigma_mats[month])
    noise = exp.(rand(rng, d))

    coef = -noise .* gamma_mat[month, :] .*
           exp_mu[month, :] ./ (exp_mu[prev_month, :] .+ 1e-8) # Linearization 
    rhs = noise .* (1 .- gamma_mat[month, :]) .* exp_mu[month, :]

    inflow = rhs .- coef .* prev_inflow
    return clamp.(inflow, 0.0, Inf)
end

# ─────────────────────────────────────────────────────────────────
# 3. ENVIRONMENT  (mirrors SDDP model constraints)
# ─────────────────────────────────────────────────────────────────

mutable struct HydrothermalEnv
    t::Int
    stored::Vector{Float64}   # reservoir volumes (state)
    inflow::Vector{Float64}   # last inflow realisation (state)
    rng::AbstractRNG
    horizon::Int
end

function HydrothermalEnv(horizon=T_HORIZON; seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    HydrothermalEnv(0, copy(stored_initial), copy(inflow_initial), rng, horizon)
end

obs_dim() = 2 * N_SYSTEMS            # stored(4) + inflow(4) = 8
act_dim() = 3 * N_SYSTEMS + 5 * 5   # hydro(4) + thermal(4) + spill(4) + exchange(25) = 37

function get_obs(env::HydrothermalEnv)
    # Normalise to ~[0,1] so the network sees consistent scale
    month = mod1(env.t + 1, 12)
    stored_norm = env.stored ./ stored_ub
    inflow_norm = env.inflow ./ exp_mu[month, :]
    Float32.(vcat(stored_norm, inflow_norm))
end

function reset!(env::HydrothermalEnv)
    env.t = 0
    env.stored = copy(stored_initial)
    env.inflow = copy(inflow_initial)
    return get_obs(env)
end

function env_step!(env::HydrothermalEnv, raw_action::Vector{Float32})
    """
    Projects raw (unbounded) network output onto the feasible region,
    then applies the same constraints as sddp_ts.jl.
    """
    month = mod1(env.t + 1, 12)
    demand_t = demand_mat[month, :]

    # ── Sigmoid projection → physical bounds ──────────────────────
    a = sigmoid.(raw_action)   # element-wise sigmoid → (0,1)

    # Hydro: bounded by generation capacity AND available water
    avail_water = env.stored .+ env.inflow
    hydro_gen = Float64.(a[1:4]) .* min.(hydro_gen_ub, avail_water)

    # Thermal: bounded by aggregate lb/ub (mirrors thermal_gen bounds)
    thermal_gen = thermal_agg_lb .+
                  Float64.(a[5:8]) .* (thermal_agg_ub .- thermal_agg_lb)

    # Spill: up to 10% of current storage
    spill = Float64.(a[9:12]) .* env.stored .* 0.1

    # ── Exchange (new) ─────────────────────────────────────────────
    # a[13:37] → 25 values → 5×5 matrix scaled by upper bounds
    exchange = reshape(Float64.(a[13:37]), 5, 5) .* exchange_ub_mat

    # No self-exchange (diagonal = 0)
    for i in 1:5
        exchange[i, i] = 0.0
    end

    # Enforce transshipment balance at node 5:
    #   sum(exchange[:,5]) == sum(exchange[5,:])
    # mirrors: sum(exchange[:,5]) - sum(exchange[5,:]) == 0 in sddp_ts.jl
    inflow_5 = sum(exchange[1:4, 5])
    outflow_5 = sum(exchange[5, 1:4])
    imbalance = inflow_5 - outflow_5
    if outflow_5 > 1e-8
        exchange[5, 1:4] .+= imbalance .* exchange[5, 1:4] ./ outflow_5
    end
    exchange = clamp.(exchange, 0.0, exchange_ub_mat)

    for i in 1:4
        max_exportable = hydro_gen[i] + thermal_gen[i]
        total_export = sum(exchange[i, :])
        if total_export > max_exportable
            exchange[i, :] .*= max_exportable / (total_export + 1e-8)
        end
    end

    # ── Hydrological mass balance ──────────────────────────────────
    # mirrors: stored[i].out + spill[i] + hydro_gen[i] - stored[i].in == inflow[i].out
    new_stored = env.stored .+ env.inflow .- hydro_gen .- spill
    new_stored = clamp.(new_stored, 0.0, stored_ub)

    # ── Energy balance with exchange ───────────────────────────────
    # mirrors: sum(thermal) + sum(deficit) + hydro - sum(exchange[i,:]) + sum(exchange[:,i]) == demand
    deficit = zeros(4)
    for i in 1:4
        net_export = sum(exchange[i, :]) - sum(exchange[:, i])
        shortfall = demand_t[i] - hydro_gen[i] - thermal_gen[i] + net_export
        deficit[i] = clamp(shortfall, 0.0, demand_t[i] * deficit_ub_frac[i])
    end

    # ── Cost (same objective terms as sddp_ts.jl) ─────────────────
    thermal_cost = dot(thermal_avg_cost, thermal_gen)
    deficit_cost = dot(deficit_obj_vec, deficit)
    spill_cost_val = SPILL_COST * sum(spill)
    total_cost = thermal_cost + deficit_cost + spill_cost_val
    # note: exchange cost is zero in data.jl so not added here

    reward = Float32(-total_cost / 1e6)   # scale for stable gradients

    # ── Advance state ──────────────────────────────────────────────
    env.stored = new_stored
    env.t += 1
    env.inflow = sample_inflow(env.t, env.inflow, env.rng)
    done = env.t >= env.horizon

    return get_obs(env), reward, done, total_cost
end

# ─────────────────────────────────────────────────────────────────
# 4. ACTOR-CRITIC NETWORKS
# ─────────────────────────────────────────────────────────────────

const OBS_DIM = obs_dim()
const ACT_DIM = act_dim()

struct A2CAgent
    actor::Chain              # obs → μ (ACT_DIM,)
    critic::Chain             # obs → scalar value
    log_std::Vector{Float32}  # learnable log std (ACT_DIM,), σ = exp(log_std)
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
        entropy += sum(0.5f0 .* (LOG_2PIE .+ 2f0 .* lnσ))   # Gaussian entropy
    end

    return (actor_l + VF_COEF * critic_l - ENT_COEF * entropy) / n
end

# ─────────────────────────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

const GAMMA = 0.99f0
const LR = 3f-4
const N_STEPS = 12   # one full seasonal cycle before each update
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
            # ── Collect up to N_STEPS transitions ────────────────
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

            # ── Discounted returns with bootstrapping ─────────────
            bootstrap = done ? 0.0f0 : agent.critic(obs)[1]
            n = length(rew_buf)
            returns = Vector{Float32}(undef, n)
            R = bootstrap
            for i in n:-1:1
                R = rew_buf[i] + GAMMA * R * (1f0 - Float32(done_buf[i]))
                returns[i] = R
            end

            # ── Normalised advantages ─────────────────────────────
            adv = returns .- val_buf
            adv = (adv .- mean(adv)) ./ (std(adv) .+ 1f-8)

            # ── Gradient update ───────────────────────────────────
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
            @printf "Ep %4d | avg cost (last 100): %8.2f M R\$/month\n" ep avg
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
            μ = agent.actor(obs)          # deterministic: no exploration noise
            _, _, done, cost = env_step!(env, μ)
            ep_cost += cost
        end
        push!(costs, ep_cost)
    end
    @printf "\nEvaluation (%d episodes, deterministic policy):\n" n_episodes
    @printf "  Mean total cost : %8.2f M R\$/month\n" mean(costs) / 1e6
    @printf "  Std             : %8.2f M R\$/month\n" std(costs) / 1e6
    return costs
end

# ─────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────

println("Training A2C for Hydrothermal Scheduling (T=$T_HORIZON months)...")
agent, ep_costs = train_a2c(seed=42)

# ── Learning curve plot ───────────────────────────────────────────
smooth_n = 50
smooth = [mean(ep_costs[max(1, i - smooth_n + 1):i]) for i in eachindex(ep_costs)]

p = plot(ep_costs ./ 1e6, alpha=0.3, label="Episode cost",
    xlabel="Episode", ylabel="Total Cost (M R\$/month)",
    title="A2C Learning Curve — Hydrothermal Scheduling")
plot!(p, smooth ./ 1e6, lw=2, label="$(smooth_n)-ep moving avg")
savefig(p, "a2c_learning_curve.png")
println("Learning curve saved → a2c_learning_curve.png")

# ── Final evaluation ──────────────────────────────────────────────
eval_costs = evaluate(agent)