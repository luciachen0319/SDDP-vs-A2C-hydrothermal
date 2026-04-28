"""
A2C for Hydrothermal Scheduling — Julia implementation
Mirrors sddp_ts.jl:
  - Same 4 subsystems: SE, S, NE, N
  - Same PAR(1) inflow model (γ, Σ, ㄕㄛmu from data.jl)
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
using CSV
using DataFrames

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

# Merit-order economic dispatch projection onto the power balance hyperplane.
# Given fixed hydro and exchange, adjusts thermal and deficit so that
# generation == demand for each subsystem.  Thermal units are dispatched
# cheapest-first (thermal_sort_idx); deficit levels are added/removed in
# cost order (deficit_obj_vec ascending).
# Three-pass projection onto the power balance hyperplane:
#   Pass 1: compute per-subsystem power gaps
#   Pass 2: route surplus from over-generating subsystems to under-generating ones
#           via available exchange links (cheap spatial rebalancing before deficit)
#   Pass 3: adjust thermal and deficit for any remaining gaps
# Node-5 transshipment balance is preserved because pass 2 only touches
# exchange[j,i] for j,i in 1:4 (direct subsystem-to-subsystem links).
function project_power_balance!(thermal_units, deficit_mat, hydro_gen, exchange, demand_t)
    # Pass 1: compute gaps
    gaps = zeros(4)
    for i in 1:4
        net_import = sum(exchange[:, i]) - sum(exchange[i, :])
        gaps[i] = demand_t[i] - hydro_gen[i] -
                  sum(thermal_units[thermal_idx[i]]) -
                  sum(deficit_mat[i, :]) - net_import
    end

    # Pass 2a: direct exchange between subsystems (j→i, both in 1:4)
    for i in 1:4          # deficit region
        gaps[i] <= 1e-6 && continue
        for j in 1:4      # surplus region
            j == i && continue
            gaps[j] >= -1e-6 && continue
            headroom = exchange_ub_mat[j, i] - exchange[j, i]
            headroom <= 1e-8 && continue
            transfer = min(-gaps[j], gaps[i], headroom)
            if transfer > 1e-6
                exchange[j, i] += transfer
                gaps[i] -= transfer
                gaps[j] += transfer
            end
        end
    end

    # Pass 2b: transit routing through node 5 (the transshipment hub).
    # SDDP routes power as region1→node5→region4 because direct r1→r4 links
    # may not exist. Adding equal amounts to exchange[j,5] and exchange[5,i]
    # preserves node 5's transshipment balance (inflow == outflow).
    for i in 1:4          # deficit region
        gaps[i] <= 1e-6 && continue
        for j in 1:4      # surplus region
            j == i && continue
            gaps[j] >= -1e-6 && continue
            h_j5 = exchange_ub_mat[j, 5] - exchange[j, 5]   # headroom j→node5
            h_5i = exchange_ub_mat[5, i] - exchange[5, i]   # headroom node5→i
            headroom = min(h_j5, h_5i)
            headroom <= 1e-8 && continue
            transfer = min(-gaps[j], gaps[i], headroom)
            if transfer > 1e-6
                exchange[j, 5] += transfer
                exchange[5, i] += transfer
                gaps[i] -= transfer
                gaps[j] += transfer
            end
        end
    end

    # Pass 3: thermal / deficit adjustments for any remaining gaps
    for i in 1:4
        abs(gaps[i]) <= 1e-6 && continue

        if gaps[i] > 0  # under-generation: cheapest thermal first, then deficit
            for u in thermal_sort_idx[i]
                gaps[i] <= 1e-6 && break
                add = min(gaps[i], thermal_unit_ub[u] - thermal_units[u])
                thermal_units[u] += add
                gaps[i] -= add
            end
            for j in 1:4
                gaps[i] <= 1e-6 && break
                cap = demand_t[i] * deficit_ub_frac[j] - deficit_mat[i, j]
                add = min(gaps[i], max(cap, 0.0))
                deficit_mat[i, j] += add
                gaps[i] -= add
            end
        else  # over-generation: shed most expensive first
            excess = -gaps[i]
            for j in 4:-1:1
                excess <= 1e-6 && break
                reduce = min(excess, deficit_mat[i, j])
                deficit_mat[i, j] -= reduce
                excess -= reduce
            end
            for u in reverse(thermal_sort_idx[i])
                excess <= 1e-6 && break
                reduce = min(excess, thermal_units[u] - thermal_unit_lb[u])
                thermal_units[u] -= reduce
                excess -= reduce
            end
        end
    end
end

function env_step!(env::HydrothermalEnv, raw_action::Vector{Float32})
    month = mod1(env.t + 1, 12)
    demand_t = demand_mat[month, :]
    #a = sigmoid.(raw_action)

    # ── Action Parsing with Domain Knowledge Shifts ──
    a = zeros(Float32, length(raw_action))

    # [1:4] Hydro: Shift +3.0 so it starts near 95% usage (save thermal/prevent deficits)
    a[1:4] = sigmoid.(raw_action[1:4] .+ 3.0f0)

    # [5:8] Spill: Shift -3.0 so it starts near 5% optional spill (stop wasting water)
    a[5:8] = sigmoid.(raw_action[5:8] .- 3.0f0)

    # [9:103] Thermal: No shift, starts at 50% capacity
    a[9:103] = sigmoid.(raw_action[9:103])

    # [104:119] Deficit: Shift -5.0 so it starts near 0.6% deficit! 
    # (The agent should never intentionally ask for a blackout)
    a[104:119] = sigmoid.(raw_action[104:119] .- 5.0f0)

    # [120:144] Exchange: Multiplier of 3.0 (keeps 50% starting capacity, but makes it highly sensitive)
    a[120:144] = sigmoid.(raw_action[120:144] .* 3.0f0)

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

    # ── Step 6: Power balance projection (merit-order economic dispatch) ──
    project_power_balance!(thermal_units, deficit_mat, hydro_gen, exchange, demand_t)
    for i in 1:4
        thermal_gen[i] = sum(thermal_units[thermal_idx[i]])
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
# 6b. BEHAVIORAL CLONING (pre-train actor on SDDP demonstrations)
# ─────────────────────────────────────────────────────────────────

"""
    behavioral_cloning(agent, sddp_trajectories; n_epochs, batch_size)

Pre-trains the ACTOR to imitate SDDP actions via supervised learning.
Loss = MSE between actor output and SDDP raw action (inverse-sigmoid).
After this phase the actor starts near the SDDP policy before RL exploration.
"""
function behavioral_cloning(agent::A2CAgent, sddp_trajectories;
    n_epochs=50, batch_size=64)
    opt_actor = Flux.setup(Adam(1f-3), agent.actor)

    # collect all (obs, sddp_raw_action) pairs
    X = Vector{Float32}[]
    Y = Vector{Float32}[]
    for traj in sddp_trajectories
        for (obs, action, _, _, _) in traj
            push!(X, obs)
            push!(Y, action)
        end
    end

    num_samples = length(X)
    println("Behavioral cloning: $num_samples (obs, action) pairs, $n_epochs epochs...")

    for epoch in 1:n_epochs
        indices = shuffle(1:num_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in 1:batch_size:num_samples
            idx = indices[i:min(i + batch_size - 1, num_samples)]
            batch_x = reduce(hcat, X[idx])   # OBS_DIM × batch
            batch_y = reduce(hcat, Y[idx])   # ACT_DIM × batch

            loss, grads = Flux.withgradient(agent.actor) do ac
                pred = ac(batch_x)
                mean((pred .- batch_y) .^ 2)   # MSE — actor mimics SDDP
            end

            Flux.update!(opt_actor, agent.actor, grads[1])
            epoch_loss += loss
            n_batches += 1
        end

        if epoch % 10 == 0
            @printf "BC Epoch %3d | Actor MSE: %.6f\n" epoch (epoch_loss / n_batches)
        end
    end
    println("Behavioral cloning complete.")
end

const GAMMA = 0.99f0
const LR = 3f-4
const N_STEPS = 12        # one full seasonal cycle per update
const N_EPISODES = 10_000
const BC_LAMBDA_START = 0.5f0   # initial weight on BC loss (1.0 = pure imitation)
const BC_LAMBDA_END = 0.0f0   # final weight on BC loss (0.0 = pure A2C)

function train_a2c(agent::A2CAgent, sddp_trajectories=nothing;
    seed=42, n_episodes=N_EPISODES)
    rng = MersenneTwister(seed)
    env = HydrothermalEnv(T_HORIZON; seed=seed)
    opt = Flux.setup(Adam(LR), agent)

    # build BC lookup: pool all SDDP (obs, action) pairs for mixed loss
    bc_obs = Vector{Float32}[]
    bc_act = Vector{Float32}[]
    if sddp_trajectories !== nothing
        for traj in sddp_trajectories
            for (obs, action, _, _, _) in traj
                push!(bc_obs, obs)
                push!(bc_act, action)
            end
        end
        println("Mixed training: $(length(bc_obs)) SDDP pairs available for BC loss.")
    end

    ep_costs = Float64[]

    for ep in 1:n_episodes
        obs = reset!(env)
        ep_cost = 0.0
        done = false

        # λ anneals linearly from BC_LAMBDA_START → BC_LAMBDA_END over all episodes
        λ_bc = BC_LAMBDA_START +
               (BC_LAMBDA_END - BC_LAMBDA_START) * Float32(ep - 1) / Float32(n_episodes - 1)

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
                # A2C loss
                a2c_l = a2c_loss(ag, obs_buf, act_buf, returns, adv)

                # BC loss: pull actor toward SDDP actions on a random mini-batch
                bc_l = 0.0f0
                if length(bc_obs) > 0 && λ_bc > 0.0f0
                    n_bc = min(length(obs_buf), length(bc_obs))
                    idx = rand(1:length(bc_obs), n_bc)
                    bx = reduce(hcat, bc_obs[idx])
                    by = reduce(hcat, bc_act[idx])
                    pred = ag.actor(bx)
                    bc_l = mean((pred .- by) .^ 2)
                end

                # mixed loss: λ anneals from BC_LAMBDA_START → 0
                (1.0f0 - λ_bc) * a2c_l + λ_bc * bc_l
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
            @printf "Ep %4d | λ_bc=%.3f | avg cost (last 100): %8.2f M R\$\n" ep λ_bc avg
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
            #a = sigmoid.(μ)
            # ── Action Parsing with Domain Knowledge Shifts ──
            a = zeros(Float32, length(μ))

            # [1:4] Hydro: Shift +3.0 so it starts near 95% usage (save thermal/prevent deficits)
            a[1:4] = sigmoid.(μ[1:4] .+ 3.0f0)

            # [5:8] Spill: Shift -3.0 so it starts near 5% optional spill (stop wasting water)
            a[5:8] = sigmoid.(μ[5:8] .- 3.0f0)

            # [9:103] Thermal: No shift, starts at 50% capacity
            a[9:103] = sigmoid.(μ[9:103])

            # [104:119] Deficit: Shift -5.0 so it starts near 0.6% deficit! 
            # (The agent should never intentionally ask for a blackout)
            a[104:119] = sigmoid.(μ[104:119] .- 5.0f0)

            # [120:144] Exchange: Multiplier of 3.0 (keeps 50% starting capacity, but makes it highly sensitive)
            a[120:144] = sigmoid.(μ[120:144] .* 3.0f0)

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
# 9. SDDP DATA LOADING AND TRAINING
# ─────────────────────────────────────────────────────────────────

const SDDP_DATA_PATH = joinpath(@__DIR__, "output/simulations/sddp_eval/ts/")
const N_SDDP_SIMULATIONS = 100
# SDDP.jl records one entry per stochastic stage; the deterministic first stage
# is not separately returned, so each simulation has 23 saved stages (SDDP stages 2-24).
const SDDP_HORIZON = 23

"""
    load_sddp_data(base_path::String, num_simulations::Int, horizon::Int)

Loads SDDP simulation data from the specified path.

Reconstructs observations and rewards from the 6 CSV files generated per stage:
`inflow.csv`, `thermal_gen.csv`, `exchange.csv`, `spill.csv`, `deficit.csv`, `storage.csv`.

Returns a Vector of trajectories, where each trajectory is a Vector of
`(obs, sddp_raw_action, reward, next_obs, done)` tuples.
"""
function load_sddp_data(base_path::String, num_simulations::Int, horizon::Int)
    all_trajectories = []
    # We need the initial state to start the first stage of each simulation
    initial_stored = Float32.(stored_initial)
    initial_inflow = Float32.(inflow_initial)

    println("Loading SDDP simulation data from $base_path...")

    for sim_idx in 1:num_simulations
        sim_path = joinpath(base_path, "simulation_$(sim_idx)")
        current_trajectory = []

        # Track state across stages within a simulation
        current_stored = copy(initial_stored)
        current_inflow = copy(initial_inflow)

        sim_ok = true
        for t in 1:horizon # t is the stage number, 1 to 24
            stage_path = joinpath(sim_path, "stage_$(t)")

            # --- Load SDDP decision values ---
            local val_hydro, val_inflow, val_spill, val_thermal, val_deficit, val_exchange, next_stored
            try
                val_inflow  = Float64.(CSV.read(joinpath(stage_path, "inflow.csv"),      DataFrame).value)
                val_spill   = Float64.(CSV.read(joinpath(stage_path, "spill.csv"),       DataFrame).value)
                val_thermal = Float64.(CSV.read(joinpath(stage_path, "thermal_gen.csv"), DataFrame).value)
                val_deficit = Float64.(CSV.read(joinpath(stage_path, "deficit.csv"),     DataFrame).value)
                val_exchange = Float64.(CSV.read(joinpath(stage_path, "exchange.csv"),   DataFrame).value)
                next_stored  = Float64.(CSV.read(joinpath(stage_path, "stored.csv"),     DataFrame).value)

                # Reconstruct hydro generation from the water balance:
                #   stored_out = stored_in + inflow - hydro_gen - spill
                # => hydro_gen = stored_in + inflow - stored_out - spill
                # This is always correct and avoids relying on hydro.csv,
                # which is all-zeros in existing data due to a save-function bug
                # (:hydro_gen key was saved as :hydro, now fixed in save_results_sddp.jl).
                val_hydro = max.(0.0, Float64.(current_stored) .+ val_inflow .- next_stored .- val_spill)
            catch
                sim_ok = false
                break
            end

            # --- Construct current observation ---
            month = mod1(t, 12)
            obs_current = vcat(Float32[month/12.0],
                Float32.(current_stored ./ stored_ub),
                Float32.(current_inflow ./ 100000.0),
                Float32.(demand_mat[month, :] ./ 30000.0))

            # --- Calculate Cost (to create Reward) ---
            # Re-using logic from env_step! to ensure consistency
            thermal_cost_val = dot(thermal_unit_cost, val_thermal)
            deficit_cost_val = sum(val_deficit[i] * deficit_obj_vec[mod1(i, 4)] for i in 1:16)
            exchange_cost_val = sum(reshape(val_exchange, 5, 5) .* exchange_cost_mat)
            spill_cost_val = SPILL_COST * sum(val_spill)
            stage_total_cost = thermal_cost_val + deficit_cost_val + exchange_cost_val + spill_cost_val
            sddp_reward = Float32(-stage_total_cost / 1e6)

            # --- Normalize values to [0,1] for A2C actions ---
            # We map actual generation to fractions based on the bounds defined in constants
            avail_water = current_stored .+ current_inflow
            a_hydro = Float32.(val_hydro ./ max.(1.0, min.(hydro_gen_ub, avail_water)))
            a_spill = Float32.(val_spill ./ max.(1.0, current_stored .* 0.1))
            a_thermal = Float32.((val_thermal .- thermal_unit_lb) ./ max.(1.0, thermal_unit_ub .- thermal_unit_lb))

            # Deficit and Exchange normalization
            demand_t = demand_mat[month, :]
            caps_deficit = vcat([demand_t[i] .* deficit_ub_frac for i in 1:4]...)
            a_deficit = Float32.(val_deficit ./ max.(1.0, caps_deficit))
            a_exchange = Float32.(val_exchange ./ max.(1.0, vec(exchange_ub_mat)))

            # Combine and invert each action transformation to get the pre-transformation
            # raw targets the actor must produce.  Each group has its own shift/scale:
            #   hydro    [1:4]    : a = sigmoid(raw + 3)  →  raw = logit(a) - 3
            #   spill    [5:8]    : a = sigmoid(raw - 3)  →  raw = logit(a) + 3
            #   thermal  [9:103]  : a = sigmoid(raw)      →  raw = logit(a)
            #   deficit  [104:119]: a = sigmoid(raw - 5)  →  raw = logit(a) + 5
            #   exchange [120:144]: a = sigmoid(raw * 3)  →  raw = logit(a) / 3
            sddp_action_a = vcat(a_hydro, a_spill, a_thermal, a_deficit, a_exchange)
            sddp_action_a_clamped = clamp.(sddp_action_a, 1e-6, 1.0f0 - 1e-6)
            logit_a = Float32.(log.(sddp_action_a_clamped ./ (1.0f0 .- sddp_action_a_clamped)))
            sddp_raw_action = vcat(
                logit_a[1:4]    .- 3.0f0,   # hydro
                logit_a[5:8]    .+ 3.0f0,   # spill
                logit_a[9:103],              # thermal
                logit_a[104:119] .+ 5.0f0,  # deficit
                logit_a[120:144] ./ 3.0f0   # exchange
            )

            # inflow[i].out from SDDP is the realized water inflow for this stage —
            # use it directly rather than reconstructing from the water balance.
            realized_inflow = val_inflow

            # --- Construct next observation ---
            next_month = mod1(t + 1, 12)
            obs_next = vcat(Float32[next_month/12.0],
                Float32.(next_stored ./ stored_ub),
                Float32.(realized_inflow ./ 100000.0), # Approximate for the next state
                Float32.(demand_mat[next_month, :] ./ 30000.0))

            # --- Determine if done ---
            done = (t == horizon)

            push!(current_trajectory, (obs_current, sddp_raw_action, sddp_reward, obs_next, done))

            # Update state for next stage
            current_stored = Float32.(next_stored)
            current_inflow = Float32.(realized_inflow)
        end
        sim_ok && push!(all_trajectories, current_trajectory)
    end
    println("Finished loading SDDP data. Total trajectories: $(length(all_trajectories))")
    return all_trajectories
end

function warm_start_critic(agent::A2CAgent, sddp_trajectories; n_epochs=50, batch_size=64)
    # Use a slightly higher learning rate for supervised warm-up
    opt_critic = Flux.setup(Adam(1f-3), agent.critic)

    X = Vector{Float32}[]
    Y = Float32[]

    # Process trajectories to calculate cumulative discounted returns
    for traj in sddp_trajectories
        T = length(traj)
        running_return = 0.0f0

        # Calculate returns backwards: G_t = r_t + gamma * G_{t+1}
        for t in T:-1:1
            obs, _, reward, _, _ = traj[t]
            running_return = reward + GAMMA * running_return
            push!(X, obs)
            push!(Y, running_return)
        end
    end

    num_samples = length(X)
    println("Warm-starting Critic on $num_samples state-value pairs for $n_epochs epochs...")

    for epoch in 1:n_epochs
        indices = shuffle(1:num_samples)
        epoch_loss = 0.0

        for i in 1:batch_size:num_samples
            idx = indices[i:min(i + batch_size - 1, num_samples)]
            batch_x = reduce(hcat, X[idx])
            batch_y = Y[idx]

            loss, grads = Flux.withgradient(agent.critic) do c
                pred = vec(c(batch_x))
                mean((pred .- batch_y) .^ 2) # MSE Loss
            end

            Flux.update!(opt_critic, agent.critic, grads[1])
            epoch_loss += loss
        end

        if epoch % 10 == 0
            @printf "Warm-up Epoch %d | MSE Loss: %.4f\n" epoch (epoch_loss / ceil(num_samples / batch_size))
        end
    end
end

# ─────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────

println("Training A2C for Hydrothermal Scheduling (T=$T_HORIZON months)...")

# ── Phase 0: Load SDDP demonstrations ────────────────────────────
println("\n--- PHASE 0: Loading SDDP Data ---")
phase0_time = @elapsed sddp_trajectories = load_sddp_data(SDDP_DATA_PATH, N_SDDP_SIMULATIONS, SDDP_HORIZON)

# ── Phase 1: Behavioral cloning — warm-start actor AND critic ─────
println("\n--- PHASE 1: Behavioral Cloning (Actor + Critic Warm-Start) ---")
agent = A2CAgent()
phase1_time = @elapsed begin
    behavioral_cloning(agent, sddp_trajectories; n_epochs=50)
    warm_start_critic(agent, sddp_trajectories; n_epochs=50)
end

# ── Phase 2: Mixed BC + A2C training with λ annealing ─────────────
# λ_bc starts at 0.5 (half imitation, half RL) and anneals to 0 (pure RL)
# agent continues exploring while still being pulled toward SDDP behavior early on
println("\n--- PHASE 2: Mixed BC + A2C Training (λ annealing 0.5 → 0) ---")
phase2_time = @elapsed agent, ep_costs = train_a2c(agent, sddp_trajectories; seed=42, n_episodes=N_EPISODES)

# learning curve
smooth_n = 50
smooth = [mean(ep_costs[max(1, i - smooth_n + 1):i]) for i in eachindex(ep_costs)]
p = plot(ep_costs ./ 1e6, alpha=0.3, label="Episode avg cost",
    xlabel="Episode", ylabel="Avg Total Cost (M R\$)",
    title="A2C Learning Curve (Critic Warm-Start)")
plot!(p, smooth ./ 1e6, lw=2, label="$(smooth_n)-ep moving avg")
savefig(p, "a2c_learning_curve.png")
println("Learning curve saved → a2c_learning_curve.png")

evaluate_detailed(agent)

const N_EVAL = 100
const EVAL_SEED = 204
scenarios_inflow, _ = generate_eval_scenarios(
    N_EVAL, T_HORIZON, gamma_mat, sigma_mats, exp_mu,
    inflow_initial; seed=EVAL_SEED
)
eval_time = @elapsed shared_eval_costs = evaluate_on_scenarios(agent, scenarios_inflow)

println("\n--- Performance Summary ---")
println("Phase 0 (Data Loading) Time: ", round(phase0_time, digits=2), " seconds")
println("Phase 1 (Warm-start) Time:   ", round(phase1_time, digits=2), " seconds")
println("Phase 2 (RL Training) Time:  ", round(phase2_time, digits=2), " seconds")
println("Evaluation Time:             ", round(eval_time, digits=2), " seconds")
println("Total Execution Time:        ", round(phase0_time + phase1_time + phase2_time + eval_time, digits=2), " seconds")