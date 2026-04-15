"""
A2C for Hydrothermal Scheduling — Julia implementation with Behavioral Cloning
Mirrors sddp_ts.jl:
  - Same 4 subsystems: SE, S, NE, N
  - Same PAR(1) inflow model (γ, Σ, exp_mu from data.jl)
  - Pre-trains on SDDP exact outputs (Imitation Learning) before RL fine-tuning.
"""

using Flux
using Flux: mse, setup, train!
using MLUtils: DataLoader
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
gamma_mat = reduce(vcat, [γ[m]' for m in 1:12])        # 12×4
sigma_mats = [Σ[m] for m in 1:12]                      # 12 × (4×4)

# Hydro bounds
stored_initial = storedEnergy_initial
inflow_initial = round.(INITIAL_INFLOWS, digits=0)
stored_ub = storedEnergy_ub
# (Ensure hydro_gen_ub and other bounds are defined here per your original file)

# ─────────────────────────────────────────────────────────────────
# 2. BEHAVIORAL CLONING: DATA EXTRACTION
# ─────────────────────────────────────────────────────────────────

function extract_expert_data(sim_dir::String, n_sims::Int, T::Int, initial_stored::Vector{Float64})
    println("Extracting SDDP Expert Data from $sim_dir...")
    X_data = Vector{Vector{Float32}}()
    Y_data = Vector{Vector{Float32}}()
    
    for sim in 1:n_sims
        path_sim = joinpath(sim_dir, "simulation_$sim")
        prev_stored = copy(initial_stored)
        
        for t in 1:T
            path_stage = joinpath(path_sim, "stage_$t")
            
            # Read SDDP outputs
            df_inflow  = CSV.read(joinpath(path_stage, "inflow.csv"), DataFrame)
            df_stored  = CSV.read(joinpath(path_stage, "stored.csv"), DataFrame)
            df_spill   = CSV.read(joinpath(path_stage, "spill.csv"), DataFrame)
            df_deficit = CSV.read(joinpath(path_stage, "deficit.csv"), DataFrame)
            df_exchange= CSV.read(joinpath(path_stage, "exchange.csv"), DataFrame)
            df_thermal = CSV.read(joinpath(path_stage, "thermal_gen.csv"), DataFrame)
            
            curr_inflow = df_inflow.value
            curr_stored = df_stored.value
            curr_spill  = df_spill.value
            
            # --- Build State Vector (X) ---
            month = mod1(t, 12)
            # IMPORTANT: Ensure this normalization exactly matches your `get_obs()`
            obs = Float32.(vcat(
                prev_stored ./ stored_ub, 
                curr_inflow ./ 100000.0,  # Or whatever max inflow you use
                [sin(2π * month / 12), cos(2π * month / 12)]
            ))
            push!(X_data, obs)
            
            # --- Build Action Vector (Y) ---
            thermal_action = Float32.(df_thermal.value) # 95 dims
            hydro_action   = Float32.(prev_stored .+ curr_inflow .- curr_stored .- curr_spill) # 4 dims
            
            # Deficit: Sum the 4 depths per region into 1 level per region (4 dims)
            deficit_action = zeros(Float32, 4)
            for r in 1:nrow(df_deficit)
                reg = df_deficit.region[r]
                deficit_action[reg] += df_deficit.value[r]
            end
            
            spill_action    = Float32.(curr_spill) # 4 dims
            exchange_action = Float32.(df_exchange.value) # 25 dims
            
            # Concatenate to match your Actor's exact output layer
            # Total size = 95 + 4 + 4 + 4 + 25 = 132 dimensions
            target_action = vcat(thermal_action, hydro_action, deficit_action, spill_action, exchange_action)
            
            # IMPORTANT: If your Actor network uses a Sigmoid() final layer, 
            # you MUST divide `target_action` by the absolute capacities here so it falls between 0 and 1!
            push!(Y_data, target_action)
            
            prev_stored = copy(curr_stored)
        end
    end
    
    return hcat(X_data...), hcat(Y_data...)
end

function pretrain_behavioral_cloning!(agent, X_expert, Y_expert; epochs=50, batchsize=32)
    println("Pre-training Actor via Behavioral Cloning for $epochs epochs...")
    opt = setup(Adam(1e-3), agent.actor)
    data_loader = DataLoader((X_expert, Y_expert), batchsize=batchsize, shuffle=true)
    
    for epoch in 1:epochs
        total_loss = 0.0
        for (x_batch, y_batch) in data_loader
            loss, grads = Flux.withgradient(agent.actor) do m
                pred = m(x_batch)
                mse(pred, y_batch)
            end
            Flux.update!(opt, agent.actor, grads[1])
            total_loss += loss
        end
        if epoch % 10 == 0 || epoch == 1
            @printf("  Epoch %d | Actor MSE Loss: %.6f\n", epoch, total_loss / length(data_loader))
        end
    end
    println("Behavioral Cloning Complete.")
end

# ─────────────────────────────────────────────────────────────────
# 3. ENVIRONMENT & AGENT ARCHITECTURE
# ─────────────────────────────────────────────────────────────────

# --- INSERT YOUR EXISTING `HydroEnv` STRUCT AND `get_obs` FUNCTION HERE ---
# (Keep your state normalization exactly the same as in `extract_expert_data`)

# --- INSERT YOUR EXISTING `env_step!` FUNCTION HERE ---
# (This remains unchanged. It receives the actions from the Actor and computes the transition & cost)

# --- INSERT YOUR EXISTING `build_networks` AND AGENT DEFINITIONS HERE ---
# (Ensure your Actor network outputs exactly the 132-dimension vector we built above)

# ─────────────────────────────────────────────────────────────────
# 4. TRAINING LOGIC (A2C FINE-TUNING)
# ─────────────────────────────────────────────────────────────────

# --- INSERT YOUR EXISTING `train_a2c` FUNCTION HERE ---
# (Keep the PPO/A2C RL loop exactly as you had it. We will call it after BC.)

# --- INSERT YOUR EXISTING `evaluate_a2c` FUNCTION HERE ---

# ─────────────────────────────────────────────────────────────────
# 5. MAIN EXECUTION (PIPELINE)
# ─────────────────────────────────────────────────────────────────

function main()
    println("--- Starting Hybrid A2C Pipeline ---")
    
    # 1. Initialize Agent
    # (Replace `build_networks` with whatever your instantiation function is called)
    # agent = build_networks(...) 
    
    # 2. Extract Expert Data
    sddp_path = "output/simulations/sddp_train/ts"
    X_mat, Y_mat = extract_expert_data(sddp_path, 100, T_HORIZON, stored_initial)
    
    # 3. Behavioral Cloning (Imitation Learning Phase)
    # This aligns the neural network weights perfectly with SDDP's optimal policy
    pretrain_behavioral_cloning!(agent, X_mat, Y_mat, epochs=50, batchsize=32)
    
    # Optional Check: You can run `evaluate_a2c(agent)` here to see how good the cloned policy is 
    # before RL even starts! It should already score very close to 72.5M R$.
    
    # 4. Reinforcement Learning (Fine-Tuning Phase)
    println("\nTransitioning to RL fine-tuning (A2C)...")
    agent, ep_costs = train_a2c(seed=42) # Call your existing training loop
    
    # 5. Plotting
    smooth_n = 50
    smooth = [mean(ep_costs[max(1, i - smooth_n + 1):i]) for i in eachindex(ep_costs)]
    p = plot(ep_costs ./ 1e6, alpha=0.3, label="Episode cost",
        xlabel="Episode", ylabel="Total Cost (M R\$)",
        title="A2C Learning Curve (Post-Imitation) — Hydrothermal")
    plot!(p, smooth ./ 1e6, lw=2, label="$(smooth_n)-ep moving avg")
    savefig(p, "a2c_learning_curve.png")
    println("\nSaved learning curve to a2c_learning_curve.png.")
    
    # 6. Final Evaluation
    evaluate_a2c(agent)
end

# Run the pipeline
main()