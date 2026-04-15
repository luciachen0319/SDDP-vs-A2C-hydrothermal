using CSV, DataFrames, Printf

function extract_expert_data(sim_dir::String, n_sims::Int, T::Int, initial_stored::Vector{Float64})
    # Pre-allocate lists for States (X) and Actions (Y)
    X_data = Vector{Vector{Float32}}()
    Y_data = Vector{Vector{Float32}}()
    
    for sim in 1:n_sims
        path_sim = joinpath(sim_dir, "simulation_$sim")
        
        # Track the previous storage to calculate hydro generation
        prev_stored = copy(initial_stored)
        
        for t in 1:T
            path_stage = joinpath(path_sim, "stage_$t")
            
            # --- 1. Read the raw data ---
            df_inflow  = CSV.read(joinpath(path_stage, "inflow.csv"), DataFrame)
            df_stored  = CSV.read(joinpath(path_stage, "stored.csv"), DataFrame)
            df_spill   = CSV.read(joinpath(path_stage, "spill.csv"), DataFrame)
            df_deficit = CSV.read(joinpath(path_stage, "deficit.csv"), DataFrame)
            df_exchange= CSV.read(joinpath(path_stage, "exchange.csv"), DataFrame)
            df_thermal = CSV.read(joinpath(path_stage, "thermal_gen.csv"), DataFrame)
            
            # Extract raw vectors
            curr_inflow = df_inflow.value
            curr_stored = df_stored.value
            curr_spill  = df_spill.value
            
            # --- 2. Build the State Vector (X) ---
            month = mod1(t, 12)
            # Normalization (ensure this matches the max bounds used in A2C.jl's get_obs)
            # E.g., obs = [stored ./ MAX_STORED; inflow ./ MAX_INFLOW; sin; cos]
            # You will need to adapt the normalization here to match your `get_obs(env)`
            obs = Float32.(vcat(
                prev_stored ./ 200000.0,  # Replace with actual stored_ub
                curr_inflow ./ 100000.0,  # Replace with actual inflow max
                [sin(2π * month / 12), cos(2π * month / 12)]
            ))
            push!(X_data, obs)
            
            # --- 3. Build the Action Vector (Y) ---
            # a. Thermal Gen (95 values)
            thermal_action = Float32.(df_thermal.value)
            
            # b. Hydro Gen (Calculated via mass balance)
            hydro_action = Float32.(prev_stored .+ curr_inflow .- curr_stored .- curr_spill)
            
            # c. Deficit (Sum the 4 depths per region)
            deficit_action = zeros(Float32, 4)
            for r in 1:nrow(df_deficit)
                reg = df_deficit.region[r]
                deficit_action[reg] += df_deficit.value[r]
            end
            
            # d. Spill
            spill_action = Float32.(curr_spill)
            
            # e. Exchange (Flattened 5x5)
            exchange_action = Float32.(df_exchange.value)
            
            # Concatenate the exact same way your A2C Actor outputs them
            target_action = vcat(
                thermal_action, 
                hydro_action, 
                deficit_action, 
                spill_action, 
                exchange_action
            )
            
            # NOTE: If your A2C actor outputs normalized values (e.g. 0 to 1 via Sigmoid),
            # you must divide `target_action` by the absolute upper bounds here!
            push!(Y_data, target_action)
            
            # Update previous storage for the next stage
            prev_stored = copy(curr_stored)
        end
    end
    
    # Convert arrays to Flux-friendly matrices (Features x Samples)
    X_mat = hcat(X_data...)
    Y_mat = hcat(Y_data...)
    
    return X_mat, Y_mat
end