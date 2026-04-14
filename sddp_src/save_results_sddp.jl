# save_results_sddp.jl
# Function to save simulation results
function save_results_sddp(simulations, output_dir; thermal, save_individual=true)
    mkpath(output_dir)
    n_stages = length(simulations[1])
    n_sims = length(simulations)
    
    # Initialize storage for sums
    objectives_sum = zeros(n_stages)
    z_sum = zeros(4, 3, n_stages)
    stored_sum = zeros(4, n_stages)
    spill_sum = zeros(4, n_stages)
    hydro_sum = zeros(4, n_stages)
    deficit_sum = zeros(4, 4, n_stages)
    exchange_sum = zeros(5, 5, n_stages)
    thermal_sum = Dict(p => zeros(size(thermal[p], 1), n_stages) for p in 1:4)
    inflow_sum = zeros(4, n_stages)
    
    # Process each simulation
    for (sim_idx, simulation) in enumerate(simulations)
        sim_dir = joinpath(output_dir, "simulation_$sim_idx")
        if save_individual
            mkpath(sim_dir)
        end
        
        # Save and accumulate stage objectives
        objectives = [stage[:stage_objective] for stage in simulation]
        
        objectives_sum .+= objectives
        if save_individual
            CSV.write(
                joinpath(sim_dir, "objective.csv"), 
                DataFrame(stage = 1:n_stages, objective = objectives )
            )
        end
        
        # Process each stage
        for (stage_idx, stage_data) in enumerate(simulation)
            stage_dir = joinpath(sim_dir, "stage_$stage_idx")
            if save_individual
                mkpath(stage_dir)
            end
            
            # Process z
            if haskey(stage_data, :z)
                # Extract SDDP.State objects in a 4×3 matrix
                values = [stage_data[:z][r, d] for r in 1:4, d in 1:3]
                
                # Convert each state to a numeric value, e.g., using the :out field
                numeric_values = [s.out for s in values]

                # Now perform element-wise addition
                z_sum[:, :, stage_idx] .+= numeric_values

                if save_individual
                    CSV.write(
                        joinpath(stage_dir, "z.csv"),
                        DataFrame(
                            region = repeat(1:4, inner = 3),
                            level = repeat(1:3, outer = 4),
                            value = vec(numeric_values)
                        )
                    )
                end
            end

            # Process stored
            if haskey(stage_data, :stored)
                values = [stage_data[:stored][i].out for i in 1:4]  # Add .out here
                stored_sum[:, stage_idx] .+= values
                if save_individual
                    CSV.write(
                        joinpath(stage_dir, "stored.csv"),
                        DataFrame(reservoir = 1:4, value = values )
                    )
                end
            end
            
            # Process spill
            if haskey(stage_data, :spill)
                values = [stage_data[:spill][i] for i in 1:4]
                spill_sum[:, stage_idx] .+= values
                if save_individual
                    CSV.write(
                        joinpath(stage_dir, "spill.csv"),
                        DataFrame(reservoir = 1:4, value = values )
                    )
                end
            end
            
            # Process hydro
            if haskey(stage_data, :hydro)
                values = [stage_data[:hydro][i] for i in 1:4]
                hydro_sum[:, stage_idx] .+= values
                if save_individual
                    CSV.write(
                        joinpath(stage_dir, "hydro.csv"),
                        DataFrame(plant = 1:4, value = values )
                    )
                end
            end
            
            # Process deficit
            if haskey(stage_data, :deficit)
                values = [stage_data[:deficit][r,d] for r in 1:4, d in 1:4]
                deficit_sum[:, :, stage_idx] .+= values
                if save_individual
                    CSV.write(
                        joinpath(stage_dir, "deficit.csv"),
                        DataFrame(
                            region = repeat(1:4, inner=4),
                            depth = repeat(1:4, outer=4),
                            value = vec(values) 
                        )
                    )
                end
            end
            
            # Process exchange
            if haskey(stage_data, :exchange)
                values = [stage_data[:exchange][i,j] for i in 1:5, j in 1:5]
                exchange_sum[:, :, stage_idx] .+= values
                if save_individual
                    CSV.write(
                        joinpath(stage_dir, "exchange.csv"),
                        DataFrame(
                            from = repeat(1:5, inner=5),
                            to = repeat(1:5, outer=5),
                            value = vec(values) 
                        )
                    )
                end
            end
            
            # Process thermal_gen
            if haskey(stage_data, :thermal_gen)
                for p in 1:4
                    values = [stage_data[:thermal_gen][p,u] for u in 1:size(thermal[p], 1)]
                    thermal_sum[p][:, stage_idx] .+= values
                end
                if save_individual
                    thermal_data = []
                    thermal_plants = []
                    thermal_units = []
                    for p in 1:4
                        for u in 1:size(thermal[p], 1)
                            push!(thermal_data, stage_data[:thermal_gen][p,u])
                            push!(thermal_plants, p)
                            push!(thermal_units, u)
                        end
                    end
                    CSV.write(
                        joinpath(stage_dir, "thermal_gen.csv"),
                        DataFrame(
                            plant = thermal_plants,
                            unit = thermal_units,
                            value = thermal_data 
                        )
                    )
                end
            end
            
            # Process inflow
            if haskey(stage_data, :inflow)
                values = [stage_data[:inflow][i].out for i in 1:4]
                inflow_sum[:, stage_idx] .+= values
                if save_individual
                    CSV.write(
                        joinpath(stage_dir, "inflow.csv"),
                        DataFrame(reservoir = 1:4, value = values )
                    )
                end
            end
        end
    end
    
    # Save averages
    avg_dir = joinpath(output_dir, "average_results")
    mkpath(avg_dir)
    
    # Save average objectives
    CSV.write(
        joinpath(avg_dir, "objective.csv"),
        DataFrame(stage = 1:n_stages, objective = objectives_sum ./ n_sims )
    )
    
    # Save average results for each stage
    for stage_idx in 1:n_stages
        stage_dir = joinpath(avg_dir, "stage_$stage_idx")
        mkpath(stage_dir)
        
        # Save stored averages
        CSV.write(
            joinpath(stage_dir, "stored.csv"),
            DataFrame(reservoir = 1:4, value = stored_sum[:, stage_idx] ./ n_sims )
        )
        
        # Save spill averages
        CSV.write(
            joinpath(stage_dir, "spill.csv"),
            DataFrame(reservoir = 1:4, value = spill_sum[:, stage_idx] ./ n_sims )
        )
        
        # Save hydro averages
        CSV.write(
            joinpath(stage_dir, "hydro.csv"),
            DataFrame(plant = 1:4, value = hydro_sum[:, stage_idx] ./ n_sims )
        )
        
        # Save deficit averages
        CSV.write(
            joinpath(stage_dir, "deficit.csv"),
            DataFrame(
                region = repeat(1:4, inner=4),
                depth = repeat(1:4, outer=4),
                value = vec(deficit_sum[:, :, stage_idx]) ./ n_sims 
            )
        )
        
        # Save exchange averages
        CSV.write(
            joinpath(stage_dir, "exchange.csv"),
            DataFrame(
                from = repeat(1:5, inner=5),
                to = repeat(1:5, outer=5),
                value = vec(exchange_sum[:, :, stage_idx]) ./ n_sims 
            )
        )
        
        # Save thermal_gen averages
        thermal_data = []
        thermal_plants = []
        thermal_units = []
        for p in 1:4
            for u in 1:size(thermal[p], 1)
                push!(thermal_data, thermal_sum[p][u, stage_idx] / n_sims)
                push!(thermal_plants, p)
                push!(thermal_units, u)
            end
        end
        CSV.write(
            joinpath(stage_dir, "thermal_gen.csv"),
            DataFrame(
                plant = thermal_plants,
                unit = thermal_units,
                value = thermal_data 
            )
        )
        
        # Save inflow averages
        CSV.write(
            joinpath(stage_dir, "inflow.csv"),
            DataFrame(reservoir = 1:4, value = inflow_sum[:, stage_idx] ./ n_sims )
        )
    end
    
    println("Simulation results have been saved to $output_dir")
end