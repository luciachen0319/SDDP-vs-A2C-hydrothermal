# Read sddp first-stage values
function read_first_stage_sddp(results_dir)
    stage_dir = joinpath(results_dir, "simulation_1/stage_1")
    
    # Read z (needs reshaping)
    z_df = CSV.read(joinpath(stage_dir, "z.csv"), DataFrame)
    z_sddp = reshape(z_df.value , (4, 3))

    # Read each variable's average values
    stored_df = CSV.read(joinpath(stage_dir, "stored.csv"), DataFrame)
    stored_sddp = stored_df.value 
    
    spill_df = CSV.read(joinpath(stage_dir, "spill.csv"), DataFrame)
    spill_sddp = spill_df.value 
    
    hydro_df = CSV.read(joinpath(stage_dir, "hydro.csv"), DataFrame)
    hydro_sddp = hydro_df.value 
    
    # Read deficit (needs reshaping)
    deficit_df = CSV.read(joinpath(stage_dir, "deficit.csv"), DataFrame)
    deficit_sddp = reshape(deficit_df.value , (4, 4))
    
    # Read exchange (needs reshaping)
    exchange_df = CSV.read(joinpath(stage_dir, "exchange.csv"), DataFrame)
    exchange_sddp = reshape(exchange_df.value , (5, 5))
    
    # Read thermal generation (needs special handling due to varying units)
    thermal_df = CSV.read(joinpath(stage_dir, "thermal_gen.csv"), DataFrame)
    thermal_sddp = Dict{Int, Vector{Float64}}()
    for p in 1:4
        plant_data = filter(row -> row.plant == p, thermal_df)
        thermal_sddp[p] = plant_data.value 
    end
    
    return Dict(
        :stored => stored_sddp,
        :spill => spill_sddp,
        :hydro => hydro_sddp,
        :deficit => deficit_sddp,
        :exchange => exchange_sddp,
        :thermal_gen => thermal_sddp,
        :z => z_sddp
    )
end