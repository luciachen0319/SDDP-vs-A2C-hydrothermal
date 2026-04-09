function read_data_sddp(mode::String = "train")
    # Read data for sddp
    T = 24

    hydro_ = CSV.read("data/hydro.csv", DataFrame)

    demand = Matrix(CSV.read("data/demand.csv", DataFrame))[:,2:end]
    deficit_ = CSV.read("data/deficit.csv", DataFrame)

    exchange_ub = Matrix(CSV.read("data/exchange.csv", DataFrame))[:,2:end]
    exchange_cost = Matrix(CSV.read("data/exchange_cost.csv", DataFrame))[:,2:end]

    # Read thermal data
    thermal = Vector{DataFrame}(undef, 4)
    for i in 0:3
        thermal[i+1] = CSV.read("data/thermal_$i.csv", DataFrame)[:,2:end]
    end

    # Initial conditions
    stored_initial = hydro_.INITIAL[1:4]
    inflow_initial = round.(hydro_.INITIAL[5:8],digits=0)

    # Read Markov states and transition matrices
    Markov_states = Vector{Matrix{Float64}}(undef, T)
    transition_matrix = Vector{Matrix{Float64}}(undef, T)
    for i in 1:T
        Markov_states[i] = round.(Matrix(CSV.read("output/SA10$(T)_$(mode)/Markov_states_$i.csv", DataFrame)),digits=0)#[:,2:end]
        transition_matrix[i] = Matrix(CSV.read("output/SA10$(T)_$(mode)/transition_matrix_$i.csv", DataFrame))#[:,2:end]
    end

    return (
        T = T,
        hydro_ = hydro_,
        demand = demand,
        deficit_ = deficit_,
        exchange_ub = exchange_ub,
        exchange_cost = exchange_cost,
        stored_initial = stored_initial,
        inflow_initial = inflow_initial,
        thermal = thermal,
        Markov_states = Markov_states,
        transition_matrix = transition_matrix
    )
end