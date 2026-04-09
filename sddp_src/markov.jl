module markov

using Random
using Statistics
using DataFrames
using CSV
using Clustering
using LinearAlgebra
using Distributions
using SDDP, HiGHS

export Markovian, SA, RSA, SAA, simulate, write, save_sample_paths, load_sample_paths
export create_sddp_markov_graph, solve_markov_chain

"""
    Markovian

A struct representing a Markovian process approximation.

# Fields
- `samples::Array{Float64,3}`: Sample paths
- `T::Int`: Number of time steps
- `dim_Markov_states::Int`: Dimension of state space
- `n_Markov_states::Vector{Int}`: Number of Markov states at each stage
- `f::Function`: Sample path generator function
- `int_flag::Bool`: Whether to round states to integers
- `n_samples::Int`: Number of samples
- `Markov_states::Vector{Any}`: Markov states at each stage
- `transition_matrix::Vector{Any}`: Transition matrices between stages
- `iterate::Union{Nothing, Vector{Array{Float64,2}}}`: Storage for RSA method
"""
mutable struct Markovian
    samples::Array{Float64,3}
    T::Int
    dim_Markov_states::Int
    n_Markov_states::Vector{Int}
    f::Union{Function,Nothing}  # Make f optional
    int_flag::Bool
    n_samples::Int
    Markov_states::Vector{Any}
    transition_matrix::Vector{Any}
    iterate::Union{Nothing, Vector{Array{Float64,2}}}

    # Modified constructor to accept either a generator function or pre-generated samples
    function Markovian(
        samples_or_f::Union{Array{Float64,3},Function}, 
        n_Markov_states::Vector{Int}, 
        n_sample_paths::Int; 
        int_flag::Bool=false
    )
        if isa(samples_or_f, Function)
            rng = MersenneTwister(0)
            samples = samples_or_f(rng, n_sample_paths)
            f = samples_or_f
        else
            samples = samples_or_f
            f = nothing
        end
        
        (n_samples, T, dim_Markov_states) = size(samples)
        Markov_states = Vector{Matrix{Float64}}(undef, T)
        Markov_states[1] = reshape(samples[1,1,:], 1, dim_Markov_states)
        transition_matrix = Vector{Any}()
        
        return new(
            samples, 
            T, 
            dim_Markov_states, 
            n_Markov_states, 
            f, 
            int_flag, 
            n_samples, 
            Markov_states, 
            transition_matrix, 
            nothing
        )
    end
end

"""
    _initialize(self)

Initialize Markov states.
"""
function _initialize(self::Markovian)
    for t in 2:self.T
        n_states = min(self.n_Markov_states[t], self.n_samples)
        self.Markov_states[t] = self.samples[1:n_states, t, :]
        if size(self.Markov_states[t], 1) == 0
            # Handle zero-size Markov states, perhaps by initializing with random samples
            self.Markov_states[t] = self.samples[1, t, :]'  # Transpose to get a row vector
            self.n_Markov_states[t] = 1
        end
    end
end

"""
    _initialize_matrix(self)

Initialize transition matrix.
"""
function _initialize_matrix(self::Markovian)
    # Initialize transition_matrix[1] as a 1x1 Matrix
    self.transition_matrix = [ reshape([1.0], 1, 1) ]
    for t in 2:self.T
        push!(self.transition_matrix, zeros(Float64, self.n_Markov_states[t-1], self.n_Markov_states[t]))
    end
end

"""
    round_to_int(self)

Round Markov states to integer.
"""
function round_to_int(self::Markovian)
    for t in 2:self.T
        self.Markov_states[t] .= round.(Int, self.Markov_states[t])
    end
end

"""
    save_sample_paths(samples::Array{Float64,3}, output_path::String)

Save the generated sample paths to CSV files.
"""
# In markov.jl
function save_sample_paths(samples::Array{Float64,3}, output_path::String, subset_name::String)
    n_samples, T, dim_states = size(samples)
    
    # Create output directory if it doesn't exist
    folder_path = joinpath(output_path, "sample_paths$(T)_$(subset_name)")
    if !isdir(folder_path)
        mkpath(folder_path)
    end
    
    # Save sample paths for each state dimension
    for d in 1:dim_states
        df = DataFrame(
            "Time" => 1:T,
            ["Scenario_$i" => samples[i,:,d] for i in 1:n_samples]...
        )
        
        CSV.write(
            joinpath(folder_path, "sample_paths_state_$d.csv"),
            df
        )
    end
end

"""
    load_sample_paths(input_path::String, n_samples::Int, T::Int, dim_states::Int)

Load saved sample paths from CSV files.
"""
function load_sample_paths(input_path::String, n_samples::Int, T::Int, dim_states::Int)
    samples = zeros(Float64, n_samples, T, dim_states)
    
    for d in 1:dim_states
        df = CSV.read(
            joinpath(input_path, "sample_paths_state_$d.csv"),
            DataFrame
        )
        
        for i in 1:n_samples
            samples[i,:,d] = df[!,"Scenario_$i"]
        end
    end
    
    return samples
end

"""
    SA(self)

Use stochastic approximation to compute the partition.
"""
function SA(self::Markovian)
    _initialize(self)
    for (idx, sample) in enumerate(eachslice(self.samples, dims=1))
        step_size = 1.0 / (idx+1)
        for t in 2:self.T
            sample_t = sample[t, :]
            temp = self.Markov_states[t] .- sample_t'
            dist = sum(temp.^2, dims=2)
            dist = vec(dist)
            idx_min = argmin(dist)
            self.Markov_states[t][idx_min, :] += (sample_t - self.Markov_states[t][idx_min, :]) * step_size
        end
    end
    train_transition_matrix(self)
    return (self.Markov_states, self.transition_matrix)
end

"""
    RSA(self)

Use robust stochastic approximation to compute the partition.
"""
function RSA(self::Markovian)
    _initialize(self)
    self.iterate = [copy(self.Markov_states[t]) for t in 1:self.T]
    step_size = 1.0 / sqrt(self.n_samples)
    for (idx, sample) in enumerate(eachslice(self.samples, dims=1))
        for t in 2:self.T
            sample_t = sample[t, :]
            temp = self.iterate[t] .- sample_t'
            dist = sum(temp.^2, dims=2)
            dist = vec(dist)
            idx_min = argmin(dist)
            self.iterate[t][idx_min, :] += (sample_t - self.iterate[t][idx_min, :]) * step_size
        end
        for t in 2:self.T
            self.Markov_states[t] .+= self.iterate[t]
        end
    end
    for t in 2:self.T
        self.Markov_states[t] ./= self.n_samples
    end
    train_transition_matrix(self)
    return (self.Markov_states, self.transition_matrix)
end

"""
    SAA(self)

Use K-means method to discretize the Markovian process.
"""
function SAA(self::Markovian)
    if self.int_flag == false
        labels = ones(Int, self.n_samples)
    end
    _initialize_matrix(self)
    for t in 2:self.T
        data = self.samples[:, t, :]
        data = data'
        result = kmeans(data, self.n_Markov_states[t]; maxiter=100, display=:none)
        self.Markov_states[t] = result.centers'
        if self.int_flag == false
            labels_new = result.assignments
            counts = zeros(Float64, self.n_Markov_states[t-1])
            for i in 1:self.n_samples
                counts[labels[i]] += 1
                self.transition_matrix[t][labels[i], labels_new[i]] += 1
            end
            for i in 1:self.n_Markov_states[t-1]
                if counts[i] != 0
                    self.transition_matrix[t][i, :] /= counts[i]
                end
            end
            labels = labels_new
        end
    end
    if self.int_flag == true
        train_transition_matrix(self)
    end
    return (self.Markov_states, self.transition_matrix)
end

"""
    train_transition_matrix(self)

Use the generated sample to train the transition matrix by frequency counts.
"""
function train_transition_matrix(self::Markovian)
    if self.int_flag
        round_to_int(self)
    end
    labels = ones(Int, self.n_samples, self.T)
    for t in 2:self.T
        self.Markov_states[t] = unique(self.Markov_states[t], dims=1)
        self.n_Markov_states[t] = size(self.Markov_states[t], 1)
    end
    for t in 2:self.T
        dist = zeros(Float64, self.n_samples, self.n_Markov_states[t])
        for (idx, markov_state) in enumerate(eachrow(self.Markov_states[t]))
            temp = self.samples[:, t, :] .- reshape(markov_state, 1, :)
            dist[:, idx] = sum(temp.^2, dims=2)
        end
        for i in 1:self.n_samples
            if isempty(dist[i, :])
                labels[i, t] = 1  # Default to index 1 or handle as needed
            else
                labels[i, t] = argmin(dist[i, :])
            end
        end
    end
    _initialize_matrix(self)
    # Replace the existing loop with the modified code
    for k in 1:self.n_samples
        for t in 2:self.T
            row_idx = labels[k, t-1]
            col_idx = labels[k, t]
            # Output the indices
            @show k, t, row_idx, col_idx
            # Check if indices are valid
            if row_idx < 1 || col_idx < 1
                println("Error: Invalid index detected at sample $k, time $t: (row_idx=$row_idx, col_idx=$col_idx)")
                # Optionally, you can throw an error or handle it as needed
                # throw(BoundsError("Invalid index in transition matrix"))
            end
            # Proceed with the update
            self.transition_matrix[t][row_idx, col_idx] += 1
        end
    end
    # Continue with the rest of your function...
    for t in 2:self.T
        counts = sum(self.transition_matrix[t], dims=2)
        idx = findall(counts .== 0)
        if !isempty(idx)
            # Remove zero transition states
            self.Markov_states[t-1] = deleteat!(copy(self.Markov_states[t-1]), idx)
            self.n_Markov_states[t-1] -= length(idx)
            self.transition_matrix[t-1] = deleteat!(copy(self.transition_matrix[t-1]), idx, dims=2)
            self.transition_matrix[t] = deleteat!(copy(self.transition_matrix[t]), idx, dims=1)
            counts = deleteat!(counts, idx)
        end
        for i in 1:size(self.transition_matrix[t], 1)
            if counts[i] != 0
                self.transition_matrix[t][i, :] /= counts[i]
            end
        end
    end
end

"""
    write(self, path)

Write Markov states and transition matrix to disk.
"""
function write(self::Markovian, path::String)
    for t in 1:self.T
        # Ensure the output directory exists
        if !isdir(path)
            mkpath(path)
        end
        # Construct DataFrames with :auto to generate column names
        df_states = DataFrame(self.Markov_states[t], :auto)
        CSV.write(joinpath(path, "Markov_states_$t.csv"), df_states; writeheader=true)
        df_transitions = DataFrame(self.transition_matrix[t], :auto)
        CSV.write(joinpath(path, "transition_matrix_$t.csv"), df_transitions; writeheader=true)
    end
end

"""
    simulate(self, n_samples)

Generate a three-dimensional array `(n_samples x T x n_states)` representing `n_samples` number of sample paths.
"""
function simulate(self::Markovian, n_samples::Int)
    sim = zeros(Float64, n_samples, self.T, self.dim_Markov_states)
    for i in 1:n_samples
        state = 1
        rng = MersenneTwister(i)
        for t in 1:self.T
            p = self.transition_matrix[t][state, :]
            state = rand(rng, Categorical(p))
            sim[i, t, :] = self.Markov_states[t][state, :]
        end
    end
    return sim
end


"""
    create_sddp_markov_graph(markovian::Markovian)

Convert Markovian object to SDDP.MarkovianPolicyGraph format.
"""
function create_sddp_markov_graph(markovian::Markovian)
    # Create transition matrices for each stage
    transition_matrices = Vector{Matrix{Float64}}()
    
    # First stage transition (from root)
    push!(transition_matrices, [1.0]')  # Root to first stage
    
    # Add transition matrices for subsequent stages
    for t in 2:markovian.T
        n_prev = size(markovian.Markov_states[t-1], 1)
        n_curr = size(markovian.Markov_states[t], 1)
        push!(transition_matrices, markovian.transition_matrix[t])
    end
    
    # Create the Markovian graph
    graph = SDDP.MarkovianGraph(transition_matrices)
    
    return graph
end

function solve_markov_chain(process_generator, n_markov_states, n_samples; 
                          method=:RSA, int_flag=false)
    # Create Markovian object
    markovian = Markovian(process_generator, n_markov_states, n_samples; 
                         int_flag=int_flag)
    
    # Get Markov chain approximation using specified method
    if method == :SA
        Markov_states, transition_matrix = SA(markovian)
    elseif method == :RSA
        Markov_states, transition_matrix = RSA(markovian)
    else # SAA
        Markov_states, transition_matrix = SAA(markovian)
    end
    
    # Create SDDP policy graph
    graph = create_sddp_markov_graph(markovian)
    
    return graph, markovian
end

"""
    solve_markov_chain(process_generator, n_markov_states, n_samples; 
                      method=:RSA, int_flag=false)

Create and solve Markov chain approximation, return graph for SDDP.
"""
function solve_markov_chain(process_generator, n_markov_states, n_samples; 
                          method=:RSA, int_flag=false)
    # Create Markovian object
    markovian = Markovian(process_generator, n_markov_states, n_samples; 
                         int_flag=int_flag)
    
    # Get Markov chain approximation using specified method
    if method == :SA
        Markov_states, transition_matrix = SA(markovian)
    elseif method == :RSA
        Markov_states, transition_matrix = RSA(markovian)
    else # SAA
        Markov_states, transition_matrix = SAA(markovian)
    end
    
    # Create SDDP policy graph
    graph = create_sddp_markov_graph(markovian)
    
    return graph, markovian
end

end