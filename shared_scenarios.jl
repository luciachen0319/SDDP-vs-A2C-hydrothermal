"""
shared_scenarios.jl — Pre-generate evaluation scenarios for fair SDDP vs A2C comparison.

Each scenario is a T-step inflow trajectory sampled from the PAR(1) model.
Both representations are returned:
  - scenarios_inflow[s][t]  : 4-vector of inflow at step t  (used by A2C replay)
  - scenarios_omega[s][t]   : 8-vector vcat(coef, rhs) for the t-th noise draw,
                               corresponding to SDDP stage t+1  (t = 1..T-1)

The (coef, rhs) parameterisation matches the SDDP constraint:
    inflow[i].out + coef[i] * inflow[i].in == rhs[i]
which is identical to A2C's update:
    inflow_new = rhs - coef .* prev_inflow
"""

using Distributions
using LinearAlgebra
using Random

function generate_eval_scenarios(
    n_scenarios::Int,
    T::Int,
    gamma_mat::Matrix{Float64},
    sigma_mats::Vector{Matrix{Float64}},
    exp_mu_mat::Matrix{Float64},
    inflow_init::Vector{Float64};
    seed::Int = 100
)
    rng = MersenneTwister(seed)

    scenarios_inflow = Vector{Vector{Vector{Float64}}}(undef, n_scenarios)
    scenarios_omega  = Vector{Vector{Vector{Float64}}}(undef, n_scenarios)

    for s in 1:n_scenarios
        inflow_seq = Vector{Vector{Float64}}(undef, T)
        omega_seq  = Vector{Vector{Float64}}(undef, T - 1)

        prev           = copy(inflow_init)
        inflow_seq[1]  = copy(prev)

        for t in 2:T
            month      = mod1(t, 12)
            prev_month = mod1(t - 1, 12)

            noise = exp.(rand(rng, MvNormal(zeros(4), sigma_mats[month])))

            coef = -noise .* gamma_mat[month, :] .*
                   exp_mu_mat[month, :] ./ (exp_mu_mat[prev_month, :] .+ 1e-8)
            rhs  = noise .* (1.0 .- gamma_mat[month, :]) .* exp_mu_mat[month, :]

            # Clamp coef for numerical stability (mirrors sddp_ts_2.jl support_point clamping)
            omega_seq[t - 1] = vcat(clamp.(coef, -1e6, 1e6), round.(rhs, digits=4))

            inflow_new    = clamp.(rhs .- coef .* prev, 0.0, Inf)
            inflow_seq[t] = inflow_new
            prev          = inflow_new
        end

        scenarios_inflow[s] = inflow_seq
        scenarios_omega[s]  = omega_seq
    end

    return scenarios_inflow, scenarios_omega
end