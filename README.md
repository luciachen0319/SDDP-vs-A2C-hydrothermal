# ISEN 637 Course Project — Hydrothermal Scheduling: SDDP vs. A2C

## Project Overview

This project compares two optimization approaches for the **Brazilian hydrothermal power dispatch problem** over a 24-month planning horizon:

- **SDDP** (Stochastic Dual Dynamic Programming) — a classical stochastic optimization method that builds a piecewise-linear approximation of the value function.
- **A2C** (Advantage Actor-Critic) — a deep reinforcement learning algorithm that learns a dispatch policy through interaction, warm-started via behavioral cloning on SDDP demonstrations.

The system models **4 interconnected subsystems** of the Brazilian power grid: Southeast (SE), South (S), Northeast (NE), and North (N). At each time step the dispatcher decides hydro generation, thermal dispatch (95 units), energy exchange between subsystems, voluntary deficit, and reservoir spill — while minimizing total operating cost subject to power balance and physical constraints.

Inflows follow a **PAR(1) log-normal model** (periodic autoregressive, monthly seasonality) calibrated from historical data.

---

## Repository Structure

```
.
├── A2C.jl                   # A2C agent: architecture, training, evaluation (main entry point)
├── data.jl                  # Loads and exposes all data constants used by A2C.jl
├── shared_scenarios.jl      # Shared scenario generator — ensures identical evaluation paths for both methods
│
├── sddp_src/
│   ├── sddp_ts_time.jl      # SDDP (time-series noise model) with training-time measurement  ← primary SDDP script
│   ├── sddp_ts.jl           # SDDP (time-series noise model), no timing
│   ├── sddp_mc.jl           # SDDP (Markov chain noise model)
│   ├── sddp_test.jl         # Ad-hoc testing script
│   ├── save_results_sddp.jl # Saves SDDP simulation results to CSV
│   ├── read_data_sddp.jl    # Data reader helper for SDDP scripts
│   ├── read_first_stage_sddp.jl
│   └── markov.jl            # Markov chain discretization utilities
│
├── data/
│   ├── hydro.csv            # Reservoir bounds, generation limits, initial conditions
│   ├── demand.csv           # Monthly demand per subsystem (12 months × 4 regions)
│   ├── thermal_0..3.csv     # Per-unit thermal cost, lower/upper bounds per subsystem
│   ├── deficit.csv          # Deficit levels and penalty costs
│   ├── exchange.csv         # Exchange capacity upper bounds (5×5)
│   ├── exchange_cost.csv    # Exchange cost coefficients (5×5)
│   ├── gamma.csv            # PAR(1) autoregressive coefficients (12×4)
│   ├── exp_mu.csv           # PAR(1) expected inflow levels (12×4)
│   └── sigma_0..11.csv      # Monthly inflow covariance matrices (4×4, one per month)
│
├── output/
│   └── simulations/
│       └── sddp_train/ts/   # SDDP simulation CSVs consumed by A2C behavioral cloning
│
└── a2c_learning_curve.png   # Learning curve saved after A2C training
```

---

## Dependencies

All dependencies are Julia packages. Open the Julia REPL, press `]` to enter Pkg mode, and run:

```julia
add SDDP HiGHS JuMP CSV DataFrames Distributions LinearAlgebra Plots Random Printf Flux Statistics
```

Press `Backspace` to exit Pkg mode.

**Julia version**: 1.9 or later recommended.

---

## How to Run

The two methods must be run **in order**: SDDP first, then A2C (A2C loads SDDP simulation output for warm-starting).

### Step 1 — Train and simulate SDDP

From the `sddp_src/` directory:

```bash
julia sddp_ts_time.jl
```

This will:
1. Build a `SDDP.LinearPolicyGraph` with 24 stages and PAR(1) inflow uncertainty (100 scenario samples per stage).
2. Train via `SDDP.train()` up to 1 000 iterations, stopping early if the lower bound stalls for 20 iterations within 1e-4.
3. Print training time (seconds and minutes).
4. Run 100 forward simulations on shared evaluation scenarios (seed 204).
5. Save per-simulation CSV results to `../output/simulations/sddp_train/ts/`.
6. Print the final lower bound and the true expected simulation cost.

> To run without timing instrumentation use `sddp_ts.jl` instead.
> For the Markov chain variant use `sddp_mc.jl`.

### Step 2 — Train and evaluate A2C

From the project root:

```bash
julia A2C.jl
```

This will:
1. **Phase 0** — Load the 100 SDDP simulation trajectories from `output/simulations/sddp_train/ts/`.
2. **Phase 1** — Behavioral cloning: pre-train the actor (50 epochs, MSE against SDDP actions) and warm-start the critic (50 epochs, MSE against SDDP costs).
3. **Phase 2** — Mixed BC + A2C training over 10 000 episodes with λ annealing (0.5 → 0), transitioning from imitation to pure RL.
4. Save the learning curve to `a2c_learning_curve.png`.
5. Evaluate the trained agent on the same 100 shared scenarios used by SDDP (seed 204) and print the mean and std dev of total cost for a direct comparison.
6. Print a timing summary for each phase.

---

## Key Hyperparameters

| Parameter | Value | Location |
|---|---|---|
| Planning horizon | 24 months | `A2C.jl:30`, `sddp_ts_time.jl:79` |
| SDDP iteration limit | 1 000 | `sddp_ts_time.jl:206` |
| SDDP early-stop | 20 iters, 1e-4 gap | `sddp_ts_time.jl:209` |
| Inflow scenarios per stage | 100 | `sddp_ts_time.jl:165` |
| A2C episodes | 10 000 | `A2C.jl:533` |
| Behavioral cloning epochs | 50 | `A2C.jl:977` |
| Evaluation scenarios | 100 (seed 204) | `A2C.jl:999-1000` |

---

## Output

| File / Directory | Contents |
|---|---|
| `output/simulations/sddp_train/ts/` | Per-simulation CSVs with stage-level costs, storage, generation, exchange |
| `a2c_learning_curve.png` | Episode average cost and 50-episode moving average over A2C training |
