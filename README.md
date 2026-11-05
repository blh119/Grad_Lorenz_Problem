# Lorenz System Data Assimilation

Graduate research project in numerical methods. It uses **nudging**, a form of continuous data assimilation, to reconstruct the chaotic Lorenz system from partial observations. A "nudged" copy of the system runs alongside the true solution and is pulled toward the observed values. The project then measures how quickly the copy's error shrinks.

## The Lorenz system

$$
\begin{aligned}
\dot{x} &= \sigma (y - x) \\
\dot{y} &= \rho x - y - xz \\
\dot{z} &= xy - \beta z
\end{aligned}
$$

All solutions are computed with a forward Euler scheme written from scratch in NumPy.

## What the scripts do

**Parameter recovery** (`Original Lorenz Problem/`)
Starts with a wrong estimate of the parameter σ (the true value plus 100). It nudges the model toward observations of the true solution and updates the estimate of σ at regular intervals until it converges. The setup and parameter values follow the approach in [unis-ing/lorenz-parameter-learning](https://github.com/unis-ing/lorenz-parameter-learning).

**Recovering an unobserved variable** (`Lorenz_Proceed*.py`)
Observes only *x* and *z*. The nudged system starts its *y* equation at zero and has to reconstruct both *y* and the nonlinear term *g = −xz* from the observed components. The system is solved over a series of time windows, with each window starting where the last one ended.

- `Lorenz_Proceed.py` sweeps nudging strengths (μ = 1000 to 1900) to compare how fast the error converges
- `Lorenz_Proceed_v2.py` and `Lorenz_Proceed_v3.py` are refined versions using σ = 16, β = 4, ρ = 45.92
- `Lorenz_Class.py` packages the solver, the nudged system, and the plotting functions in a reusable `Lorenz` class

## Output

Each run plots the absolute error between the true and nudged solutions over time on a log scale, for each of *x*, *y*, *z*, and *g*. Errors that decay steadily mean the nudged system has locked onto the true dynamics.

## Running

Requires Python 3 with NumPy, pandas, and Matplotlib:

```
pip install numpy pandas matplotlib
python Lorenz_Proceed_v3.py
```

To run the parameter-recovery experiment, run `Lorenz Problem.py` from inside the `Original Lorenz Problem` folder.
