# Painleve.jl

[![CI](https://github.com/JuliaHolomorphic/Painleve.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaHolomorphic/Painleve.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JuliaHolomorphic/Painleve.jl/graph/badge.svg)](https://codecov.io/gh/JuliaHolomorphic/Painleve.jl)

A Julia package for computing Painlevé transcendents by numerically solving their Riemann–Hilbert problems
using [RiemannHilbert.jl](https://github.com/JuliaHolomorphic/RiemannHilbert.jl).

For example, the Hastings–McLeod solution of Painlevé II, with Stokes multipliers `(s₁,s₂,s₃) = (-i,0,i)`, at `x = 0`:
```julia
julia> using Painleve

julia> painleve2_6ray((-im,0,im), 0.0)
0.36706155154807807 + 3.533949646070574e-17im
```
