module Painleve

using LinearAlgebra, RiemannHilbert, BlockArrays
using RiemannHilbert: rh_sie_solve
using ContinuumArrays: expand, ⊎

export painleve2_6ray, pl2def_no_s2_pos_x, pl2def_no_s2_neg_x1, pl2def_no_s2_neg_x2, pl2def_no_s2_neg_x3

"""
    painleve2_u(jumps, n)

Solves the matrix Riemann–Hilbert problem `Φ₊ = Φ₋*G`, `Φ(∞) = I`, where `jumps` is a collection of
pairs `Γ => G` of a `Segment` and the jump function on it, using `n` coefficients on each segment,
and returns `2 * lim_{z→∞} z*Φ[1,2](z)`.

`rhsolve` solves `Ψ₊ = G*Ψ₋` so we solve with `transpose(G)` and take `Φ = transpose(Ψ)`.
Writing `Ψ = I + 𝒞U`, the limit is `-∫U[2,1](t)dt/(πi)`.
"""
function painleve2_u(jumps, n)
    G = ⊎((expand(permutedims(G(z)) for z in Γ) for (Γ, G) in jumps)...)
    P, c = rh_sie_solve(G, n)[2,1].args
    -sum(sum(P.args[k] * c[Block(k)]) for k in eachindex(P.args)) / (π*im)
end

function painleve2_6ray((s1,s2,s3),x; n=100)
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()

    Θ(z) = 8/3*z^3+2*x*z

    S1(z) = [1 0; s1*exp(im*Θ(z)) 1]
    S2(z) = [1 s2*exp(-im*Θ(z)); 0 1]
    S3(z) = [1 0; s3*exp(im*Θ(z)) 1]
    S4(z) = [1 -s1*exp(-im*Θ(z)); 0 1]
    S5(z) = [1 0; -s2*exp(im*Θ(z)) 1]
    S6(z) = [1 -s3*exp(-im*Θ(z)); 0 1]

    painleve2_u((Segment(0, 2.5exp(im*π/6))   => S1,
                 Segment(0, 2.5exp(im*π/2))   => S2,
                 Segment(0, 2.5exp(5im*π/6))  => S3,
                 Segment(0, 2.5exp(-5im*π/6)) => S4,
                 Segment(0, 2.5exp(-im*π/2))  => S5,
                 Segment(0, 2.5exp(-im*π/6))  => S6), n)
end

function pl2def_no_s2_pos_x((s1,s2,s3),x; n=100)
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()
    @assert x > 0
    @assert s2 == 0

    z_0 = (im*sqrt(x))/2

    Θ(z) = 8/3*z^3+2*x*z

    S1(z) = [1 0; s1*exp(im*Θ(z)) 1]
    S3(z) = [1 0; s3*exp(im*Θ(z)) 1]
    S4(z) = [1 -s1*exp(-im*Θ(z)); 0 1]
    S6(z) = [1 -s3*exp(-im*Θ(z)); 0 1]

    painleve2_u((Segment(z_0, z_0 + 2.5)   => S1,
                 Segment(z_0, z_0 - 2.5)   => S3,
                 Segment(-z_0, -z_0 - 2.5) => S4,
                 Segment(-z_0, -z_0 + 2.5) => S6), n)
end

function pl2def_no_s2_neg_x1((s1,s2,s3),x; n=100, l=0.7)
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()
    @assert s2 == 0
    @assert x < 0
    @assert l*cos(π/4) < 1/2

    Θ(z) = (2*z/3)*(sqrt(-x)^3)*(4*z^2-3)

    S1(z)   = [1 0; s1*exp(im*Θ(z)) 1]
    S3(z)   = [1 0; s3*exp(im*Θ(z)) 1]
    S4(z)   = [1 -s1*exp(-im*Θ(z)); 0 1]
    S6(z)   = [1 -s3*exp(-im*Θ(z)); 0 1]

    painleve2_u((Segment(-0.5, 0.5)                    => z -> S6(z)S1(z),
                 Segment(0.5, 0.5 + l*exp(im*π/4))     => S1,
                 Segment(-0.5, -0.5 + l*exp(im*3π/4))  => S3,
                 Segment(-0.5, -0.5 + l*exp(-im*3π/4)) => S4,
                 Segment(0.5, 0.5 + l*exp(-im*π/4))    => S6), n) * sqrt(-x)
end

function pl2def_no_s2_neg_x2((s1,s2,s3),x; n=50, l=0.7)
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()
    @assert s2 == 0
    @assert s1*s3 != 1
    @assert x < 0
    @assert l*cos(π/4) < 1/2

    Θ(z) = (2*z/3)*(sqrt(-x)^3)*(4*z^2-3)

    D(z)     = [1-s1*s3 0; 0 1/(1-s1*s3)]
    S1(z)    = [1 0; s1*exp(im*Θ(z)) 1]
    Ui(z)    = [1 s3*exp(-im*Θ(z))/(1-s1*s3); 0 1]
    U(z)     = [1 -s3*exp(-im*Θ(z))/(1-s1*s3); 0 1]
    S3(z)    = [1 0; s3*exp(im*Θ(z)) 1]
    S4(z)    = [1 -s1*exp(-im*Θ(z)); 0 1]
    L(z)     = [1 0; s1*exp(im*Θ(z))/(1-s1*s3) 1]
    Li(z)    = [1 0; -s1*exp(im*Θ(z))/(1-s1*s3) 1]
    S6(z)    = [1 -s3*exp(-im*Θ(z)); 0 1]

    painleve2_u((Segment(-0.5, 0.5)                    => D,
                 Segment(0.5, 0.5 + l*exp(im*π/4))     => S1,
                 Segment(0.5, 0.5 + l*exp(im*3π/4))    => Ui,
                 Segment(-0.5, -0.5 + l*exp(im*π/4))   => U,
                 Segment(-0.5, -0.5 + l*exp(im*3π/4))  => S3,
                 Segment(-0.5, -0.5 + l*exp(-im*3π/4)) => S4,
                 Segment(-0.5, -0.5 + l*exp(-im*π/4))  => L,
                 Segment(0.5, 0.5 + l*exp(-im*3π/4))   => Li,
                 Segment(0.5, 0.5 + l*exp(-im*π/4))    => S6), n) * sqrt(-x)
end

function pl2def_no_s2_neg_x3((s1,s2,s3),x; n=100, l=0.6, R=0.1)
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()
    @assert s2 == 0
    @assert s1*s3 != 1
    @assert x < 0
    @assert (l+R)*cos(π/4) < 0.5

    Θ(z) = (2*z/3)*(sqrt(-x)^3)*(4*z^2-3)
    f(z) = ((1+2z)/(2z-1))^(im/(2*π))

    D(z)     = [1-s1*s3 0; 0 1/(1-s1*s3)]
    S1(z)    = [1 0; s1*exp(im*Θ(z)) 1]
    Ui(z)    = [1 s3*exp(-im*Θ(z))/(1-s1*s3); 0 1]
    U(z)     = [1 -s3*exp(-im*Θ(z))/(1-s1*s3); 0 1]
    S3(z)    = [1 0; s3*exp(im*Θ(z)) 1]
    S3i(z)   = [1 0; -s3*exp(im*Θ(z)) 1]
    S4(z)    = [1 -s1*exp(-im*Θ(z)); 0 1]
    S4i(z)   = [1 s1*exp(-im*Θ(z)); 0 1]
    L(z)     = [1 0; s1*exp(im*Θ(z))/(1-s1*s3) 1]
    Li(z)    = [1 0; -s1*exp(im*Θ(z))/(1-s1*s3) 1]
    S6(z)    = [1 -s3*exp(-im*Θ(z)); 0 1]

    P(z)     = [f(z)^(log(1-s1*s3)) 0; 0 f(z)^(log(1/(1-s1*s3)))]
    Pi(z)    = [f(z)^(log(1/(1-s1*s3))) 0; 0 f(z)^(log(1-s1*s3))]

    # rays emanating from the circles of radius R around ±1/2
    ray(c, θ) = Segment(c + R*exp(im*θ), c + (R+l)*exp(im*θ))
    # the jumps on C13 and C23 are discontinuous across the real axis, so they are split there
    a = 0.5 - R*cos(π/4)
    b = -0.5 + R*cos(π/4)

    painleve2_u((ray(0.5, π/4)    => z -> P(z)S1(z)Pi(z),
                 ray(0.5, 3π/4)   => z -> P(z)Ui(z)Pi(z),
                 ray(-0.5, π/4)   => z -> P(z)U(z)Pi(z),
                 ray(-0.5, 3π/4)  => z -> P(z)S3(z)Pi(z),
                 ray(-0.5, -3π/4) => z -> P(z)S4(z)Pi(z),
                 ray(-0.5, -π/4)  => z -> P(z)L(z)Pi(z),
                 ray(0.5, -3π/4)  => z -> P(z)Li(z)Pi(z),
                 ray(0.5, -π/4)   => z -> P(z)S6(z)Pi(z),

                 Segment(0.5 + R*exp(im*π/4), 0.5 + R*exp(-im*π/4))    => z -> Li(z)S6(z)Pi(z),        # C11
                 Segment(0.5 + R*exp(-im*π/4), 0.5 + R*exp(-im*3π/4))  => z -> Li(z)Pi(z),             # C12
                 Segment(0.5 + R*exp(-im*3π/4), a)                     => Pi,                          # C13, lower
                 Segment(a, 0.5 + R*exp(im*3π/4))                      => z -> D(z)Pi(z),              # C13, upper
                 Segment(0.5 + R*exp(3im*π/4), 0.5 + R*exp(im*π/4))    => z -> Li(z)S6(z)S1(z)Pi(z),   # C14

                 Segment(-0.5 + R*exp(-im*3π/4), -0.5 + R*exp(-im*5π/4)) => z -> Li(z)S4i(z)Pi(z),     # C21
                 Segment(-0.5 + R*exp(im*3π/4), -0.5 + R*exp(im*π/4))    => z -> Li(z)S4i(z)S3i(z)Pi(z), # C22
                 Segment(-0.5 + R*exp(im*π/4), b)                        => z -> D(z)Pi(z),            # C23, upper
                 Segment(b, -0.5 + R*exp(-im*π/4))                       => Pi,                        # C23, lower
                 Segment(-0.5 + R*exp(-im*π/4), -0.5 + R*exp(-im*3π/4))  => z -> Li(z)Pi(z)),          # C24
                n) * sqrt(-x)
end

end # module
