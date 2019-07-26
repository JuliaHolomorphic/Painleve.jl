module Painleve

using LinearAlgebra, RiemannHilbert, ApproxFun

export painleve2, painleve2deformed_no_s2

function painleve2((s1,s2,s3),x; n=600)
    @assert mod(n,6) == 0
	@assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()

	Γ = Segment(0, 2.5exp(im*π/6))   ∪
    Segment(0, 2.5exp(im*π/2))       ∪
    Segment(0, 2.5exp(5im*π/6))      ∪
    Segment(0, 2.5exp(-5im*π/6))     ∪
    Segment(0, 2.5exp(-im*π/2))      ∪
    Segment(0, 2.5exp(-im*π/6));

    G = Fun( z -> if angle(z) ≈ π/6
                    [1                             0;
                     s1*exp(8im/3*z^3+2im*x*z)     1]
                elseif angle(z) ≈ π/2
                    [1                 s2*exp(-8im/3*z^3-2im*x*z);
                     0                 1]
                elseif angle(z) ≈ 5π/6
                    [1                             0;
                     s3*exp(8im/3*z^3+2im*x*z)     1]
                elseif angle(z) ≈ -π/6
                    [1                -s3*exp(-8im/3*z^3-2im*x*z);
                     0                 1]
                elseif angle(z) ≈ -π/2
                    [1                             0;
                     -s2*exp(8im/3*z^3+2im*x*z)    1]
                elseif angle(z) ≈ -5π/6
                    [1                -s1*exp(-8im/3*z^3-2im*x*z);
                     0                 1]
                end
                    , Γ);

    Φ = transpose(rhsolve(transpose(G), n));
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)
end

function painleve2deformed_no_s2((s1,s2,s3),x; n=800)
    @assert mod(n,4) == 0
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()

    Γ = Segment((im*x^(1/2))/2, 2.5+(im*x^(1/2))/2)   ∪
        Segment((im*x^(1/2))/2, -2.5+(im*x^(1/2))/2)      ∪
        Segment((-im*x^(1/2))/2, 2.5-(im*x^(1/2))/2)     ∪
        Segment((-im*x^(1/2))/2, -2.5-(im*x^(1/2))/2)

    G = Fun( z -> if angle(z-(im*x^(1/2))/2) ≈ 0
                    [1                             0;
                     s1*exp(8im/3*z^3+2im*x*z)     1]
                elseif angle(z-(im*x^(1/2))/2) ≈ π
                    [1                             0;
                     s3*exp(8im/3*z^3+2im*x*z)     1]
                elseif angle(z+(im*x^(1/2))/2) ≈ 0
                    [1                -s3*exp(-8im/3*z^3-2im*x*z);
                     0                 1]
                elseif angle(z+(im*x^(1/2))/2) ≈ π
                    [1                -s1*exp(-8im/3*z^3-2im*x*z);
                     0                 1]
                end
                    , Γ);

    Φ = transpose(rhsolve(transpose(G), n));
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)
end

end # module
