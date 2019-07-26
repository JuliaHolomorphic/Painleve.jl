module Painleve

using LinearAlgebra, RiemannHilbert, ApproxFun

export painleve2_6ray, pl2def_no_s2_pos_x, pl2def_no_s2_neg_x1, pl2def_no_s2_neg_x2

function painleve2_6ray((s1,s2,s3),x; n=600)
    @assert mod(n,6) == 0
	@assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()

	Γ = Segment(0, 2.5exp(im*π/6))   ∪
    Segment(0, 2.5exp(im*π/2))       ∪
    Segment(0, 2.5exp(5im*π/6))      ∪
    Segment(0, 2.5exp(-5im*π/6))     ∪
    Segment(0, 2.5exp(-im*π/2))      ∪
    Segment(0, 2.5exp(-im*π/6));
    
    Θ(z) = 8/3*z^3+2*x*z

    G = Fun( z -> if angle(z) ≈ π/6
                    [1                  0;
                     s1*exp(im*Θ(z))    1]
                elseif angle(z) ≈ π/2
                    [1                  s2*exp(-im*Θ(z));
                     0                  1]
                elseif angle(z) ≈ 5π/6
                    [1                  0;
                     s3*exp(im*Θ(z))    1]
                elseif angle(z) ≈ -π/6
                    [1                  -s3*exp(-im*Θ(z));
                     0                  1]
                elseif angle(z) ≈ -π/2
                    [1                  0;
                     -s2*exp(im*Θ(z))   1]
                elseif angle(z) ≈ -5π/6
                    [1                  -s1*exp(-im*Θ(z));
                     0                  1]
                end
                    , Γ);

    Φ = transpose(rhsolve(transpose(G), n));
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)
end

function pl2def_no_s2_pos_x((s1,s2,s3),x; n=400)
    @assert mod(n,4) == 0
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()
    @assert x>0
    @assert s2 == 0
    
    z_0 = (im*sqrt(x))/2

    Γ = Segment(z_0, z_0 + 2.5)      ∪
        Segment(z_0, z_0 - 2.5)      ∪
        Segment(-z_0, -z_0 + 2.5)    ∪
        Segment(-z_0, -z_0 - 2.5)

    Θ(z) = 8/3*z^3+2*x*z
    
    G = Fun( z -> if angle(z-z_0) ≈ 0
                    [1                  0;
                     s1*exp(im*Θ(z))    1]
                elseif angle(z-z_0) ≈ π
                    [1                  0;
                     s3*exp(im*Θ(z))    1]
                elseif angle(z+z_0) ≈ 0
                    [1                  -s3*exp(-im*Θ(z));
                     0                  1]
                elseif abs(angle(z+z_0)) ≈ π
                    [1                  -s1*exp(-im*Θ(z));
                     0                  1]
                end
                    , Γ);

    Φ = transpose(rhsolve(transpose(G), n));
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)
end

function pl2def_no_s2_neg_x1((s1,s2,s3),x; n=50)
    @assert mod(n,5) == 0
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()
    @assert x<0
    @assert s2 == 0
    
    z_0 = sqrt(-x)/2

    Γ = Segment(-z_0, z_0)                      ∪
        Segment(z_0, z_0 + 2.5exp(im*π/4))      ∪
        Segment(z_0, z_0 + 2.5exp(-im*π/4))     ∪
        Segment(-z_0, -z_0 + 2.5exp(im*3π/4))   ∪
        Segment(-z_0, -z_0 + 2.5exp(-im*3π/4))      
    
    Θ(z) = 8/3*z^3+2*x*z
    
    G = Fun( z -> if imag(z) ≈ 0
                    [1-s1*s3            -s3*exp(-im*Θ(z));
                     s1*exp(im*Θ(z))    1]
                elseif angle(z-z_0) ≈ π/4
                    [1                  0;
                     s1*exp(im*Θ(z))    1]
                elseif angle(z-z_0) ≈ -π/4
                    [1                  -s3*exp(-im*Θ(z));
                     0                  1]
                elseif angle(z+z_0) ≈ 3π/4
                    [1                  0;
                     s3*exp(im*Θ(z))   1]
                elseif angle(z+z_0) ≈ -3π/4
                    [1                  -s1*exp(-im*Θ(z));
                     0                  1]
                end
                    , Γ);

    Φ = transpose(rhsolve(transpose(G), n));
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)
end

function pl2def_no_s2_neg_x2((s1,s2,s3),x; n=450)
    @assert mod(n,9) == 0
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()
    @assert x<0
    @assert s2 == 0
    
    z_0 = sqrt(-x)/2

    Γ = Segment(-z_0, z_0)                      ∪
        Segment(z_0, z_0 + 2.5exp(im*π/4))      ∪
        Segment(z_0 + 2.5exp(im*3π/4), z_0)     ∪
        Segment(z_0 + 2.5exp(-im*3π/4), z_0)    ∪
        Segment(z_0, z_0 + 2.5exp(-im*π/4))     ∪
        Segment(-z_0, -z_0 + 2.5exp(im*π/4))    ∪
        Segment(-z_0, -z_0 + 2.5exp(im*3π/4))   ∪
        Segment(-z_0, -z_0 + 2.5exp(-im*3π/4))  ∪
        Segment(-z_0, -z_0 + 2.5exp(-im*π/4))      
    
    Θ(z) = 8/3*z^3+2*x*z
    
    G = Fun( z -> if imag(z) ≈ 0
                    [1-s1*s3            0;
                     0                  1/(1-s1*s3)]
                elseif angle(z-z_0) ≈ π/4
                    [1                  0;
                     s1*exp(im*Θ(z))    1]
                elseif angle(z-z_0) ≈ 3π/4
                    [1                            0;
                     s1*exp(im*Θ(z))/(1-s1*s2)    1]
                elseif angle(z-z_0) ≈ -3π/4
                    [1                  -s3*exp(-im*Θ(z))/(1-s1*s2);
                     0                  1]
                elseif angle(z-z_0) ≈ -π/4
                    [1                  -s3*exp(-im*Θ(z));
                     0                  1]
                elseif angle(z+z_0) ≈ π/4
                     [1                            0;
                     s1*exp(im*Θ(z))/(1-s1*s2)    1]
                elseif angle(z+z_0) ≈ 3π/4
                    [1                 0;
                     s3*exp(im*Θ(z))   1]
                elseif angle(z+z_0) ≈ -3π/4
                    [1                  -s1*exp(-im*Θ(z));
                     0                  1]
                elseif angle(z+z_0) ≈ -π/4
                    [1                  -s3*exp(-im*Θ(z))/(1-s1*s2);
                     0                  1]
                end
                    , Γ);

    Φ = transpose(rhsolve(transpose(G), n));
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)
end

end # module
