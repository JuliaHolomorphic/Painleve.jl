module Painleve

using LinearAlgebra, RiemannHilbert, ApproxFun

export painleve2_6ray, pl2def_no_s2_pos_x, pl2def_no_s2_neg_x1, pl2def_no_s2_neg_x2

function painleve2_6ray((s1,s2,s3),x; n=600)
    @assert mod(n,6) == 0
	@assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()

	Γ1 = Segment(0, 2.5exp(im*π/6))
    Γ2 = Segment(0, 2.5exp(im*π/2))       
    Γ3 = Segment(0, 2.5exp(5im*π/6))      
    Γ4 = Segment(0, 2.5exp(-5im*π/6))     
    Γ5 = Segment(0, 2.5exp(-im*π/2))      
    Γ6 = Segment(0, 2.5exp(-im*π/6))
    Γ = Γ1 ∪ Γ2 ∪ Γ3 ∪ Γ4 ∪ Γ5 ∪ Γ6
    
    Θ(z) = 8/3*z^3+2*x*z
    
    S1(z) = [1 0; s1*exp(im*Θ(z)) 1]
    S2(z) = [1 s2*exp(-im*Θ(z)); 0 1]
    S3(z) = [1 0; s3*exp(im*Θ(z)) 1]
    S4(z) = [1 -s1*exp(-im*Θ(z)); 0 1]
    S5(z) = [1 0; -s2*exp(im*Θ(z)) 1]
    S6(z) = [1 -s3*exp(-im*Θ(z)); 0 1]

    G = Fun( z ->   if angle(z) ≈ π/6    S1(z)
                elseif angle(z) ≈ π/2    S2(z)
                elseif angle(z) ≈ 5π/6   S3(z)
                elseif angle(z) ≈ -5π/6  S4(z)
                elseif angle(z) ≈ -π/2   S5(z)
                elseif angle(z) ≈ -π/6   S6(z)
                end, Γ);

    Φ = transpose(rhsolve(transpose(G), n));
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)
end

function pl2def_no_s2_pos_x((s1,s2,s3),x; n=400)
    @assert mod(n,4) == 0
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()
    @assert x > 0
    @assert s2 == 0
    
    z_0 = (im*sqrt(x))/2

    Γ1 = Segment(z_0, z_0 + 2.5)
    Γ3 = Segment(z_0, z_0 - 2.5)       
    Γ4 = Segment(-z_0, -z_0 - 2.5)      
    Γ6 = Segment(-z_0, -z_0 + 2.5)
    Γ = Γ1 ∪ Γ3 ∪ Γ4 ∪ Γ6

    Θ(z) = 8/3*z^3+2*x*z
    
    S1(z) = [1 0; s1*exp(im*Θ(z)) 1]
    S3(z) = [1 0; s3*exp(im*Θ(z)) 1]
    S4(z) = [1 -s1*exp(-im*Θ(z)); 0 1]
    S6(z) = [1 -s3*exp(-im*Θ(z)); 0 1]

    G = Fun( z ->   if angle(z-z_0) ≈ 0       S1(z)
                elseif angle(z-z_0) ≈ π       S3(z)
                elseif abs(angle(z+z_0)) ≈ π  S4(z)
                elseif angle(z+z_0) ≈ 0       S6(z)
                end, Γ);

    Φ = transpose(rhsolve(transpose(G), n));
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)
end

function pl2def_no_s2_neg_x1((s1,s2,s3),x; n=500)
    @assert mod(n,5) == 0
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()
    @assert s2 == 0
    @assert x < 0
    
    z_0 = sqrt(-x)/2

    Γ6Γ1 = Segment(-z_0, z_0)  
    Γ1   = Segment(z_0, z_0 + 2.5exp(im*π/4))
    Γ3   = Segment(-z_0, -z_0 + 2.5exp(im*3π/4))
    Γ4   = Segment(-z_0, -z_0 + 2.5exp(-im*3π/4))
    Γ6   = Segment(z_0, z_0 + 2.5exp(-im*π/4))  
    Γ = Γ6Γ1 ∪ Γ1 ∪ Γ3 ∪ Γ4 ∪ Γ6
    
    Θ(z) = 8/3*z^3+2*x*z
    
    S1S6(z) = [1-s1*s3 -s3*exp(-im*Θ(z)); s1*exp(im*Θ(z)) 1]
    S1(z)   = [1 0; s1*exp(im*Θ(z)) 1]
    S3(z)   = [1 0; s3*exp(im*Θ(z)) 1]
    S4(z)   = [1 -s1*exp(-im*Θ(z)); 0 1]
    S6(z)   = [1 -s3*exp(-im*Θ(z)); 0 1]    
    
    G = Fun( z -> if imag(z) ≈ 0             S1S6(z)
              elseif angle(z-z_0) ≈ π/4      S1(z)
              elseif angle(z+z_0) ≈ 3π/4     S3(z)
              elseif angle(z+z_0) ≈ -3π/4    S4(z)
              elseif angle(z-z_0) ≈ -π/4     S6(z)
              end, Γ);

    Φ = transpose(rhsolve(transpose(G), n));
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)
end

function pl2def_no_s2_neg_x2((s1,s2,s3),x; n=450)
    @assert mod(n,9) == 0
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()
    @assert s2 == 0
    @assert x < 0
    
    z_0 = sqrt(-x)/2
    
    ΓD  = Segment(-z_0, z_0)  
    Γ1  = Segment(z_0, z_0 + 2.5exp(im*π/4))
    ΓLR = Segment(z_0, z_0 + 2.5exp(im*3π/4))
    ΓLL = Segment(-z_0, -z_0 + 2.5exp(im*π/4))
    Γ3  = Segment(-z_0, -z_0 + 2.5exp(im*3π/4))
    Γ4  = Segment(-z_0, -z_0 + 2.5exp(-im*3π/4))
    ΓUR = Segment(-z_0, -z_0 + 2.5exp(-im*π/4))
    ΓUL = Segment(z_0, z_0 + 2.5exp(-im*3π/4)) 
    Γ6  = Segment(z_0, z_0 + 2.5exp(-im*π/4)) 
    Γ = ΓD ∪ Γ1 ∪ ΓLR ∪ ΓLL ∪ Γ3 ∪ Γ4 ∪ ΓUR ∪ ΓUL ∪ Γ6
    
    D(z)  = [1-s1*s3 0; 0 1/(1-s1*s3)]
    S1(z) = [1 0; s1*exp(im*Θ(z)) 1]
    L(z)  = [1 0; s1*exp(im*Θ(z))/(1-s1*s3) 1]
    S3(z) = [1 0; s3*exp(im*Θ(z)) 1]
    S4(z) = [1 -s1*exp(-im*Θ(z)); 0 1]
    U(z)  = [1 -s3*exp(-im*Θ(z))/(1-s1*s3); 0 1]
    S6(z) = [1 -s3*exp(-im*Θ(z)); 0 1]
    
    Θ(z) = 8/3*z^3+2*x*z
    
    G = Fun( z -> if imag(z) ≈ 0             D(z)
              elseif angle(z-z_0) ≈ π/4      S1(z)
              elseif angle(z-z_0) ≈ 3π/4     L(z)
              elseif angle(z+z_0) ≈ π/4      L(z)
              elseif angle(z+z_0) ≈ 3π/4     S3(z)
              elseif angle(z+z_0) ≈ -3π/4    S4(z)
              elseif angle(z+z_0) ≈ -π/4     U(z)
              elseif angle(z-z_0) ≈ -3π/4    U(z)
              elseif angle(z-z_0) ≈ -π/4     S6(z)
              end, Γ);

    Φ = transpose(rhsolve(transpose(G), n));
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)
end

end # module
