module Painleve

using LinearAlgebra, RiemannHilbert, ApproxFun

export painleve2_6ray, pl2def_no_s2_pos_x, pl2def_no_s2_neg_x1, pl2def_no_s2_neg_x2, pl2def_no_s2_neg_x3

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

    G = Fun( z ->   if z in component(Γ, 1)   S1(z)
                elseif z in component(Γ, 2)   S2(z)
                elseif z in component(Γ, 3)   S3(z)
                elseif z in component(Γ, 4)   S4(z)
                elseif z in component(Γ, 5)   S5(z)
                elseif z in component(Γ, 6)   S6(z)
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

    G = Fun( z ->   if z in component(Γ, 1)    S1(z)
                elseif z in component(Γ, 2)    S3(z)
                elseif z in component(Γ, 3)    S4(z)
                elseif z in component(Γ, 4)    S6(z)
                end, Γ);

    Φ = transpose(rhsolve(transpose(G), n));
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)
end

function pl2def_no_s2_neg_x1((s1,s2,s3),x; n=500, l=0.7)
    @assert mod(n,5) == 0
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()
    @assert s2 == 0
    @assert x < 0
    @assert l*cos(π/4) < 1/2
    
    Γ6Γ1 = Segment(-0.5, 0.5)  
    Γ1   = Segment(0.5, 0.5 + l*exp(im*π/4))
    Γ3   = Segment(-0.5, -0.5 + l*exp(im*3π/4))
    Γ4   = Segment(-0.5, -0.5 + l*exp(-im*3π/4))
    Γ6   = Segment(0.5, 0.5 + l*exp(-im*π/4))  
    Γ = Γ6Γ1 ∪ Γ1 ∪ Γ3 ∪ Γ4 ∪ Γ6
    
    Θ(z) = (2*z/3)*(sqrt(-x)^3)*(4*z^2-3)
    
    S1(z)   = [1 0; s1*exp(im*Θ(z)) 1]
    S3(z)   = [1 0; s3*exp(im*Θ(z)) 1]
    S4(z)   = [1 -s1*exp(-im*Θ(z)); 0 1]
    S6(z)   = [1 -s3*exp(-im*Θ(z)); 0 1]    
    
    G = Fun( z -> if z in component(Γ, 1)    S6(z)S1(z)
              elseif z in component(Γ, 2)    S1(z)
              elseif z in component(Γ, 3)    S3(z)
              elseif z in component(Γ, 4)    S4(z)
              elseif z in component(Γ, 5)    S6(z)
              end, Γ);

    Φ = transpose(rhsolve(transpose(G), n));
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)*sqrt(-x)
end

function pl2def_no_s2_neg_x2((s1,s2,s3),x; n=450, l=0.7)
    @assert mod(n,9) == 0
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()
    @assert s2 == 0
    @assert s1*s3 != 1
    @assert x < 0
    @assert l*cos(π/4) < 1/2
    
    ΓD  = Segment(-0.5, 0.5)  
    Γ1  = Segment(0.5, 0.5 + l*exp(im*π/4))
    ΓUi = Segment(0.5, 0.5 + l*exp(im*3π/4))
    ΓU  = Segment(-0.5, -0.5 + l*exp(im*π/4))
    Γ3  = Segment(-0.5, -0.5 + l*exp(im*3π/4))
    Γ4  = Segment(-0.5, -0.5 + l*exp(-im*3π/4))
    ΓL  = Segment(-0.5, -0.5 + l*exp(-im*π/4))
    ΓLi = Segment(0.5, 0.5 + l*exp(-im*3π/4)) 
    Γ6  = Segment(0.5, 0.5 + l*exp(-im*π/4)) 
    Γ = ΓD ∪ Γ1 ∪ ΓUi ∪ ΓU ∪ Γ3 ∪ Γ4 ∪ ΓL ∪ ΓLi ∪ Γ6
    
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
    
    G = Fun( z -> if z in component(Γ, 1)    D(z)
              elseif z in component(Γ, 2)    S1(z)
              elseif z in component(Γ, 3)    Ui(z)
              elseif z in component(Γ, 4)    U(z)
              elseif z in component(Γ, 5)    S3(z)
              elseif z in component(Γ, 6)    S4(z)
              elseif z in component(Γ, 7)    L(z)
              elseif z in component(Γ, 8)    Li(z)
              elseif z in component(Γ, 9)    S6(z)
              end, Γ);

    Φ = transpose(rhsolve(transpose(G), n))*sqrt(-x);
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)
end

function pl2def_no_s2_neg_x3((s1,s2,s3),x; n=450, l=0.6, R=0.1)
    @assert mod(n,18) == 0
    @assert abs(s1 - s2 + s3 + s1*s2*s3) ≤ 100eps()
    @assert s2 == 0
    @assert s1*s3 != 1
    @assert x < 0
    @assert (l+R)*cos(π/4) < 0.5
     
    Γ1  = Segment(0.5 + R*exp(im*π/4), 0.5 + R*exp(im*π/4) + l*exp(im*π/4))
    ΓUi = Segment(0.5 + R*exp(im*3π/4), 0.5 + R*exp(im*3π/4) + l*exp(im*3π/4))
    ΓU  = Segment(-0.5 + R*exp(im*π/4), -0.5 + R*exp(im*π/4) + l*exp(im*π/4))
    Γ3  = Segment(-0.5 + R*exp(im*3π/4), -0.5 + R*exp(im*3π/4) + l*exp(im*3π/4))
    Γ4  = Segment(-0.5 + R*exp(-im*3π/4), -0.5 + R*exp(-im*3π/4) + l*exp(-im*3π/4))
    ΓL  = Segment(-0.5 + R*exp(-im*π/4), -0.5 + R*exp(-im*π/4) + l*exp(-im*π/4))
    ΓLi = Segment(0.5 + R*exp(-im*3π/4), 0.5 + R*exp(-im*3π/4) + l*exp(-im*3π/4)) 
    Γ6  = Segment(0.5 + R*exp(-im*π/4), 0.5 + R*exp(-im*π/4) + l*exp(-im*π/4)) 
    
    C11 = Segment(0.5 + R*exp(im*π/4), 0.5 + R*exp(-im*π/4))
    C12 = Segment(0.5 + R*exp(-im*π/4), 0.5 + R*exp(-im*3π/4))
    C13 = Segment(0.5 + R*exp(-im*3π/4), 0.5 + R*exp(-im*π))
    C14 = Segment(0.5 + R*exp(im*π), 0.5 + R*exp(im*3π/4))
    C15 = Segment(0.5 + R*exp(3im*π/4), 0.5 + R*exp(im*π/4))
    
    C21 = Segment(-0.5 + R*exp(im*-3π/4), -0.5 + R*exp(im*-5π/4))
    C22 = Segment(-0.5 + R*exp(im*3π/4), -0.5 + R*exp(im*π/4))
    C23 = Segment(-0.5 + R*exp(im*π/4), -0.5 + R*exp(im*0))
    C24 = Segment(-0.5 + R*exp(im*0), -0.5 + R*exp(-im*π/4))
    C25 = Segment(-0.5 + R*exp(-im*π/4), -0.5 + R*exp(-im*3π/4))

    Γ = Γ1 ∪ ΓUi ∪ ΓU ∪ Γ3 ∪ Γ4 ∪ ΓL ∪ ΓLi ∪ Γ6 ∪ 
        C11 ∪ C12 ∪ C13 ∪ C14 ∪ C15 ∪ C21 ∪ C22 ∪ C23 ∪ C24 ∪ C25
    
    Θ(z) = (2*z/3)*(sqrt(-x)^3)*(4*z^2-3)
    f(z) = ((1+2z)/(2z-1))^(im/(2*π))
    
    D(z)     = [1-s1*s3 0; 0 1/(1-s1*s3)]
    S1(z)    = [1 0; s1*exp(im*Θ(z)) 1]
    Ui(z)    = [1 s3*exp(-im*Θ(z))/(1-s1*s3); 0 1]
    U(z)     = [1 -s3*exp(-im*Θ(z))/(1-s1*s3); 0 1]
    S3(z)    = [1 0; s3*exp(im*Θ(z)) 1]
    S3i(z)    = [1 0; -s3*exp(im*Θ(z)) 1]
    S4(z)    = [1 -s1*exp(-im*Θ(z)); 0 1]
    S4i(z)    = [1 s1*exp(-im*Θ(z)); 0 1]
    L(z)     = [1 0; s1*exp(im*Θ(z))/(1-s1*s3) 1]
    Li(z)    = [1 0; -s1*exp(im*Θ(z))/(1-s1*s3) 1]
    S6(z)    = [1 -s3*exp(-im*Θ(z)); 0 1]
    
    P(z)     = [f(z)^(log(1-s1*s3)) 0; 0 f(z)^(log(1/(1-s1*s3)))]
    Pi(z)    = [f(z)^(log(1/(1-s1*s3))) 0; 0 f(z)^(log(1-s1*s3))]
    
    G = Fun( z -> if z in component(Γ, 1)     P(z)S1(z)Pi(z)
              elseif z in component(Γ, 2)     P(z)Ui(z)Pi(z)
              elseif z in component(Γ, 3)     P(z)U(z)Pi(z)
              elseif z in component(Γ, 4)     P(z)S3(z)Pi(z)
              elseif z in component(Γ, 5)     P(z)S4(z)Pi(z)
              elseif z in component(Γ, 6)     P(z)L(z)Pi(z)
              elseif z in component(Γ, 7)     P(z)Li(z)Pi(z)
              elseif z in component(Γ, 8)     P(z)S6(z)Pi(z)
            
              elseif z in component(Γ, 9)     Li(z)S6(z)Pi(z)
              elseif z in component(Γ, 10)    Li(z)Pi(z)
              elseif z in component(Γ, 11)    Pi(z-im*eps())
              elseif z in component(Γ, 12)    D(z)Pi(z+im*eps())
              elseif z in component(Γ, 13)    Li(z)S6(z)S1(z)Pi(z)
            
              elseif z in component(Γ, 14)    Li(z)S4i(z)Pi(z)
              elseif z in component(Γ, 15)    Li(z)S4i(z)S3i(z)Pi(z)
              elseif z in component(Γ, 16)    D(z)Pi(z+im*eps())
              elseif z in component(Γ, 17)    Pi(z-im*eps())
              elseif z in component(Γ, 18)    Li(z)Pi(z)
            
              end, Γ);

    Φ = transpose(rhsolve(transpose(G), n))
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)*sqrt(-x)
end

end # module
