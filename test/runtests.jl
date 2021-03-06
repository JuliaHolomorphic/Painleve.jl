using Painleve, SingularIntegralEquations, RiemannHilbert, ApproxFun, Test
import RiemannHilbert: RiemannDual, fpstieltjesmatrix!, orientedrightendpoint, Directed
import ApproxFun: mobius

@test painleve2_6ray((-im,0,im),0.0) ≈ 0.36706155154807807 # RHPackage
@test painleve2_6ray((1,2,1/3),0.0) ≈ -0.5006840381943177 + 0.11748452477363962im # RHPackage

@test pl2def_no_s2_pos_x((-im,0,im),1.0;n=400) ≈ painleve2_6ray((-im,0,im),1.0) ≈ 0.13564354350447155
@test pl2def_no_s2_neg_x1((-im,0,im),-5.0) ≈ painleve2_6ray((-im,0,im),-5.0) ≈ 1.5794870908867027

@test pl2def_no_s2_neg_x2((-0.5im,0,0.5im),-7.0) ≈ painleve2_6ray((-0.5im,0,0.5im),-7.0)


@test pl2def_no_s2_neg_x2((-2im,0,2im),-7.0) ≈ painleve2_6ray((-2im,0,2im),-7.0) ≈ 4.819318173454774 # RHPackage
@test pl2def_no_s2_neg_x3((-2im,0,2im),-7.0) ≈ painleve2_6ray((-2im,0,2im),-7.0) ≈ 4.819318173454774

@testset "small x error" begin
    s1,s2,s3 = -2im,0,2im
    x = -7.0
    z_0 = sqrt(-x)/2
    
    ΓD  = Segment(-z_0, z_0)  
    Γ1  = Segment(z_0, z_0 + 2.5exp(im*π/4))
    ΓUi = Segment(z_0, z_0 + 2.5exp(im*3π/4))
    ΓU  = Segment(-z_0, -z_0 + 2.5exp(im*π/4))
    Γ3  = Segment(-z_0, -z_0 + 2.5exp(im*3π/4))
    Γ4  = Segment(-z_0, -z_0 + 2.5exp(-im*3π/4))
    ΓL  = Segment(-z_0, -z_0 + 2.5exp(-im*π/4))
    ΓLi = Segment(z_0, z_0 + 2.5exp(-im*3π/4)) 
    Γ6  = Segment(z_0, z_0 + 2.5exp(-im*π/4)) 
    Γ = ΓD ∪ Γ1 ∪ ΓUi ∪ ΓU ∪ Γ3 ∪ Γ4 ∪ ΓL ∪ ΓLi ∪ Γ6
    
    Θ(z) = 8/3*z^3+2*x*z
    
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

    γ = components(Γ)              
    fpstieltjesmatrix!(Array{ComplexF64}(undef,n,n),Legendre(γ[2]),γ[1])
    sp = Legendre(γ[2])
    z = Directed{false}(orientedrightendpoint(domain(sp)))
    z = mobius(sp,z)
    z̃ = Directed{false}(RiemannDual(1,-0.8))
    @test stieltjesmoment(sp,0,z̃) == stieltjesmoment(sp,0,z)
    n = 50
    for k=1:length(γ), j=1:length(γ)
        fpstieltjesmatrix!(Array{ComplexF64}(undef,n,n),Legendre(γ[k]),γ[j])
    end

    Φ = transpose(rhsolve(transpose(G), n));
end


@test_broken pl2def_no_s2_neg_x1((-im,0,im),-5.0) ≈ painleve2_6ray((-im,0,im),-5.0) ≈ 1.5794870908867027

a = 0.122
x = Fun()
f = Fun(x -> (1-x)^a * exp(x), JacobiWeight(0,a,Jacobi(0,a)))
@test cauchy(f, 1+im) ≈ sum(f/(x-(1+im)))/(2π*im)


z = RiemannDual(1,1)
x = Fun()
f = exp(x)
cauchy(f, z)
x1 = Fun(-1..0)
x2 = Fun(0..1)
f1 = exp(x1)
f2 = exp(x2)

cauchy(f,0.0⁻)
cauchy(f,0.0-eps()im)

cauchy(f,1+im)
cauchy(f1,1+im)+cauchy(f2,1+im)

cauchy(f1,0.0⁻)+cauchy(f2,0.0⁻)
z = RiemannDual(0,-im)
cauchy(f1,z)+cauchy(f2,z)

a = 0.122
f = Fun(x -> (1-x)^a * exp(x), JacobiWeight(0,a,Jacobi(0,a)))
cauchy(f,RiemannDual(-1,-1))

cauchy(f,RiemannDual(1,1))




@testset "x3 in terms of x2" begin
    s1,s2,s3 = 2im,0,-2im
    n = 450
    x = -7.0
    l = 0.7
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

    Φ = transpose(rhsolve(transpose(G), n))
    
    
    R = 0.1; l = 0.6
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
    C13 = Segment(0.5 + R*exp(-im*3π/4),0.5 + R*exp(im*3π/4))
    C15 = Segment(0.5 + R*exp(3im*π/4), 0.5 + R*exp(im*π/4))
    
    C21 = Segment(-0.5 + R*exp(im*-3π/4), -0.5 + R*exp(im*-5π/4))
    C22 = Segment(-0.5 + R*exp(im*3π/4), -0.5 + R*exp(im*π/4))
    C23 = Segment(-0.5 + R*exp(im*π/4), -0.5 + R*exp(-im*π/4))
    C25 = Segment(-0.5 + R*exp(-im*π/4), -0.5 + R*exp(-im*3π/4))

    Γ = Γ1 ∪ ΓUi ∪ ΓU ∪ Γ3 ∪ Γ4 ∪ ΓL ∪ ΓLi ∪ Γ6 ∪ 
        C11 ∪ C12 ∪ C13 ∪ C15 ∪ C21 ∪ C22 ∪ C23 ∪ C25

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
              elseif z in component(Γ, 11) && imag(z) ≤ 0    Pi(z)
              elseif z in component(Γ, 11) && imag(z) > 0    D(z)Pi(z)
              elseif z in component(Γ, 12)    Li(z)S6(z)S1(z)Pi(z)
            
              elseif z in component(Γ, 13)    Li(z)S4i(z)Pi(z)
              elseif z in component(Γ, 14)    Li(z)S4i(z)S3i(z)Pi(z)
              elseif z in component(Γ, 15) && imag(z) > 0    D(z)Pi(z)
              elseif z in component(Γ, 15) && imag(z) ≤ 0    Pi(z)
              elseif z in component(Γ, 16)    Li(z)Pi(z)
              else error("not defined at $z")
              end, Γ);

    n = ncomponents(Γ) * 100
    Φ = transpose(rhsolve(transpose(G), n))
    z = Fun(ℂ)
    2(z*Φ[1,2])(Inf)*sqrt(-x)
end