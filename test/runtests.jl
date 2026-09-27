using Painleve, Test

@testset "Painlevé II" begin
    @testset "6 rays" begin
        @test painleve2_6ray((-im,0,im),0.0) ≈ 0.36706155154807807 # RHPackage
        @test painleve2_6ray((1,2,1/3),0.0) ≈ -0.5006840381943177 + 0.11748452477363962im # RHPackage
        @test painleve2_6ray((-im,0,im),-5.0) ≈ 1.5794870908867027
        @test painleve2_6ray((-2im,0,2im),-7.0) ≈ 4.819318173454774 # RHPackage
    end

    @testset "deformed, x > 0" begin
        @test pl2def_no_s2_pos_x((-im,0,im),1.0) ≈ painleve2_6ray((-im,0,im),1.0) ≈ 0.13564354350447155
    end

    @testset "deformed, x < 0" begin
        # These currently converge only algebraically in n (≈ 1E-3 accuracy at the default n), which appears to
        # be due to the treatment of junctions in RiemannHilbert.jl
        @test pl2def_no_s2_neg_x1((-im,0,im),-5.0) ≈ 1.5794870908867027 rtol=1E-2
        @test_broken pl2def_no_s2_neg_x1((-im,0,im),-5.0) ≈ 1.5794870908867027
        @test pl2def_no_s2_neg_x2((-0.5im,0,0.5im),-7.0) ≈ painleve2_6ray((-0.5im,0,0.5im),-7.0) rtol=1E-2
        @test_broken pl2def_no_s2_neg_x2((-0.5im,0,0.5im),-7.0) ≈ painleve2_6ray((-0.5im,0,0.5im),-7.0)
        @test_broken pl2def_no_s2_neg_x2((-2im,0,2im),-7.0) ≈ 4.819318173454774
        @test pl2def_no_s2_neg_x3((-2im,0,2im),-7.0) ≈ 4.819318173454774 rtol=1E-3
        @test_broken pl2def_no_s2_neg_x3((-2im,0,2im),-7.0) ≈ 4.819318173454774
    end
end
