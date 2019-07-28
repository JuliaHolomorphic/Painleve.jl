using Painleve, Test

@test painleve2_6ray((-im,0,im),0.0) ≈ 0.36706155154807807 # RHPackage
@test painleve2_6ray((1,2,1/3),0.0) ≈ -0.5006840381943177 + 0.11748452477363962im # RHPackage

@test pl2def_no_s2_pos_x((-im,0,im),1.0;n=400) ≈ painleve2_6ray((-im,0,im),1.0) ≈ 0.13564354350447155
@test pl2def_no_s2_neg_x1((-im,0,im),-5.0) ≈ painleve2_6ray((-im,0,im),-5.0) ≈ 1.5794870908867027
@test pl2def_no_s2_neg_x2((-2im,0,2im),-7.0) ≈ painleve2_6ray((-2im,0,2im),-7.0) ≈ 4.8192987279585155
@test pl2def_no_s2_neg_x3((-2im,0,2im),-7.0) ≈ painleve2_6ray((-2im,0,2im),-7.0) ≈ 4.8192987279585155
