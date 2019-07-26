using Painleve
import Painleve: painleve2deformed_no_s2

@test painleve2((-im,0,im),0.0) ≈ 0.36706155154807807 # RHPackage
@test painleve2((1,2,1/3),0.0) ≈ -0.5006840381943177 + 0.11748452477363962im # RHPackage

@test painleve2deformed_no_s2((-im,0,im),1.0;n=400) ≈ painleve2((-im,0,im),1.0) ≈ 0.13564354350447155