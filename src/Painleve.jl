module Painleve

export painleve2

function painleve2(s1,s2,s3,x)
	@assert s1 - s2 + s3 + s1*s2*s3 ≈ 0
	error("Need to implement")
end

end # module
