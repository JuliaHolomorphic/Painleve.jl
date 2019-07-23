using DelimitedFiles

C = Vector{Matrix{Float64}}(undef, 6)
for j = 1:size(C,2)
    C[j] = readdlm("data/u$j.csv",',')
end

Fun(coefficients(C[1][:,1] + im*C[1][:,2], Chebyshev(), Legendre())
