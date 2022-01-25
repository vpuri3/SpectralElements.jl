#
"""
 Compute derivative matrix for lagrange
 interpolants on points [x]
"""
function derivMat(x)
    
n = length(x)

a = ones(1,n)
for i=1:n
    for j=1:(i-1) a[i]=a[i]*(x[i]-x[j]) end
    for j=(i+1):n a[i]=a[i]*(x[i]-x[j]) end
end
a = 1 ./ a # Barycentric weights

# diagonal elements
D = x .- x'
for i=1:n D[i,i] = 1. end
D = 1 ./ D
for i=1:n
    D[i,i] = 0.
    D[i,i] = sum(D[i,:])
end

# off-diagonal elements
for j=1:n for i=1:n
    if(i!=j) D[i,j] = a[j] / (a[i]*(x[i]-x[j])) end
end end

return D
end
