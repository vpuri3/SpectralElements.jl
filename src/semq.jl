#
export semq
#
# SEM global --> local operator
#
function semq(E,n,bc)

Q = spzeros(E*n,E*(n-1)+1);

Id= sparse(I,n,n);
i = 1;
j = 1;
for e=1:E
    Q[i:(i+n-1),j:(j+n-1)] = Id;
    i += n;
    j += n-1;
end

if(bc)
    Q[end,1] = 1;
    Q=Q[:,1:end-1];
end

return Q
end
