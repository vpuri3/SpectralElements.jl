#
#----------------------------------------------------------------------
export bdfExtK, bdfExtK!
#----------------------------------------------------------------------
""" k'th Order Backward Difference Formula with Extrapolation for integrating

 du = f(t) + g(t)           (1)
 dt

 where f(t) is rapidly changing, and g(t) is considerably slower.
 For example, we treat diffusion implicity, and convection explicityly
 while time-stepping Navier-Stokes as diffusion timescales are an order
 of magnitude faster (pressure is also treated implicitly but that is
 another story) This allows for larger time-steps based on the intertial term.

 Implementation:

 input: t: array of timesteps [t(i+1), t(i), ..., t(i-k+1)] where
           k is the order of the extrapolation

 output: a: extrapolation coefficients for f(t) (k)
         b: interpolation coefficients for du/dt (k+1)

 (1) ⟺   u' - f(t) = g(t)

 where du/dt(t=i+1) = u(i+1)b(1) + ... + u(i-k+1)*b(k+1),
           f(t=i+1) = f(i+1)
           g(t=i+1) = g(i)a(1) + ... + g(i-k+1)*a(k)

"""
function bdfExtK(t;k=3)
    t  = unique(t)
    kk = length(t) - 1
    t1 = t[1]
    t0 = t[2:end]

    a = interpMat(t1,t0)
    b = derivMat(t)[1,:]'

    if(kk < k)
        global a = hcat(a,zeros(1,k-kk))
        global b = hcat(b,zeros(1,k-kk))
    else
        a = a[1:k]
        b = b[1:k+1]
    end

    if(kk==0) # steady state
        global a[1] = 1.0
    end

    return reshape(a,k), reshape(b,k+1)
end
#--------------------------------------#
function bdfExtK!(a::Vector
                 ,b::Vector
                 ,t::Vector)

    k = length(t) - 1
    aa,bb = bdfExtK(t;k=k)

    a .= aa
    b .= bb

    return
end
#----------------------------------------------------------------------
export TimeStepper
#----------------------------------------------------------------------
struct TimeStepper{T,U}
    time::Vector{T}
    bdfA::Vector{T}
    bdfB::Vector{T}

    istep::Array{U,1} # current step
    fstep::Array{U,1} # final step

    dt::Array{T,1} # time step size
    Ti::Array{T,1} # start time
    Tf::Array{T,1} # end time

end
#--------------------------------------#
function TimeStepper(Ti,Tf,dt,k) # add flag for variable dt

    time = [Ti] .* ones(k+1)
    bdfA,bdfB = bdfExtK(time)

    istep = [0]
    fstep = [0] # steady

    if(dt != 0)
        global fstep = [Int64(ceil(Tf/dt))]
    end

    return TimeStepper{Float64,Int64}(time,bdfA,bdfB
                                     ,istep,fstep
                                     ,[dt],[Ti],[Tf])
end
#----------------------------------------------------------------------
