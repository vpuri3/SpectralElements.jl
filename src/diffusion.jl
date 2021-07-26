#
abstract type Equation end

#----------------------------------------------------------------------
export Diffusion
#----------------------------------------------------------------------
mutable struct Diffusion{T,U} <: Equation # {T,U,D,K} # Type, dimension, k (bdfK order)

    fld ::Field{T}

    ν  ::Array{T} # viscosity
    f  ::Array{T} # forcing
    rhs::Array{T} # RHS

    tstep::TimeStepper{T,U}

    mshRef::Ref{Mesh{T}} # underlying mesh
end
#--------------------------------------#
function Diffusion(bc::Array{Char,1},msh::Mesh
                  ;Ti=0.,Tf=0.,dt=0.,k=3)

    fld = Field(bc,msh)
    ν   = zero(fld.u)
    f   = zero(fld.u)
    rhs = zero(fld.u)

    tstep = TimeStepper(Ti,Tf,dt,k)

    return Diffusion(fld
                    ,ν,f,rhs
                    ,tstep
                    ,Ref(msh))
end
#----------------------------------------------------------------------
function opLHS(u::Array,ν,bdfB,mshRef,M)
    lhs = hlmz(u,ν,bdfB[1],mshRef[])

    lhs = gatherScatter(lhs,mshRef[])
    lhs = mask(lhs,M)
    return lhs
end

function opPrecond(u::Array,dfn::Diffusion)
    return u
end

function makeRHS!(dfn::Diffusion)
    @unpack fld, ν, f, mshRef = dfn
    @unpack bdfA, bdfB = dfn.tstep

    rhs =  mass(f     ,mshRef[]) # forcing
    rhs = rhs .- ν .* lapl(fld.ub,mshRef[]) # boundary data

    for i=1:length(fld.uh)             # histories
        rhs = rhs .- bdfB[1+i] .* mass(fld.uh[i],mshRef[])
    end

    rhs  = mask(rhs,fld.M)
    rhs  = gatherScatter(rhs,mshRef[])
    @pack! dfn = rhs
    return
end

function solve!(dfn::Diffusion)
    @unpack rhs,ν,mshRef,fld = dfn
    @unpack u,ub,M = fld
    bdfB = dfn.tstep.bdfB

    opP(u) = opPrecond(u,dfn)

    u = lsolve(rhs,ν,bdfB,mshRef,M;opM=opP,mult=mshRef[].mult,ifv=false)
    u = u + ub
    @pack! dfn.fld = u
    return
end

function lsolve(rhs,ν,bdfB,mshRef,M;kwargs...)
    opL(u) = opLHS(u,ν,bdfB,mshRef,M)
    pcg(rhs,opL;kwargs...)
end
Zygote.@adjoint function lsolve(rhs,ν,bdfB,mshRef,M;kwarg...)
    opL(u) = opLHS(u,ν,bdfB,mshRef,M)
    u = pcg(rhs,opL;kwarg...)
    function fun(u̅)
        λ = pcg(u̅,opL;kwarg...)
        g(rhs,ν,bdfB,mshRef,M) = rhs .- opLHS(u,ν,bdfB,mshRef,M)
        _,dgdp = pullback(g,rhs,ν,bdfB,mshRef,M)
        return dgdp(λ)
    end
    return u,fun
end
#----------------------------------------------------------------------
export evolve
#----------------------------------------------------------------------
function evolve!(dfn::Diffusion
                ,setBC! =fixU!
                ,setForcing! =fixU!
                ,setVisc! =fixU!)

    @unpack fld, f, ν, mshRef = dfn
    @unpack time, bdfA, bdfB, istep, dt = dfn.tstep

    updateHist!(fld)

    Zygote.ignore() do 
        updateHist!(time)
        istep  .+= 1
        time[1] += dt[1]
        bdfExtK!(bdfA,bdfB,time) 
    end

    ub = setBC!(fld.ub,mshRef[].x,mshRef[].y,time[1])
    f = setForcing!(f,mshRef[].x,mshRef[].y,time[1])
    ν = setVisc!(ν   ,mshRef[].x,mshRef[].y,time[1])

    @pack! dfn.fld = ub
    @pack! dfn = f, ν

    makeRHS!(dfn)
    solve!(dfn)

    return
end
#----------------------------------------------------------------------
export simulate!
#----------------------------------------------------------------------
function simulate!(dfn::Diffusion,callback!::Function
                  ,setIC! =fixU!
                  ,setBC! =fixU!
                  ,setForcing! =fixU!
                  ,setVisc! =fixU!)

    @unpack fld, mshRef = dfn
    @unpack time, istep, dt, Tf = dfn.tstep

    u = setIC!(fld.u,mshRef[].x,mshRef[].y,time[1])
    @pack! dfn.fld = u

    callback!(dfn)
    while time[1] <= Tf[1]

        evolve!(dfn,setBC!,setForcing!,setVisc!)

        callback!(dfn)

        if(time[1] < 1e-12) break end

    end

    return
end
#----------------------------------------------------------------------
#
