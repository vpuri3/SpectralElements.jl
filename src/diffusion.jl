#
abstract type Equation end

export DiffusionScheme, Diffusion, sim!
#----------------------------------------------------------------------
# Holds all (user-accessible) functions in the sim
#----------------------------------------------------------------------
struct DiffusionScheme
    cb::Function
    setIC::Function
    setBC::Function
    setForcing::Function
    setVisc::Function
    simulate!::Function
    evolve!::Function
    makeRHS!::Function
    opLHS::Function
    opPC::Function
    solve!::Function
end
#----------------------------------------------------------------------
# Diffusion object modified in the sim
#----------------------------------------------------------------------
mutable struct Diffusion{T,U} <: Equation # {T,U,D,K} # Type, dimension, k (bdfK order)
    fld ::Field{T}

    ν  ::Array{T} # viscosity
    f  ::Array{T} # forcing
    rhs::Array{T} # RHS

    tstep::TimeStepper{T,U}

    mshRef::Ref{Mesh{T}} # underlying mesh

    sch::DiffusionScheme
end
#----------------------------------------------------------------------
# Front-end
#----------------------------------------------------------------------
function DiffusionScheme(setIC=fixU!,setBC=fixU!,setForcing=fixU!,setVisc=fixU!;
                        cb=(dfn)->nothing,
                        simulate! =simulate!,
                        evolve! =evolve!,
                        makeRHS! =makeRHS!,
                        opLHS=opLHS,
                        opPC=opPrecond,
                        solve! =solve!)

    return DiffusionScheme(cb,setIC,setBC,setForcing,setVisc,
            simulate!,evolve!,
            makeRHS!,opLHS,opPC,solve!)
end

function Diffusion(bc::Array{Char,1},msh::Mesh,sch::DiffusionScheme
                  ;Ti=0.,Tf=0.,dt=0.,k=3)

    fld = Field(bc,msh)
    ν   = zero(fld.u)
    f   = zero(fld.u)
    rhs = zero(fld.u)

    tstep = TimeStepper(Ti,Tf,dt,k)

    return Diffusion(fld,ν,f,rhs,tstep,Ref(msh),sch)
end

sim!(dfn::Diffusion) = dfn.sch.simulate!(dfn)
#----------------------------------------------------------------------
# Default solver functions
#----------------------------------------------------------------------
function simulate!(dfn::Diffusion)
    @unpack fld, mshRef, sch = dfn
    @unpack time, istep, dt, Tf = dfn.tstep

    u = sch.setIC(fld.u,mshRef[].x,mshRef[].y,time[1])
    @pack! dfn.fld = u

    sch.cb(dfn)
    while time[1] <= Tf[1]

        sch.evolve!(dfn)

        sch.cb(dfn)

        if(time[1] < 1e-12) break end

    end

    return
end
#----------------------------------------------------------------------
function evolve!(dfn::Diffusion)
    @unpack fld, f, ν, mshRef, sch = dfn
    @unpack time, bdfA, bdfB, istep, dt = dfn.tstep

    updateHist!(fld)

    Zygote.ignore() do 
        updateHist!(time)
        istep  .+= 1
        time[1] += dt[1]
        bdfExtK!(bdfA,bdfB,time) 
    end

    ub = sch.setBC(fld.ub,mshRef[].x,mshRef[].y,time[1])
    f = sch.setForcing(f,mshRef[].x,mshRef[].y,time[1])
    ν = sch.setVisc(ν   ,mshRef[].x,mshRef[].y,time[1])

    @pack! dfn.fld = ub
    @pack! dfn = f, ν

    sch.makeRHS!(dfn)
    sch.solve!(dfn)

    return
end
#----------------------------------------------------------------------
# Solver steps

function opLHS(dfn::Diffusion)
    @unpack ν, mshRef = dfn
    @unpack bdfB = dfn.tstep

    opL(u,ν,bdfB,mshRef) = hlmz(u,ν,bdfB[1],mshRef[])

    return opL, (ν,bdfB,mshRef) # returns lhs operator and arguments to opL
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
    @unpack rhs,mshRef,fld,sch = dfn
    @unpack u,ub = fld

    opP(u) = opPrecond(u,dfn)

    opL, largs = sch.opLHS(dfn)
    u = pcgdiff(dfn,opL,rhs,largs...;opM=opP,mult=mshRef[].mult,ifv=false)

    u = u + ub
    @pack! dfn.fld = u
    return
end

#----------------------------------------------------------------------
# Differentiable solver
#----------------------------------------------------------------------

function GSM(u,msh,M)
    u = gatherScatter(u,msh)
    u = mask(u,M)
end

function pcgdiff(dfn,oplhs,rhs,largs...;kwargs...)
    opLGSM(u,largs...) = begin u=oplhs(u,largs...); u=GSM(u,dfn.mshRef[],dfn.fld.M);end
    opL(u) = opLGSM(u,largs...)
    pcg(rhs,opL;kwargs...)
end

Zygote.@adjoint function pcgdiff(dfn,oplhs,rhs,largs...;kwargs...)
    opLGSM(u,largs...) = begin u=oplhs(u,largs...); u=GSM(u,dfn.mshRef[],dfn.fld.M);end
    opL(u) = opLGSM(u,largs...)
    u = pcg(rhs,opL;kwargs...)
    function fun(u̅)
        g1(u) = rhs .- oplhs(u,largs...)
        _,dgdu = pullback(g1,u)
        function opT(λ)
            out = -dgdu(λ)[1]
            out = GSM(out,dfn.mshRef[],dfn.fld.M)
        end
        λ = pcg(u̅,opT;kwargs...)

        g2(rhs,largs...) = rhs .- opLGSM(u,largs...)
        _,dgdp = pullback(g2,rhs,largs...)

        return (nothing,nothing,dgdp(λ)...)
    end
    return u,fun
end