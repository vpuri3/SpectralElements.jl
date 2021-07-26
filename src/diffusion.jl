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

    opLHS::Function # LHS operator
    LHSargs::Function # returns args to opLHS
end
#--------------------------------------#
function Diffusion(bc::Array{Char,1},msh::Mesh
                  ;Ti=0.,Tf=0.,dt=0.,k=3)

    fld = Field(bc,msh)
    ν   = zero(fld.u)
    f   = zero(fld.u)
    rhs = zero(fld.u)

    tstep = TimeStepper(Ti,Tf,dt,k)

    function opLHS(u::Array,args...)
        ν,bdfB,mshRef,M = args
        lhs = hlmz(u,ν,bdfB[1],mshRef[])
    
        # lhs = gatherScatter(lhs,mshRef[])
        # lhs = mask(lhs,M)
        return lhs
    end
    LHSargs(dfn::Diffusion) = dfn.ν, dfn.tstep.bdfB, dfn.mshRef, dfn.fld.M

    return Diffusion(fld,
                    ν,f,rhs,
                    tstep,
                    Ref(msh),
                    opLHS,
                    LHSargs)
end
#----------------------------------------------------------------------

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
    @unpack rhs,mshRef,fld = dfn
    @unpack u,ub = fld

    opP(u) = opPrecond(u,dfn)

    largs = dfn.LHSargs(dfn)
    u = pcgdiff(dfn,dfn.opLHS,rhs,largs...;opM=opP,mult=mshRef[].mult,ifv=false)

    u = u + ub
    @pack! dfn.fld = u
    return
end

function GSM(f,mshRef,M)
    function opLHS(u,largs...)
        lhs = f(u,largs...)
        lhs = gatherScatter(lhs,mshRef[])
        lhs = mask(lhs,M)
    end
end

function pcgdiff(dfn,oplhs,rhs,largs...;kwargs...)
    opLHS = GSM(oplhs,dfn.mshRef,dfn.fld.M)
    opL(u) = opLHS(u,largs...)
    pcg(rhs,opL;kwargs...)
end
Zygote.@adjoint function pcgdiff(dfn,oplhs,rhs,largs...;kwargs...)
    opLHS = GSM(oplhs,dfn.mshRef,dfn.fld.M)
    opL(u) = opLHS(u,largs...)
    u = pcg(rhs,opL;kwargs...)
    function fun(u̅)
        g1(u) = rhs .- oplhs(u,largs...)
        _,dgdu = pullback(g1,u)
        function opT(λ)
            out = -dgdu(λ)[1]
            out = gatherScatter(out,dfn.mshRef[])
            mask(out,dfn.fld.M)
        end
        λ = pcg(u̅,opT;kwargs...)
        g2(rhs,largs...) = rhs .- opLHS(u,largs...)
        _,dgdp = pullback(g2,rhs,largs...)
        # display(opL(λ).-opT(λ))
        # display(opT(λ))
        return (nothing,nothing,dgdp(λ)...)
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
