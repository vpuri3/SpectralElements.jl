#
#----------------------------------------------------------------------
export Diffusion
#----------------------------------------------------------------------
struct Diffusion{T,U} # {T,N,K} # type, ndim, k (bdfK order)

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

    fld  = Field(bc,msh)
    ν    = zero(fld.u)
    f    = zero(fld.u)
    rhs  = zero(fld.u)

    tstep = TimeStepper(Ti,Tf,dt,k)

    return Diffusion(fld
                    ,ν,f,rhs
                    ,tstep
                    ,Ref(msh))
end
#----------------------------------------------------------------------
function opLHS(u::AbstractArray,dfn::Diffusion)
    @unpack fld, mshRef, ν = dfn
    @unpack bdfB = dfn.tstep

    lhs = hlmz(u,ν,bdfB[1],mshRef[])

    lhs .= gatherScatter(lhs,mshRef[])
    lhs .= mask(lhs,fld.M)
    return lhs
end

function opPrecond(u::AbstractArray,dfn::Diffusion)
    return u
end

function makeRHS!(dfn::Diffusion)
    @unpack fld, rhs, ν, f, mshRef = dfn
    @unpack bdfA, bdfB = dfn.tstep

    rhs  .=            mass(f     ,mshRef[]) # forcing
    rhs .-= ν       .* lapl(fld.ub,mshRef[]) # boundary data
    rhs .-= bdfB[2] .* mass(fld.u1,mshRef[]) # histories
    rhs .-= bdfB[3] .* mass(fld.u2,mshRef[])
    rhs .-= bdfB[4] .* mass(fld.u3,mshRef[])

    rhs  .= mask(rhs,fld.M)
    rhs  .= gatherScatter(rhs,mshRef[])
    return
end

function solve!(dfn::Diffusion)
    @unpack rhs, mshRef, fld = dfn
    @unpack u,ub = fld

    opL(u) = opLHS(u,dfn)
    opP(u) = opPrecond(u,dfn)

    pcg!(u,rhs,opL;opM=opP,mult=mshRef[].mult,ifv=false)
    u .= u + ub
    return
end
#----------------------------------------------------------------------
function fixU!(u::AbstractArray,x,y,t)
    return
end
function fixU!(u::Number)
    return
end
#----------------------------------------------------------------------
export evolve!
#----------------------------------------------------------------------
function evolve!(dfn::Diffusion
                ,setBC! =fixU!
                ,setForcing! =fixU!
                ,setVisc! =fixU!)

    @unpack fld, f, ν, mshRef = dfn
    @unpack time, bdfA, bdfB, istep, dt = dfn.tstep

    updateHist!(fld)
    updateHist!(time)

    istep .+= 1
    time[1] += dt[1]
    bdfExtK!(bdfA,bdfB,time)

    setBC!(fld.ub,mshRef[].x,mshRef[].y,time[1])
    setForcing!(f,mshRef[].x,mshRef[].y,time[1])
    setVisc!(ν   ,mshRef[].x,mshRef[].y,time[1])

    makeRHS!(dfn)
    solve!(dfn)

    return
end
#----------------------------------------------------------------------
export simulate!
#----------------------------------------------------------------------
function simulate!(dfn::Diffusion,setIC!::Function,setBC!::Function
                  ,setForcing!::Function,setVisc!::Function
                  ,callback!::Function)

    @unpack fld, mshRef = dfn
    @unpack time, istep, dt, Tf = dfn.tstep

    setIC!(fld.u,mshRef[].x,mshRef[].y,time[1])

    while time[1] <= Tf[1]

        evolve!(dfn,setBC!,setForcing!,setVisc!)

        callback!(dfn)

        if(time[1] < 1e-12) break end

    end

    return
end
#----------------------------------------------------------------------
#
