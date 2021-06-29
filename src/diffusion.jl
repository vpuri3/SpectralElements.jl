#
#----------------------------------------------------------------------
export Diffusion
#----------------------------------------------------------------------

struct Diffusion{T,N}
    fld ::Field{T,N}

    time::Vector{T}
    bdfA::Vector{T}
    bdfB::Vector{T}

    ν  ::Array{T,N} # viscosity
    f  ::Array{T,N} # forcing
    rhs::Array{T,N} # RHS

    istep::Array{Int64,1} # time step
    dt   ::Array{T,1}
    Tend ::Array{T,1}     # end time

    mshRef::Ref{Mesh{T,N}} # underlying mesh
end

function Diffusion(fld::Field)
    time = zeros(4)
    bdfA,bdfB = bdfExtK(time)

    ν    = zero(fld.u)
    f    = zero(fld.u)
    rhs  = zero(fld.u)

    istep = [0]
    dt    = zeros(1)
    Tend  = zeros(1)

    return Diffusion(fld,time,bdfA,bdfB
                    ,ν,f,rhs
                    ,istep,dt,Tend
                    ,fld.mshRef)
end

function Diffusion(bc::Array{Char,1},msh::Mesh{T,N}) where {T,N}
    return Diffusion(Field(bc,msh))
end
#----------------------------------------------------------------------
function opLHS(u,dfn::Diffusion)
    @unpack fld,mshRef, ν,bdfB = dfn

    lhs = hlmz(u,ν,bdfB[1],mshRef[])

    lhs  .= gatherScatter(lhs,mshRef[])
    lhs  .= mask(lhs,fld.M)
    return lhs
end

function opPrecond(u,dfn::Diffusion)
    return u
end

function makeRHS!(dfn::Diffusion)
    @unpack fld,rhs,ν,f,bdfA,bdfB,mshRef = dfn

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

    pcg!(u,rhs,opL;opM=opP,mult=mshRef[].mult,ifv=true)
    u .= u + ub
    return
end

#----------------------------------------------------------------------
export evolve!
#----------------------------------------------------------------------
function evolve!(dfn::Diffusion,setIC!::Function,setBC!::Function
                ,setForcing!::Function,setVisc!::Function
                ,setDT!::Function,callback!::Function)

    @unpack fld, f, ν, mshRef, time, bdfA, bdfB, istep, dt, Tend = dfn

    setIC!(fld.u,mshRef[].x,mshRef[].y,time[1])

    while time[1] <= Tend[1]
        updateHist!(fld)
        updateHist!(time)

        istep .+= 1
        setDT!(dt)
        dfn.time[1] += dt[1]
        bdfExtK!(bdfA,bdfB,time)

        setBC!(fld.ub,mshRef[].x,mshRef[].y,time[1])
        setForcing!(f,mshRef[].x,mshRef[].y,time[1])
        setVisc!(ν   ,mshRef[].x,mshRef[].y,time[1])

        makeRHS!(dfn)
        solve!(dfn)

        callback!(dfn)

    end

    return
end
#----------------------------------------------------------------------
