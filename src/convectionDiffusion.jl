#
#----------------------------------------------------------------------
export ConvectionDiffusion
#----------------------------------------------------------------------
struct ConvectionDiffusion{T,U}
    fld::Field{T}
    vx ::Array{T}
    vy ::Array{T}

    ν  ::Array{T} # viscosity
    f  ::Array{T} # forcing
    rhs::Array{T} # RHS
    exH::Array{Array} # storage history of explicit term (convection)

    tstep::TimeStepper{T,U}

    JrVD::Array{T,2}
    JsVD::Array{T,2}

    mshVRef::Ref{Mesh{T}} # underlying mesh
    mshDRef::Ref{Mesh{T}} # dealiasing mesh
end
#--------------------------------------#
function ConvectionDiffusion(bc::Array{Char,1},mshV::Mesh,mshD::Mesh
                            ;Ti=0.,Tf=0.,dt=0.,k=3)

    fld = Field(bc,mshV)
    vx  = zero(fld.u)
    vy  = zero(fld.u)

    ν   = zero(fld.u)
    f   = zero(fld.u)
    rhs = zero(fld.u)
    exH = Array[ zero(fld.u) for i in 1:k]

    tstep = TimeStepper(Ti,Tf,dt,k)

    JrVD = interpMat(mshD.zr,mshV.zr)
    JsVD = interpMat(mshD.zs,mshV.zs)

    return ConvectionDiffusion(fld,vx,vy
                    ,ν,f,rhs,exH
                    ,tstep
                    ,JrVD,JsVD
                    ,Ref(mshV)
                    ,Ref(mshD))
end
#----------------------------------------------------------------------
function opLHS(u::Array,cdn::ConvectionDiffusion)
    @unpack fld, mshVRef, ν = cdn
    @unpack bdfB = cdn.tstep

    lhs = hlmz(u,ν,bdfB[1],mshVRef[])

    lhs .= gatherScatter(lhs,mshVRef[])
    lhs .= mask(lhs,fld.M)
    return lhs
end

function opPrecond(u::Array,cdn::ConvectionDiffusion)
    @unpack tstep, mshVRef = cdn
    Mu = u ./ mshVRef[].B ./ tstep.bdfB[1]
    return Mu
end

function makeRHS!(cdn::ConvectionDiffusion)
    @unpack fld, rhs, ν, f, mshVRef, mshDRef = cdn
    @unpack vx,vy,exH,JrVD,JsVD = cdn
    @unpack bdfA, bdfB = cdn.tstep

    rhs  .=      mass(f     ,mshVRef[]) # forcing
    rhs .-= ν .* lapl(fld.ub,mshVRef[]) # boundary data

    # explicit convection term
    exH[1] .= -advect(fld.uh[1],vx,vy,mshVRef[],mshDRef[],JrVD,JsVD)

    for i=1:length(fld.uh)              # histories
        rhs .-= bdfB[1+i] .* mass(fld.uh[i],mshVRef[])
        rhs .+= bdfA[i]   .* exH[i]
    end

    rhs  .= mask(rhs,fld.M)
    rhs  .= gatherScatter(rhs,mshVRef[])
    return
end

function solve!(cdn::ConvectionDiffusion)
    @unpack rhs, mshVRef, fld = cdn
    @unpack u,ub = fld

    opL(u) = opLHS(u,cdn)
    opM(u) = opPrecond(u,cdn)

    pcg!(u,rhs,opL;opM=opM,mult=mshVRef[].mult,ifv=true)
    u .= u + ub
    return
end
#----------------------------------------------------------------------
export evolve!
#----------------------------------------------------------------------
function evolve!(cdn::ConvectionDiffusion
                ,setBC! =fixU!
                ,setForcing! =fixU!
                ,setVisc! =fixU!)

    @unpack fld, f, ν, mshVRef = cdn
    @unpack time, bdfA, bdfB, istep, dt = cdn.tstep

    updateHist!(fld)
    updateHist!(time)

    istep  .+= 1
    time[1] += dt[1]
    bdfExtK!(bdfA,bdfB,time)

    setBC!(fld.ub,mshVRef[].x,mshVRef[].y,time[1])
    setForcing!(f,mshVRef[].x,mshVRef[].y,time[1])
    setVisc!(ν   ,mshVRef[].x,mshVRef[].y,time[1])

    makeRHS!(cdn)
    solve!(cdn)

    return
end
#----------------------------------------------------------------------
export simulate!
#----------------------------------------------------------------------
function simulate!(cdn::ConvectionDiffusion,callback!::Function
                  ,setIC! =fixU!
                  ,setBC! =fixU!
                  ,setForcing! =fixU!
                  ,setVisc! =fixU!)

    @unpack fld, mshVRef = cdn
    @unpack time, istep, dt, Tf = cdn.tstep

    setIC!(fld.u,mshVRef[].x,mshVRef[].y,time[1])

    callback!(cdn)
    while time[1] <= Tf[1]

        evolve!(cdn,setBC!,setForcing!,setVisc!)

        callback!(cdn)

        if(time[1] < 1e-12) break end

    end

    return
end
#----------------------------------------------------------------------
#
