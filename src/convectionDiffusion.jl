#
#----------------------------------------------------------------------
export ConvectionDiffusion
#----------------------------------------------------------------------
struct ConvectionDiffusion{T,U} <: Equation
    name::String

    fld::Field{T}
    vx ::Array{T}
    vy ::Array{T}

    ν  ::Array{T}     # viscosity
    f  ::Array{T}     # forcing
    rhs::Array{T}     # RHS
    exH::Array{Array} # storage history of explicit term (convection)

    tstep::TimeStepper{T,U}

    JrVD::Array{T,2}
    JsVD::Array{T,2}

    mshV::Mesh{T} # velocity/scalar mesh
    mshD::Mesh{T} # dealiasing mesh

    set0!::Function # initial condition
    set∂!::Function # dirichlet boundary condition
    setF!::Function # forcing function
    setν!::Function # viscosity
end
#--------------------------------------#
function ConvectionDiffusion(name::String
                            ,fld::Field,vx::Array,vy::Array
                            ,tstep::TimeStepper
                            ,mshD::Mesh
                            ,set0! =fixU!,set∂! =fixU!
                            ,setF! =fixU!,setν! =fixU!)

    ν   = zero(fld.u)
    f   = zero(fld.u)
    rhs = zero(fld.u)

    k = length(fld.uh)
    exH = Array[zero(fld.u) for i in 1:k]

    mshV = fld.msh
    JrVD = interpMat(mshD.zr,mshV.zr)
    JsVD = interpMat(mshD.zs,mshV.zs)

    return ConvectionDiffusion(name
                              ,fld,vx,vy
                              ,ν,f,rhs,exH
                              ,tstep
                              ,JrVD,JsVD
                              ,mshV,mshD
                              ,set0!,set∂!,setF!,setν!)
end
#--------------------------------------#
#function ConvectionDiffusion(name::String,bc::Array{Char,1}
#                            ,mshV::Mesh,mshD::Mesh
#                            ;Ti=0.,Tf=0.,dt=0.,k=3)
#
#    fld = Field(bc,mshV)
#    vx  = zero(fld.u)
#    vy  = zero(fld.u)
#
#    tstep = TimeStepper(Ti,Tf,dt,k)
#
#    return ConvectionDiffusion(name
#                              ,fld,vx,vy
#                              ,tstep
#                              ,mshD)
#end
#----------------------------------------------------------------------
# solve
#----------------------------------------------------------------------
function opLHS(u::Array,cdn::ConvectionDiffusion)
    @unpack fld, mshV, ν = cdn
    @unpack bdfB = cdn.tstep

    Au = hlmz(u,ν,bdfB[1],mshV)

    Au .= gatherScatter(Au,mshV)
    Au .= mask(Au,fld.M)
    return Au
end
#--------------------------------------#
function opPrecond(u::Array,cdn::ConvectionDiffusion)
    @unpack fld, tstep, mshV = cdn
    Mu = u ./ mshV.B ./ tstep.bdfB[1]
    return Mu
end
#--------------------------------------#
function makeRHS!(cdn::ConvectionDiffusion)
    @unpack fld, rhs, ν, f, mshV, mshD = cdn
    @unpack vx,vy,exH,JrVD,JsVD = cdn
    @unpack bdfA, bdfB = cdn.tstep

    rhs  .=      mass(f     ,mshV) # forcing
    rhs .-= ν .* lapl(fld.ub,mshV) # boundary data

    for i=1:length(fld.uh)
        exH[i] .= -advect(fld.uh[i],vx,vy,mshV,mshD,JrVD,JsVD)
        rhs   .-= bdfB[1+i] .* mass(fld.uh[i],mshV)
        rhs   .+= bdfA[i]   .* exH[i]
    end

    rhs .= mask(rhs,fld.M)
    rhs .= gatherScatter(rhs,mshV)
    return
end
#--------------------------------------#
function solve!(cdn::ConvectionDiffusion)
    @unpack rhs, mshV, fld = cdn
    @unpack u,ub = fld

    opL(u) = opLHS(u,cdn)
    opM(u) = opPrecond(u,cdn)

    pcg!(u,rhs,opL;opM=opM,mult=mshV.mult,ifv=false)
    u .+= ub
    return
end
#----------------------------------------------------------------------
# time evolution
#----------------------------------------------------------------------
function updateHist!(cdn::ConvectionDiffusion)

    updateHist!(cdn.fld)

    return
end
#--------------------------------------#
function evolve!(cdn::ConvectionDiffusion)

    @unpack fld, f, ν, mshV, tstep = cdn
    @unpack time = tstep

    cdn.set∂!(fld.ub,mshV.x,mshV.y,time[1])
    cdn.setF!(f     ,mshV.x,mshV.y,time[1])
    cdn.setν!(ν     ,mshV.x,mshV.y,time[1])

    makeRHS!(cdn)
    solve!(cdn)

    return
end
#--------------------------------------#
export step!
#--------------------------------------#
function step!(cdn::ConvectionDiffusion)

    updateHist!(cdn)
    updateHist!(cdn.tstep)
    evolve!(cdn)

    return
end
#--------------------------------------#
export simulate!
#--------------------------------------#
function simulate!(cdn::ConvectionDiffusion,callback!::Function)

    @unpack fld, mshV, tstep = cdn
    @unpack time, istep, dt, Tf = tstep

    cdn.set0!(fld.u,mshV.x,mshV.y,time[1])

    callback!(cdn)
    while time[1] <= Tf[1]

        step!(cdn)
        callback!(cdn)

        if(time[1] < 1e-12) break end

    end

    return
end
#----------------------------------------------------------------------
#
