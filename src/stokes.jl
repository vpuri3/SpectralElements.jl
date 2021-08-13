#
#----------------------------------------------------------------------
export Stokes
#----------------------------------------------------------------------
"""
 Steady stokes flow

 -ν∇²u⃗ + ∇p = f⃗
        ∇⋅u⃗ = 0

 variational formulation:

 find u⃗, p such that ∀ v⃗,p

          | trial | test |
 -----------------------------------
 vel space|   u⃗   |  v⃗   | ∈ H¹₀(Ω)ᵈ
 -----------------------------------
 pr  space|   p   |  q   | ∈ H¹₀(Ω)

 H(v⃗,u⃗) + (v⃗,∇p) = (v⃗,∇p)   ---(1)
 (q,∇⋅u⃗)         = (q,0)    ---(2)

 [ HH -DD']*[u̲⃗] = [f̲⃗]
 [-DD     ] [p̲̲̲]   [0̲]

 where HH * u̲⃗ = [H  ] * [ux],
                [  H]   [uy]

 is the Diffusion operator acting diagonally on X, Y velocity
 components.

 DD * u̲⃗ = DD * [ux]
               [uy]

 is the divergence operator on the pressure mesh.

 To sove the coupled system of equations, we diagonalize the linear system
 with Schur complements as follows.

 (2) -> (2) + DD HH^-1 *(1)

 [HH -DD']*[u̲⃗] = [f̲⃗]
 [    EE ] [p̲̲̲]   [g]

 where 
 EE = -DD HH^-1 * DD'
 g  = -DD HH^-1 * f̲⃗

 is the pressure system
"""
struct Stokes{T,U} <: Equation
    vx::ConvectionDiffusion{T,U}
    vy::ConvectionDiffusion{T,U}

    pr::Field{T}

    px ::Array{T} # pressure forcing (on mshV)
    py ::Array{T} # 
    rhs::Array{T} # Stokes rhs

    JrPV::Array{T,2}
    JsPV::Array{T,2}

    mshV::Mesh{T} # velocity mesh
    mshP::Mesh{T} # pressure mesh
    mshD::Mesh{T} # dealias  mesh

    set0!::Function # initial condition
    set∂!::Function # dirichlet boundary condition
    setF!::Function # forcing function
    setν!::Function # viscosity
end
#--------------------------------------#
function Stokes(
                bcVX::Array{Char,1},bcVY::Array{Char,1}#,bcPS::Array{Char,1}
               ,mshV::Mesh,mshD::Mesh,mshP::Mesh
               ;Ti=0.,Tf=0.,dt=0.,k=3)

    # create set functions for vx, vy


    # create convection diffusion objects
    velX = Field(bcVX,mshV)
    velY = Field(bcVY,mshV)

    tstep = TimeStepper(Ti,Tf,dt,k)

    vx = ConvectionDiffusion("vx",velX,velX.uh[1],velY.uh[1],tstep,mshD
                            ,set0!,set∂!,setF!,setν!)
    vy = ConvectionDiffusion("vy",velY,velX.uh[1],velY.uh[1],tstep,mshD,
                            ,set0!,set∂!,setF!,setν!)

#   ps = [
#         ConvectionDiffusion("ps$i",Field(bcPS,mshV),velX.u,velY.u,mshD
#                            ,Tf=1.0,dt=5e-3)
#        for i in 1:k]

    pr = Field(['N','N','N','N'],mshP;k=k-1)

    JrPV = interpMat(mshV.zr,mshP.zr)
    JsPV = interpMat(mshV.zs,mshP.zs)

    return Stokes(vx,vy,pr
                 ,tstep
                 ,JrPV,JsPV
                 ,mshV,mshP,mshD)
end
#----------------------------------------------------------------------
function opStokesLHS(q::Array,sks::Stokes)
    @unpack mshV, mshP, JrPV, JsPV = sks
    @unpack bdfB = sks.tstep

    Mvx = vx.fld.M
    Mvy = vy.fld.M

    Eq = stokesOp(q,mshV,Mvx,Mvy,JrPV,JsPV)
    Eq = gatherScatter(Eq,mshP)

    return Eu
end

function opPrecond(u::Array,sks::Stokes)
    Mu = u
    return Mu
end

function makeStokesRHS!(sks::Stokes)
    @unpack rhs, mshV = sks
    @unpack vx,vy,JrPV,JsPV = sks
    @unpack bdfB = sks.tstep

    fx = approxHlmzInv(vx.f,mshV,bdfB[1]) .+ vx.fld.ub
    fy = approxHlmzInv(vy.f,mshV,bdfB[1]) .+ vy.fld.ub

    rhs  .= diver(fx,fy,mshV,JrPV,JsPV)

#   rhs .= mask(rhs,pr.M) # all Neumann BC
    rhs .= gatherScatter(rhs,mshV)
    return
end

function solveStokes!(sks::Stokes)
    @unpack rhs, mshV, fld = sks
    @unpack u,ub = fld

    opL(u) = opLHS(u,sks)
    opM(u) = opPrecond(u,sks)

    δp = pcg(rhs,opL;opM=opM,mult=mshV.mult,ifv=false)

    return δp
end
#----------------------------------------------------------------------
"""
 Project velocity field to divergence
 free subspace 
"""
function pressureProject!(sts::Stokes)

    makeStokesRHS!(sks)

    δp = solveStokes!(sks)

    px,py = diverᵀ(δp,mshV,JrPV,JsPV)

    # Ainv approximate
    px = approxHlmzInv(px,mshV)
    py = approxHlmzInv(py,mshV)

    # correction
    vx .+= px
    vy .+= py
    pr .+= δp

    return
end
#----------------------------------------------------------------------
function update!(sks::Stokes
                ,setBC! =fixU!
                ,setForcing! =fixU!
                ,setVisc! =fixU!)

#   updateHist!(ps)
    updateHist!(vx)
    updateHist!(vy)
    updateHist!(pr)

    update!(tstep)

    return
end
#----------------------------------------------------------------------
export evolve!
#----------------------------------------------------------------------
function evolve!(sks::Stokes
                ,setBC! =fixU!
                ,setForcing! =fixU!
                ,setVisc! =fixU!)

    @unpack vx,vy,pr, tstep = sks
    @unpack time = tstep

    update!(sks)

    # explicit pressure update
    pr .= zero(pr)
    for i=1:pr.k
        pr .+= pr.tstep[i]*pr.uh[i]
    end

    # pressure forcing
    px,py = diverᵀ(pr,mshV,JrPV,JsPV)

    evolve!(vx,setBC!,setForcing! .+ px,setVisc!)
    evolve!(vy,setBC!,setForcing! .+ py,setVisc!)

    # project velocity to divergence free subspace with pressure
    pressureProject(sks)

    return
end
#----------------------------------------------------------------------
export simulate!
#----------------------------------------------------------------------
function simulate!(sks::Stokes,callback!::Function
                  ,setIC! =fixU!
                  ,setBC! =fixU!
                  ,setForcing! =fixU!
                  ,setVisc! =fixU!)

    @unpack fld, mshV = sks
    @unpack time, istep, dt, Tf = sks.tstep

    setIC!(fld.u,mshV.x,mshV.y,time[1])

    callback!(sks)
    while time[1] <= Tf[1]

        evolve!(sks,setBC!,setForcing!,setVisc!)

        callback!(sks)

        if(time[1] < 1e-12) break end

    end

    return
end
#----------------------------------------------------------------------
#
