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

 A(v⃗,u⃗) + (v⃗,∇p) = (v⃗,∇p)   ---(1)
 (q,∇⋅u⃗)         = (q,0)    ---(2)

 [ AA -DD']*[u̲⃗] = [f̲⃗]
 [-DD     ] [p̲̲̲]   [0̲]

 where AA * u̲⃗ = [A  ] * [ux],
                [  A]   [uy]

 is the Diffusion operator acting diagonally on X, Y velocity
 components.

 DD * u̲⃗ = DD * [ux]
               [uy]

 is the divergence operator on the pressure mesh.

 To sove the coupled system of equations, we diagonalize the linear system
 with Schur complements as follows.

 (2) -> (2) + DD AA^-1 *(1)

 [AA -DD']*[u̲⃗] = [f̲⃗]
 [    EE ] [p̲̲̲]   [g]

 where 
 EE = -DD AA^-1 * DD'
 g  = -DD AA^-1 * f̲⃗

 is the pressure system
"""
struct Stokes{T,U} <: Equation
    vx::ConvectionDiffusion{T,U}
    vy::ConvectionDiffusion{T,U}

    pr::Field{T}

    JrPV::Array{T,2}
    JsPV::Array{T,2}

    mshVMesh{T} # velocity mesh
    mshPMesh{T} # pressure mesh
    mshDMesh{T} # dealias  mesh
end
#--------------------------------------#
function Stokes(bcVX::Array{Char,1},bcVY::Array{Char,1}
               ,mshV::Mesh,mshD::Mesh,mshP::Mesh
               ;Ti=0.,Tf=0.,dt=0.,k=3)

    velX = Field(bcVX,mshV)
    velY = Field(bcVY,mshV)

    tstep = TimeStepper(Ti,Tf,dt,k)

    vx = ConvectionDiffusion("vx",velX,velX.uh[1],velY.uh[1],tstep,mshD)
    vy = ConvectionDiffusion("vy",velY,velX.uh[1],velY.uh[1],tstep,mshD)

#   ps = [
#         ConvectionDiffusion("ps$i",Field(bcPS,mshV),velX.u,velY.u,mshD
#                            ,Tf=1.0,dt=5e-3)
#        for i in 1:k]

    pr = Field(['N','N','N','N'],mshP;k=k-1)

    JrVD = interpMat(mshV.zr,mshP.zr)
    JsVD = interpMat(mshV.zs,mshP.zs)

    return Stokes(vx,vy
                 ,tstep
                 ,JrPV,JsPV
                 ,mshV,mshP,mshD)
end
#----------------------------------------------------------------------
function opLHS(u::Array,sks::Stokes)
    @unpack fld, mshV, ν = sks
    @unpack bdfB = sks.tstep

    lhs = hlmz(u,ν,bdfB[1],mshV)

    lhs .= gatherScatter(lhs,mshV)
    lhs .= mask(lhs,fld.M)
    return lhs
end

function opPrecond(u::Array,sks::Stokes)
    @unpack fld, tstep, mshV = sks
    Mu = u ./ mshV.B ./ tstep.bdfB[1]
    return Mu
end

function makeRHS!(sks::Stokes)
    @unpack fld, rhs, ν, f, mshV, mshD = sks
    @unpack vx,vy,exH,JrPV,JsPV = sks
    @unpack bdfA, bdfB = sks.tstep

    rhs .= -diver(ux,uy,mshV,JrPV,JsPV)

    rhs .= mask(rhs,fld.M)
    rhs .= gatherScatter(rhs,mshV)
    return
end

function solve!(sks::Stokes)
    @unpack rhs, mshV, fld = sks
    @unpack u,ub = fld

    opL(u) = opLHS(u,sks)
    opM(u) = opPrecond(u,sks)

    pcg!(u,rhs,opL;opM=opM,mult=mshV.mult,ifv=false)
    u .+= ub
    return
end
#----------------------------------------------------------------------
"""
 Project velocity field to divergence
 free subspace 
"""
function pressureProject!(sts::Stokes)

    makeRHS!(sks)

    δp = solve!(sks)

    px,py = diverᵀ(δp,mshV,JrPV,JsPV)

    # Ainv approximate
    px = px .* mshV.Bi
    py = py .* mshV.Bi

    # add correction
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

    # update pressure from history explicity
    pr .= zero(pr)
    for i=1:pr.k
        pr .+= pr.tstep[i]*pr.uh[i]
    end

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
#
#----------------------------------------------------------------------
