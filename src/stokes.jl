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

    mshVRef::Ref{Mesh{T}} # velocity mesh
    mshPRef::Ref{Mesh{T}} # pressure mesh
    mshDRef::Ref{Mesh{T}} # dealias  mesh
end
#--------------------------------------#
function Stokes(bcVX::Array{Char,1},bcVY::Array{Char,1}
               ,mshV::Mesh,mshD::Mesh,mshP::Mesh
               ;Ti=0.,Tf=0.,dt=0.,k=3)

    vx = ConvectionDiffusion("vx",bcVX,mshV,mshD,Tf=1.0,dt=5e-3)
    vy = ConvectionDiffusion("vy",bcVY,mshV,mshD,Tf=1.0,dt=5e-3)

    pr = Field(['N','N','N','N'],mshP;k=k-1)

    # maintain same tiemstepper across all variables
    tstep = TimeStepper(Ti,Tf,dt,k)

    JrVD = interpMat(mshV.zr,mshP.zr)
    JsVD = interpMat(mshV.zs,mshP.zs)

    return Stokes(vx,vy
                 ,ν,f,tstep
                 ,JrPV,JsPV
                 ,Ref(mshV),Ref(mshD),Ref(mshP))
end
#----------------------------------------------------------------------
function opLHS(u::Array,sks::Stokes)
    @unpack fld, mshVRef, ν = sks
    @unpack bdfB = sks.tstep

    lhs = hlmz(u,ν,bdfB[1],mshVRef[])

    lhs .= gatherScatter(lhs,mshVRef[])
    lhs .= mask(lhs,fld.M)
    return lhs
end

function opPrecond(u::Array,sks::Stokes)
    @unpack fld, tstep, mshVRef = sks
    Mu = u ./ mshVRef[].B ./ tstep.bdfB[1]
    return Mu
end

function makeRHS!(sks::Stokes)
    @unpack fld, rhs, ν, f, mshVRef, mshDRef = sks
    @unpack vx,vy,exH,JrPV,JsPV = sks
    @unpack bdfA, bdfB = sks.tstep

    rhs .= -diver(ux,uy,mshV,JrPV,JsPV)

    rhs .= mask(rhs,fld.M)
    rhs .= gatherScatter(rhs,mshVRef[])
    return
end

function solve!(sks::Stokes)
    @unpack rhs, mshVRef, fld = sks
    @unpack u,ub = fld

    opL(u) = opLHS(u,sks)
    opM(u) = opPrecond(u,sks)

    pcg!(u,rhs,opL;opM=opM,mult=mshVRef[].mult,ifv=false)
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
export evolve!
#----------------------------------------------------------------------
function evolve!(sks::Stokes
                ,setBC! =fixU!
                ,setForcing! =fixU!
                ,setVisc! =fixU!)

    @unpack vx,vy,pr = sks
    @unpack time, bdfA, bdfB, istep, dt = sks.tstep

    # update velocity field for convection
    vx.vx = vx.fld.u
    vx.vy = vy.fld.u

    vy.vx = vx.fld.u
    vy.vy = vy.fld.u

#   ps.vx = vx.fld.u
#   ps.vy = vy.fld.u

#   evolve!(ps,setBC!,setForcing!,setVisc!)

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

    @unpack fld, mshVRef = sks
    @unpack time, istep, dt, Tf = sks.tstep

    setIC!(fld.u,mshVRef[].x,mshVRef[].y,time[1])

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
