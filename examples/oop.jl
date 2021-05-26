#
using SEM
#-----------------------------------

nr = 8
ns = 8
Ex = 4
Ey = 4

function deform(x,y)
    return x,y
end

ifperiodic = [false,false] # [x,y]

m1 = Mesh(nr,ns,Ex,Ey,deform,ifperiodic)

bc = ['D','D','D','D'] # [xmin,xmax,ymin,ymax]

u = Field( (x,y) -> sin(π*x),bc,m1)

#----------------------------------
nothing
