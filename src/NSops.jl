#
#--------------------------------------#
export gradp
export advect
export diver
export hlmhltz
export pres_op
export pres_proj
export diff_coeffs
#--------------------------------------#

function gradp(p,Qx1,Qy1,Bv,Jrpv,Jspv,Drv,Dsv,rxv,ryv,sxv,syv)

	Jp = ABu(Jspv,Jrpv,p)
	Bp = mass(Jp,[],Bv,[],[],[],[])

	px = ABu([],Drv',rxv .* Bp) + ABu(Dsv',[],sxv .* Bp)
	py = ABu([],Drv',ryv .* Bp) + ABu(Dsv',[],syv .* Bp)

	px = gatherScatter(px,Qx1,Qy1)
	py = gatherScatter(py,Qx1,Qy1)
	return px, py

end


function advect(u,cx,cy,M,Qx,Qy,Bmd,Jr,Js,Dr,Ds,rx,ry,sx,sy)

    uu  = mask(u,M);

    ux,uy = grad(uu,Dr,Ds,rx,ry,sx,sy)

    uxd = ABu(Js,Jr,ux)
    uyd = ABu(Js,Jr,uy)
    cxd = ABu(Js,Jr,cx)
    cyd = ABu(Js,Jr,cy)

    Cud = cxd.*uxd + cyd.*uyd
    Cud = Bmd.*Cud

    Cu  = ABu(tpose(Js),tpose(Jr),Cud)

    Cu = gatherScatter(Cu,Qx,Qy)
    Cu = mask(Cu,M)
    return Cu

end


function diver(ux,uy,Qx2,Qy2,Bv,Jrpv,Jspv,Drv,Dsv,rxv,ryv,sxv,syv)

    uxdx,_ = grad(ux,Drv,Dsv,rxv,ryv,sxv,syv)
	_,uydy = grad(uy,Drv,Dsv,rxv,ryv,sxv,syv)

	Du = ABu(tpose(Jspv),tpose(Jrpv),Bv.*(uxdx+uydy))

	Du = gatherScatter(Du,Qx2,Qy2)
	return Du

end


function hlmhltz(u,visc,b0,M,Jr,Js,Qx,Qy,B,Dr,Ds,rxd,ryd,sxd,syd)

	viscd = ABu(Js,Jr,visc)
	G11 = @. viscd * B * (rxd * rxd + ryd * ryd);
    G12 = @. viscd * B * (rxd * sxd + ryd * syd);
    G22 = @. viscd * B * (sxd * sxd + syd * syd);

	Hu = lapl(u,[],Jr,Js,[],[],Dr,Ds,G11,G12,G22)
	Hu = Hu + b0*mass(u,[],B,Jr,Js,[],[])

	Hu = mass(Hu,M,[],[],[],Qx,Qy)
	return Hu

end


function pres_op(p,Mvx,Mvy,Qx1,Qy1,Qx2,Qy2,Bv,Biv,Jrpv,Jspv,Drv,Dsv,rxv,ryv,sxv,syv)

	px,py = gradp(p,Qx1,Qy1,Bv,Jrpv,Jspv,Drv,Dsv,rxv,ryv,sxv,syv)

	px = mass(px,Mvx,Biv,[],[],Qx1,Qy1)
	py = mass(py,Mvy,Biv,[],[],Qx1,Qy1)

	Ep = diver(px,py,Qx2,Qy2,Bv,Jrpv,Jspv,Drv,Dsv,rxv,ryv,sxv,syv)
	return Ep

end


function pres_proj(ux,uy,pr,b0,Mvx,Mvy,Qx1,Qy1,Qx2,Qy2,Bv,Biv,Jrpv,Jspv,Drv,Dsv,rxv,ryv,sxv,syv,opM,mult)

	g = -diver(ux,uy,Qx2,Qy2,Bv,Jrpv,Jspv,Drv,Dsv,rxv,ryv,sxv,syv)

	pOp(v) = pres_op(v,Mvx,Mvy,Qx1,Qy1,Qx2,Qy2,Bv,Biv,Jrpv,Jspv,Drv,Dsv,rxv,ryv,sxv,syv)
	delp = b0 * pcg(g,pOp,opM,mult,false,x0=g)

	px,py = gradp(delp,Qx1,Qy1,Bv,Jrpv,Jspv,Drv,Dsv,rxv,ryv,sxv,syv)

	px = mass(px,Mvx,Biv/b0,[],[],Qx1,Qy1)
	py = mass(py,Mvx,Biv/b0,[],[],Qx1,Qy1)

	vx = ux + px
	vy = uy + py

	pr = pr + delp

	return vx, vy, pr

end

function diff_coeffs(ts,n)
	ts = reshape(ts,1,:)
	A = ts.^(0:length(ts)-1)
	rhs = zeros(length(ts))
	rhs[n+1] = factorial(n)
	return qr(A,Val(true))\rhs
end
