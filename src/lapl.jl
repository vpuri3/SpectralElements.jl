#
export lapl
#
# (v,-\del^2 u) = (\grad v, \grad u)
#
#      = v'*(Dx'*B*Dx + Dy'*B*Dy)*u
#
# implemented as
#
#
#
function lapl(u,M,Qx,Qy,Dr,Ds,G11,G12,G22)

	Mu = mask(u,M);
	
	ur = ABu([],Dr,Mu);
	us = ABu(Ds,[],Mu);
	
 	wr = @. G11*ur + G12*us;
 	ws = @. G12*ur + G22*us;
	
	Au = ABu([],Dr',wr) + ABu(Ds',[],ws);
	
	Au = mask(Au,M);
	Au = gatherScatter(Au,Qx,Qy);

	return Au
end
#
#function lapl(u,M,QQtx,QQty,Dr,Ds,G11,G12,G22)
#
#	Mu = mask(u,M);
#	
#	ur = ABu([],Dr,Mu);
#	us = ABu(Ds,[],Mu);
#	
#	wr = G11.*ur + G12.*us;
#	ws = G12.*ur + G22.*us;
#	
#	Au = ABu([],Dr',wr) + ABu(Ds',[],ws);
#	
#	Au = gatherScatter(Au,Qx,Qy);
#	Au = mask(Au,M);
#
#	return Au
#end
