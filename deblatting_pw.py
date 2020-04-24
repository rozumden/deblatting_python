import random
import pdb
import os

import cv2 as cv
import matplotlib.pyplot as plt

import numpy as np
from numpy.fft import fft2, ifft2
import scipy.sparse.linalg
from scipy import sparse

from utils import *
from vis import *
from deblatting import *

def estimateFM_pw(I, B, H, M=None, F=None, F_T=None, M_T=None, state=None, params=None):
	## Estimate F_i,M_i in FMO equation I = \sum_i H_i*F_i + (1 - \sum_i H_i*M_i)B, where * is convolution
	## M is suggested to be specified to know approximate object size, at least as am array of zeros, for speed-up
	## F_T, M_T - template for F_i and M_i
	if params is None:
		params = Params()
	if len(H.shape) == 2: ## not piece-wise
		H = H[:,:,np.newaxis,np.newaxis]
	elif len(H.shape) == 3:
		H = H[:,:,np.newaxis,:]
	ns = H.shape[3]
	if M is None:
		if F is not None:
			M = np.zeros(F.shape[:2]+(1,1,))
		else:
			M = np.zeros(I.shape[:2]+(1,1,))
	elif len(M.shape) == 2:
		M = M[:,:,np.newaxis,np.newaxis]
	elif len(M.shape) == 3:
		M = M[:,:,:,np.newaxis]
	if F is None:
		F = np.zeros((M.shape[0],M.shape[1],3,ns))
	single_m = (M.shape[2] == 1)
	Fshape = F.shape
	f = vec3(F)
	m = vec3(M)
	if F_T is not None:
		F_T = vec3(F_T)
	if M_T is not None:
		M_T = M_T.flatten()
	Me = np.zeros(I.shape[:2]+(1,M.shape[3],))
	Fe = np.zeros(I.shape+(ns,))

	idx_f, idy_f, idz_f, idf_f = psfshift_idx(F.shape, Fe.shape)
	idx_m, idy_m, idz_m, idf_m = psfshift_idx(M.shape, Me.shape)

	alpha_f = params.alpha_f/ns
	beta_f = params.beta_f/ns 
	lambda_T = params.lambda_T/ns
	alpha_cross_f = params.alpha_cross_f/max(1,ns-1)
	beta_cross_f = params.beta_cross_f/max(1,ns-1)

	alpha_m = params.alpha_f/M.shape[3]
	beta_m = params.beta_f/M.shape[3]
	alpha_cross_m = params.alpha_cross_f/max(1,M.shape[3]-1)
	beta_cross_m = params.beta_cross_f/max(1,M.shape[3]-1)
	lambda_R = params.lambda_R/M.shape[3]
	lambda_MT = params.lambda_T/M.shape[3]
	if single_m:
		beta_cross_m = 0

	## init
	Dx = None
	if state is not None:
		Dx = state.Dx; Dy = state.Dy; DTD = state.DTD; Rn = state.Rn
		vx = state.vx; vy = state.vy; ax = state.ax; ay = state.ay
		vx_m = state.vx_m; vy_m = state.vy_m; ax_m = state.ax_m; ay_m = state.ay_m
		vf = state.vf; af = state.af; vm = state.vm; am = state.am 
		vc = state.vc; ac = state.ac; vc_m = state.vc_m; ac_m = state.ac_m
	if Dx is None:
		Dx, Dy = createDerivatives0(Fshape)
		DTD = (Dx.T @ Dx) + (Dy.T @ Dy)
		vx = np.zeros((Dx.shape[0],Fshape[2]))
		vy = np.zeros((Dy.shape[0],Fshape[2]))
		ay = 0; ax = 0 ## v=Df splitting due to TV and its assoc. Lagr. mult.
		vx_m = np.zeros((Dx.shape[0],1))
		vy_m = np.zeros((Dy.shape[0],1))
		ay_m = 0; ax_m = 0 ## v_m=Dm splitting due to TV (m-part) and its assoc. Lagr. mult.
		af = 0; vf = 0 ## vf=f splitting due to positivity and f=0 outside mask constraint
		am = 0; vm = 0 ## vm=m splitting due to mask between [0,1]
		vc = 0; ac = 0 ## vc=D_cross*f splitting due to cross-image TV and its assoc. Lagr. mult. (TV works along the 3rd dim of 'f', across the different apeparances of the object)
		vc_m = 0; ac_m = 0 ## vc_m=D_cross*m splitting due to cross-mask TV and its assoc. Lagr. mult. (TV works along the 3rd dim of 'm', across the different masks of the object)
		if lambda_R > 0: 
			RnA = createRnMatrix(Fshape[:2]).A
			Rn = RnA.T @ RnA - RnA.T - RnA + np.eye(RnA.shape[0])
			Rn = sparse.csc_matrix(Rn)

	## cross-img derivatives and single image (mask) cases
	if ns == 1 or beta_cross_f == 0: # single image - disable cross-img derivatives (would cause errors)
		crossDf = lambda xx: 0
		crossDf_T = lambda xx: 0 
		crossDf_DTD = lambda xx: 0
	else:
		crossDf = lambda xx: crossD(xx,Fshape[2])
		crossDf_T = lambda xx: crossD_T(xx,Fshape[2])
		crossDf_DTD = lambda xx: crossD_DTD(xx,Fshape[2])
	if single_m or beta_cross_m == 0:
		crossDm = lambda xx: 0
		crossDm_T = lambda xx: 0
		crossDm_DTD = lambda xx: 0
	else:
		crossDm = lambda xx: crossD(xx,1)
		crossDm_T = lambda xx: crossD_T(xx,1)
		crossDm_DTD = lambda xx: crossD_DTD(xx,1)

	fH = fft2(H,axes=(0,1)) # precompute FT
	HT = np.conj(fH) 
	HT3 = np.repeat(HT, 3, axis=2)
	## precompute const RHS for 'f/m' subproblem
	rhs_f = np.real(ifft2(HT3*(fft2(I-B,axes=(0,1))[:,:,:,np.newaxis]),axes=(0,1)))
	rhs_f = params.gamma*np.reshape(rhs_f[idx_f,idy_f,idz_f,idf_f], (Fshape[0]*Fshape[1],-1))
	if lambda_T > 0 and F_T is not None:
		rhs_f += (lambda_T*F_T) ## template matching term lambda_T*|F-F_T|  
	if single_m:
		rhs_m = np.real(ifft2(np.sum(HT,3)*fft2(np.sum(B*(I-B),2),axes=(0,1))[:,:,np.newaxis],axes=(0,1)))[:,:,:,np.newaxis]
	else:
		rhs_m = np.real(ifft2(HT*fft2(np.sum(B*(I-B),2),axes=(0,1))[:,:,np.newaxis,np.newaxis],axes=(0,1)))
	rhs_m = -params.gamma*np.reshape(rhs_m[idx_m,idy_m,idz_m,idf_m],(Fshape[0]*Fshape[1],-1))
	if lambda_MT > 0 and M_T is not None:
		rhs_m += (lambda_MT*M_T) ## template matching term lambda_MT*|M-M_T|  
	
	beta_tv4 = np.r_[np.repeat(beta_f, Fshape[2]*Fshape[3]), np.repeat(beta_m, M.shape[2]*M.shape[3])]
	rel_tol2 = params.rel_tol_f**2
	## ADMM loop
	for iter in range(params.maxiter):
		fdx = Dx @ f; fdy = Dy @ f
		mdx = Dx @ m; mdy = Dy @ m
		fdc = crossDf(f)
		mdc = crossDm(m)
		f_old = f; m_old = m
		## dual/auxiliary var updates, vx/vy minimization (splitting due to TV regularizer)
		if alpha_f > 0 and beta_f > 0:
			val_x = fdx + ax
			val_y = fdy + ay
			shrink_factor = lp_proximal_mapping(np.sqrt(val_x**2 + val_y**2), alpha_f/beta_f, params.lp) # isotropic "TV"
			vx = val_x*shrink_factor
			vy = val_y*shrink_factor
			ax = ax + fdx - vx ## 'a' step
			ay = ay + fdy - vy 
		## vx_m/vy_m minimization (splitting due to TV regularizer for the mask)
		if alpha_m > 0 and beta_m > 0:
			val_x = mdx + ax_m 
			val_y = mdy + ay_m
			shrink_factor = lp_proximal_mapping(np.sqrt(val_x**2 + val_y**2), alpha_m/beta_m, params.lp)
			vx_m = val_x*shrink_factor
			vy_m = val_y*shrink_factor
			ax_m = ax_m + mdx - vx_m
			ay_m = ay_m + mdy - vy_m		
		if beta_cross_f > 0: # cross-image derivative
			val = fdc + ac
			shrink_factor = lp_proximal_mapping(val, alpha_cross_f/beta_cross_f, 1)
			vc = val*shrink_factor
			ac = ac + fdc - vc
		if beta_cross_m > 0: # cross-mask derivative
			val = mdc + ac_m
			shrink_factor = lp_proximal_mapping(val, alpha_cross_m/beta_cross_m, 1)
			vc_m = val*shrink_factor
			ac_m = ac_m + mdc - vc_m
		## vf/vm minimization (positivity and and f-m relation so that F is not large where M is small, mostly means F<=M)
		if params.beta_fm > 0:
			vf = f + af
			vm = m + am
			if params.pyramid_eps > 0: # (m,f) constrained to convex pyramid-like shape to force f<=const*m where const = 1/eps
				if single_m:
					vm, vf = project2pyramid(vm, vf, params.pyramid_eps)
				else:
					for ii in range(ns):
						vm[:,ii], vf[:,ii*Fshape[2]:ii*Fshape[2]+3] = project2pyramid(vm[:,ii], vf[:,ii*Fshape[2]:ii*Fshape[2]+3], params.pyramid_eps)
			else: # just positivity and m in [0,1]
				vf[vf < 0] = 0
				vm[vm < 0] = 0 
				vm[vm > 1] = 1
			# lagrange multiplier (dual var) update
			af = af + f - vf
			am = am + m - vm
	
		## F,M step
		rhs1 = rhs_f + beta_f*(Dx.T @ (vx-ax) + Dy.T @ (vy-ay)) + beta_cross_f*crossDf_T(vc-ac) + params.beta_fm*(vf-af) # f-part of RHS
		rhs2 = rhs_m + beta_m*(Dx.T @ (vx_m-ax_m) + Dy.T @ (vy_m-ay_m)) + beta_cross_m*crossDm_T(vc_m-ac_m) + params.beta_fm*(vm-am) # m-part of RHS
		def estimateFM_cg_Ax(fmfun0):
			fmfun = np.reshape(fmfun0, (-1,(f.shape[1]+m.shape[1])))
			xf = fmfun[:,:f.shape[1]] 
			xm = fmfun[:,f.shape[1]:]
			Fe[idx_f,idy_f,idz_f,idf_f] = xf.flatten()
			Me[idx_m,idy_m,idz_m,idf_m] = xm.flatten()

			HF = np.sum(fH*fft2(Fe,axes=(0,1)),3)
			if single_m:
				bHM = B*(np.real(ifft2(np.sum(fH,3)*fft2(Me,axes=(0,1))[:,:,:,0],axes=(0,1))))
			else:
				bHM = B*(np.real(ifft2(np.sum(fH*fft2(Me,axes=(0,1)),3),axes=(0,1))))

			yf = np.real(ifft2(HT3*(HF - fft2(bHM,axes=(0,1)))[:,:,:,np.newaxis],axes=(0,1)))
			yf = params.gamma*np.reshape(yf[idx_f,idy_f,idz_f,idf_f],(Fshape[0]*Fshape[1],-1))
			if single_m:
				ym = np.real(ifft2(np.sum(HT,3)*fft2(np.sum(B*(bHM - np.real(ifft2(HF,axes=(0,1)))),2),axes=(0,1))[:,:,np.newaxis],axes=(0,1)))[:,:,:,np.newaxis]
			else:
				ym = np.real(ifft2(HT*fft2(np.sum(B*(bHM - np.real(ifft2(HF,axes=(0,1)))),2),axes=(0,1))[:,:,np.newaxis,np.newaxis],axes=(0,1)))
			ym = params.gamma*np.reshape(ym[idx_m,idy_m,idz_m,idf_m],(Fshape[0]*Fshape[1],-1))
			if lambda_T > 0 and F_T is not None:
				yf = yf + lambda_T*xf			
			yf = yf + beta_cross_f*crossDf_DTD(xf)
			ym = ym + beta_cross_m*crossDm_DTD(xm)
			if lambda_MT > 0 and M_T is not None:
				ym = ym + lambda_MT*xm 
			if lambda_R > 0: 
				ym = ym + lambda_R*(Rn @ xm) # mask regularizers
			res = np.c_[yf,ym] + beta_tv4*(DTD @ fmfun) + params.beta_fm*fmfun # common regularizers/identity terms
			return res.flatten()
		A = scipy.sparse.linalg.LinearOperator([(f.shape[1]+m.shape[1])*f.shape[0]]*2, matvec=estimateFM_cg_Ax)
		fm, info = scipy.sparse.linalg.cg(A, np.c_[rhs1,rhs2].flatten(), np.c_[f,m].flatten(), params.cg_tol, params.cg_maxiter)
		fm = np.reshape(fm, (-1,(f.shape[1]+m.shape[1])))
			
		f = fm[:, :f.shape[1]]
		m = fm[:, f.shape[1]:]

		if state is not None:
			continue

		ff = f.flatten()
		df = ff-f_old.flatten()
		mm = m.flatten()
		dm = mm-m_old.flatten()
		rel_diff2_f = (df @ df)/(ff @ ff)
		rel_diff2_m = (dm @ dm)/(mm @ mm)
		
		if params.visualize:
			f_img = ivec3(f, Fshape); m_img = ivec3(m, M.shape)
			if single_m:
				m_img = np.repeat(m_img,ns,axis=3)
			fmon = montageF(f_img)
			mmon = montageF(m_img)[:,:,0]
			imshow_nodestroy(get_visim(None,fmon,mmon,I), 400/np.max(I.shape))
		if params.verbose:
			if params.do_cost: # not fully implemented for all terms, e.g. lambda_R
				Fe[idx_f,idy_f,idz_f,idf_f] = ff
				Me[idx_m,idy_m,idz_m,idf_m] = m
				err = np.sum(np.reshape(np.real(ifft2(fH[:,:,np.newaxis]*fft2(Fe,axes=(0,1)),axes=(0,1)))-B*np.real(ifft2(fH*fft2(Me)))[:,:,np.newaxis]-(I-B), (-1,1))**2)
				cost = (params.gamma/2)*err + alpha_f*np.sum(np.sqrt(fdx**2+fdy**2))
				cost = cost + alpha_m*np.sum(np.sqrt(mdx**2+mdy**2)) 
				if F_T is not None:
					cost = cost + lambda_T*np.sum((f-F_T)**2)/2 
				if M_T is not None:
					cost = cost + np.sum(lambda_MT*(m-M_T)**2)/2
				print("FM: iter={}, reldiff=({}, {}), err={}, cost={}".format(iter, np.sqrt(rel_diff2_f), np.sqrt(rel_diff2_m),err,cost))	
			else:
				print("FM: iter={}, reldiff=({}, {})".format(iter, np.sqrt(rel_diff2_f), np.sqrt(rel_diff2_m)))	

		if rel_diff2_f < rel_tol2 and rel_diff2_m < rel_tol2:
			break

	f_img = ivec3(f, Fshape)
	m_img = ivec3(m, Fshape[:2])
	if state is None:
		return f_img,m_img
	else:
		state.Dx = Dx; state.Dy = Dy; state.DTD = DTD; state.Rn = Rn
		state.vx = vx; state.vy = vy; state.ax = ax; state.ay = ay
		state.vx_m = vx_m; state.vy_m = vy_m; state.ax_m = ax_m; state.ay_m = ay_m
		state.vf = vf; state.af = af; state.vm = vm; state.am = am 
		state.vc = vc; state.vc_m = vc_m; state.ac = ac; state.ac_m = ac_m
		return f_img,m_img,state

def crossD(xx, num_channels=3):
	## cross-image forward derivative (img2-img1); expects input format of 'x' as in 'f' and 'm' in the main function
	return xx[:,num_channels:] - xx[:,:-num_channels]

def crossD_T(xx, num_channels=3):
	## transpose of cross-image derivative (img2-img1); expects input format of 'x' as in 'f' and 'm' in the main function but 1 image less (as is output of crossD)
	return np.c_[-xx[:,:num_channels], xx[:,:-num_channels]-xx[:,num_channels:], xx[:,-num_channels:]]

def crossD_DTD(xx, num_channels=3):
	## short for crossD_T(crossD(x)) (better memory managenemt when written explicitly)
	return np.c_[xx[:,:num_channels]-xx[:,num_channels:2*num_channels], 2*xx[:,num_channels:-num_channels]-xx[:,:-2*num_channels]-xx[:,2*num_channels:], xx[:,-num_channels:]-xx[:,-2*num_channels:-num_channels]]

def main():
	B = cv2.imread(os.path.join('imgs','beach.jpg'))/255
	ns = 4
	H = np.zeros((B.shape[0],B.shape[1],ns))
	M = diskMask(40)
	M1 = np.expand_dims(M,-1)
	F = np.zeros((M.shape[0],M.shape[1],3,ns))
	stx = 60/ns; sty = 120/ns
	for ni in range(ns):
		pars = np.array([[100+ni*stx, 100+ni*sty], [stx, sty]]).T
		H[:,:,ni] = renderTraj(pars, np.zeros(B.shape[:-1]))
		rc = (ni/ns)*0.5 + ((ns-ni)/ns)*0.95
		F[:,:,:,ni] = np.concatenate((0*M1,rc*M1,0.4*M1),2)

	H /= np.sum(H)

	I = fmo_model(B,H,F,M)
	Hmask = fmo_model(np.zeros(B.shape),H,np.repeat(diskMask(20)[:,:,np.newaxis],3,2),M)[:,:,0] > 0.01
	M0 = np.ones(M.shape[:2])

	# imshow(montageF(F),0,3)
	# pdb.set_trace()

	# He = estimateH(I, B, M, F, Hmask)
	# Fe,Me = estimateFM(I,B,np.sum(H,2),M0)
	# Fe,Me = estimateFM_pw(I,B,np.sum(H,2),M0)
	Fe,Me = estimateFM_pw(I,B,H,M0)
	# He,Fe,Me = estimateFMH(I, B, M0, Hmask=Hmask)



if __name__ == "__main__":
    main()