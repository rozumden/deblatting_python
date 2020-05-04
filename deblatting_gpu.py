import numpy as np
from numpy.fft import fft2, ifft2
import scipy.sparse.linalg
from scipy import sparse

from utils import *
from vis import *
from deblatting import *

def estimateH_tf(I, B, M, F, Hmask=None, state=None, params=None):
	## Estimate H in FMO equation I = H*F + (1 - H*M)B, where * is convolution
	## Hmask represents a region in which computations are done
	if params is None:
		params = Params()
	if Hmask is None:
		Hmask = np.ones(I.shape[:2]).astype(bool)
		Hmask_small = Hmask
	else: ## speed-up by padding and ROI
		if state is None:
			pads = np.ceil( (np.array(M.shape)-1)/2 ).astype(int)
			rmin, rmax, cmin, cmax = boundingBox(Hmask, pads)
			I = I[rmin:rmax,cmin:cmax,:]
			B = B[rmin:rmax,cmin:cmax,:]
			Hmask_small = Hmask[rmin:rmax,cmin:cmax]
		else:
			Hmask_small = Hmask
			
	H = None
	if state is None:
		v_lp = 0 ## init 
		a_lp = 0 
	else:
		H = state.H
		v_lp = state.v_lp
		a_lp = state.a_lp

	if H is None:
		H = np.zeros((np.count_nonzero(Hmask_small),))

	hsize = Hmask_small.shape

	iF = fft2(psfshift(F, hsize),axes=(0,1))
	iM = fft2(psfshift(M, hsize),axes=(0,1))
	Fgb = fft2(I-B,axes=(0,1))
	Fbgb = fft2(B*(I-B),axes=(0,1))
	iM3 = np.repeat(iM[:, :, np.newaxis], 3, axis=2)

	## precompute RHS for the 'h' subproblem
	rhs_const = np.sum(np.real(ifft2(np.conj(iF)*Fgb-np.conj(iM3)*Fbgb,axes=(0,1))),2)
	rhs_const = params.gamma*rhs_const[Hmask_small]

	He = np.zeros(hsize)
	rel_tol2 = params.rel_tol_h**2
	## ADMM loop
	for iter in range(params.maxiter):
		H_old = H
		if params.beta_h > 0: ## also forces positivity
			v_lp = H + a_lp
			if params.sum1:
				v_lp = proj2simplex(v_lp)
			elif params.lp == 1:
				temp = v_lp < params.alpha_h/params.beta_h
				v_lp[temp] = 0 
				v_lp[~temp] -= (params.alpha_h/params.beta_h)
			elif params.lp == 0:
				v_lp[v_lp <= np.sqrt(2*params.alpha_h/params.beta_h)] = 0
			a_lp = a_lp + H - v_lp

		rhs = rhs_const + params.beta_h*(v_lp-a_lp)

		def estimateH_cg_Ax(hfun):
			He[Hmask_small] = hfun
			FH = fft2(He,axes=(0,1))
			Fh = iF*np.repeat(FH[:, :, np.newaxis], 3, axis=2) ## apply forward conv (->RGB image, summation over angles)
			BMh = B*np.repeat(np.real(ifft2(iM*FH,axes=(0,1)))[:, :, np.newaxis], 3, axis=2)
			Fh_BMh = Fh - fft2(BMh,axes=(0,1))
			res = np.sum(np.real(ifft2(np.conj(iF)*Fh_BMh - np.conj(iM3)*fft2(B*np.real(ifft2(Fh_BMh,axes=(0,1))),axes=(0,1)),axes=(0,1))),2)
			res = params.gamma*res[Hmask_small] + (params.beta_h)*hfun
			return res

		A = scipy.sparse.linalg.LinearOperator((H.shape[0],H.shape[0]), matvec=estimateH_cg_Ax)
		H, info = scipy.sparse.linalg.cg(A, rhs, H, params.cg_tol, params.cg_maxiter)

		if state is not None:
			continue

		Diff = (H - H_old)
		rel_diff2 = (Diff @ Diff)/(H @ H)

		if params.visualize:
			imshow_nodestroy(get_visim(He,F,M,I), 600/np.max(I.shape))
		if params.verbose:
			if params.do_cost:
				FH = fft2(He,axes=(0,1))
				FH3 = np.repeat(FH[:, :, np.newaxis], 3, axis=2)
				Fh = iF*FH3
				BMh = B*np.real(ifft2(iM3*FH3,axes=(0,1)));
				err = np.sum((np.real(ifft2(Fh,axes=(0,1)))-BMh-(I-B))**2)
				cost = params.gamma/2*err + params.alpha_h*np.sum(np.abs(H)**params.lp)
				print("H: iter={}, reldiff={}, err={}, cost={}".format(iter, np.sqrt(rel_diff2), err, cost))	
			else:
				print("H: iter={}, reldiff={}".format(iter, np.sqrt(rel_diff2)))	
		if rel_diff2 < rel_tol2:
			break

	oHe = np.zeros(Hmask.shape)
	oHe[Hmask] = H
	if state is None:
		return oHe
	else:
		state.a_lp = a_lp
		state.v_lp = v_lp
		state.H = H
		return oHe, state