import numpy as np
from numpy.fft import fft2, ifft2
import scipy.sparse.linalg
from scipy import sparse

from utils import *
from vis import *
from deblatting import *

import torch

def estimateH_gpu(Ic, Bc, Mc, Fc, params=None):
	## Estimate H in FMO equation I = H*F + (1 - H*M)B, where * is convolution
	## Hmask represents a region in which computations are done
	if params is None:
		params = Params()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)
	I = torch.from_numpy(Ic[:,:,:,np.newaxis]).float().to(device).permute([3,2,0,1])
	B = torch.from_numpy(Bc[:,:,:,np.newaxis]).float().to(device).permute([3,2,0,1])
	M = torch.from_numpy(Mc[:,:,np.newaxis,np.newaxis]).float().to(device).permute([3,2,0,1])
	F = torch.from_numpy(Fc[:,:,:,np.newaxis]).float().to(device).permute([3,2,0,1])

	nx = I.shape[2]*I.shape[3]

	v_lp = 0
	a_lp = 0 
	H = torch.zeros((nx,))
	hsize = I.shape[2:]
	
	iF = torch.rfft(psfshift_gpu(F, hsize, device), signal_ndim=2, normalized=False, onesided=False)
	iFconj = torch.cat([iF[:,:,:,:,:1],-iF[:,:,:,:,1:]], dim=4)
	iM = torch.rfft(psfshift_gpu(M, hsize, device), signal_ndim=2, normalized=False, onesided=False)
	iMconj = torch.cat([iM[:,:,:,:,:1],-iM[:,:,:,:,1:]], dim=4)
	Fgb = torch.rfft(I-B, signal_ndim=2, normalized=False, onesided=False)
	Fbgb = torch.rfft(B*(I-B), signal_ndim=2, normalized=False, onesided=False)

	pdb.set_trace()

	## precompute RHS for the 'h' subproblem
	term = iFconj*Fgb - iMconj*Fbgb
	rhs_const = np.sum(np.real(ifft2(term,axes=(0,1))),2)
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
	return oHe
	
def psfshift_gpu(H, usize, device):
	## PSFSHIFT Moves PSF center to origin and extends the PSF to be the same size as image (for use with FT). 
	## ipsfshift_gpu does the reverse. GPU torch version
	hsize = H.shape
	Hp = torch.zeros((hsize[0],hsize[1],usize[0],usize[1])).float().to(device)
	Hp[:,:,:hsize[2],:hsize[3]] = H ## pad zeros

	shift = tuple((-np.ceil( (np.array(hsize[2:])+1)/2 )+1).astype(int))
	Hr = torch.roll(Hp, shifts=shift, dims=(2,3))
	return Hr

def ipsfshift_gpu(H, hsize):
	## IPSFSHIFT Performs the inverse of 'psfshift' + crops PSF to desired size.
	shift = tuple((np.ceil((np.array(hsize[2:])+1)/2) - 1).astype(int))
	Hr = np.roll(H, shifts=shift, dims=(2,3))
	Hc = Hr[:,:,:hsize[2], :hsize[3]]
	return Hc
