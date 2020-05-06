import numpy as np
from numpy.fft import fft2, ifft2
import scipy.sparse.linalg
from scipy import sparse

from utils import *
from vis import *
from deblatting import *
from gpu.torch_cg import *

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
	BS = I.shape[0]

	H = torch.zeros((BS,1,I.shape[2],I.shape[3])).float().to(device)
	v_lp = torch.zeros_like(H).float().to(device)
	a_lp = torch.zeros_like(H).float().to(device)
	rgnA = torch.arange(1,I.shape[2]*I.shape[3]+1).float().to(device)
	hsize = I.shape[2:]
	
	iF = torch.rfft(psfshift_gpu(F, hsize, device), signal_ndim=2, normalized=False, onesided=False)
	iFconj = complex_conj(iF)
	iM = torch.rfft(psfshift_gpu(M, hsize, device), signal_ndim=2, normalized=False, onesided=False)
	iMconj = complex_conj(iM)
	Fgb = torch.rfft(I-B, signal_ndim=2, normalized=False, onesided=False)
	Fbgb = torch.rfft(B*(I-B), signal_ndim=2, normalized=False, onesided=False)

	## precompute RHS for the 'h' subproblem
	term = complex_multiplication(iFconj,Fgb) - complex_multiplication(iMconj, Fbgb)
	rhs_const = params.gamma*torch.irfft(term, signal_ndim=2, normalized=False, onesided=False).sum(1).unsqueeze(1)
		
	rel_tol2 = params.rel_tol_h**2
	## ADMM loop
	for iter in range(params.maxiter):
		H_old = H
		if params.beta_h > 0: ## also forces positivity
			v_lp = H + a_lp
			if params.sum1:
				v_lp = proj2simplex_gpu(v_lp, rgnA)
			a_lp = a_lp + H - v_lp

		rhs = rhs_const + params.beta_h*(v_lp-a_lp)

		def A_bmm(hfun):
			FH = torch.rfft(hfun, signal_ndim=2, normalized=False, onesided=False)
			Fh = complex_multiplication(iF, FH)
			BMh = B*torch.irfft(complex_multiplication(iM, FH),signal_ndim=2, normalized=False, onesided=False)
			Fh_BMh = Fh - torch.rfft(BMh, signal_ndim=2, normalized=False, onesided=False)
			term = complex_multiplication(iFconj,Fh_BMh) - complex_multiplication(iMconj, torch.rfft( B*torch.irfft(Fh_BMh,signal_ndim=2, normalized=False, onesided=False),signal_ndim=2, normalized=False, onesided=False))
			res = torch.irfft(term,signal_ndim=2, normalized=False, onesided=False).sum(1).unsqueeze(1)
			res = params.gamma*res + params.beta_h*hfun
			return res

		H, info = cg_batch(A_bmm, rhs, X0=H, rtol=params.cg_tol, maxiter=params.cg_maxiter)
		rel_diff2 = torch.sum((H - H_old) ** 2)/torch.sum( H ** 2)

		if params.visualize:
			imshow_nodestroy(get_visim(H[0,0,:,:].data.cpu().detach().numpy(),Fc,Mc,Ic), 600/np.max(I.shape))
		if params.verbose:
			print("H: iter={}, reldiff={}".format(iter, torch.sqrt(rel_diff2)))	
		if rel_diff2 < rel_tol2:
			break

	return H.data.cpu().detach().numpy()[0,0,:,:]

def proj2simplex_gpu(Y, rgnA):
	## euclidean projection of a batch of y to a simplex defined as x>=0 and sum(x(:)) = 1
	Yf = torch.reshape(Y, (Y.shape[0],-1))
	X = torch.sort(Yf, dim=-1, descending=True)[0]
	temp = (torch.cumsum(X, dim=-1)-1)/rgnA
	inds = torch.argmax( (X > temp).float()*rgnA ,dim=1)
	X = torch.reshape(Yf - temp[(torch.arange(0,X.shape[0]),inds)].unsqueeze(1), Y.shape)
	X[X < 0] = 0
	return X

def complex_multiplication(arr1, arr2):
	re1 = arr1[:,:,:,:,0]
	im1 = arr1[:,:,:,:,1]
	re2 = arr2[:,:,:,:,0]
	im2 = arr2[:,:,:,:,1]
	return torch.stack([re1 * re2 - im1 * im2, re1 * im2 + im1 * re2], dim = -1)

def complex_conj(arr):
	return torch.stack([arr[:,:,:,:,0],-arr[:,:,:,:,1]], dim=-1)

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
