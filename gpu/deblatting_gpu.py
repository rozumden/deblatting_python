import numpy as np
from numpy.fft import fft2, ifft2
import scipy.sparse.linalg
from scipy import sparse

from utils import *
from vis import *
from deblatting import *
from gpu.torch_cg import *

import torch

def estimateFM_gpu(Ic, Bc, Hc, Mc=None, Fc=None, params=None):
	## Estimate F,M in FMO equation I = H*F + (1 - H*M)B, where * is convolution
	## M is suggested to be specified to know approximate object size, at least as an array of zeros, for speed-up
	if params is None:
		params = Params()
	if Mc is None:
		if Fs is not None:
			Mc = np.zeros(Fc.shape[:2])
		else:
			Mc = np.zeros(Ic.shape[:2])
	if Fc is None:
		Fc = np.zeros((Mc.shape[0],Mc.shape[1],3))

	Fshape = F.shape
	f = vec3(F)
	m = M.flatten('F')
	if F_T is not None:
		F_T = vec3(F_T)
	if M_T is not None:
		M_T = M_T.flatten('F')
	Me = np.zeros(I.shape[:2])
	Fe = np.zeros(I.shape)

	idx_f, idy_f, idz_f = psfshift_idx(F.shape, I.shape)
	idx_m, idy_m = psfshift_idx(M.shape, I.shape[:2])

	## init
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
	if params.lambda_R > 0: 
		RnA = createRnMatrix(Fshape[:2]).A
		Rn = RnA.T @ RnA - RnA.T - RnA + np.eye(RnA.shape[0])
		Rn = sparse.csc_matrix(Rn)

	fH = fft_gpu(H) # precompute FT
	HT = cconj(fH) 
	## precompute const RHS for 'f/m' subproblem
	rhs_f = ifft_gpu(HT*fft_gpu(I-B))
	rhs_f = params.gamma*np.reshape(rhs_f[idx_f,idy_f,idz_f], (-1,Fshape[2]),'F')
	if params.lambda_T > 0 and F_T is not None:
		rhs_f += (params.lambda_T*F_T) ## template matching term lambda_T*|F-F_T|  
	rhs_m = ifft_gpu(HT*fft_gpu((B*(I-B)).sum(1,True)))
	rhs_m = -params.gamma*rhs_m[idx_m,idy_m] 
	if params.lambda_T > 0 and M_T is not None:
		rhs_m += (params.lambda_T*M_T) ## template matching term lambda_T*|M-M_T|  
	
	beta_tv4 = np.repeat(params.beta_f, Fshape[2]+1)
	rel_tol2 = params.rel_tol_f**2
	## ADMM loop
	for iter in range(params.maxiter):
		fdx = Dx @ f; fdy = Dy @ f
		mdx = Dx @ m; mdy = Dy @ m
		f_old = f; m_old = m
		## dual/auxiliary var updates, vx/vy minimization (splitting due to TV regularizer)
		if params.alpha_f > 0 and params.beta_f > 0:
			val_x = fdx + ax
			val_y = fdy + ay
			shrink_factor = lp_proximal_mapping(np.sqrt(val_x**2 + val_y**2), params.alpha_f/params.beta_f, params.lp) # isotropic "TV"
			vx = val_x*shrink_factor
			vy = val_y*shrink_factor
			ax = ax + fdx - vx ## 'a' step
			ay = ay + fdy - vy 
			## vx_m/vy_m minimization (splitting due to TV regularizer for the mask)
			val_x = mdx + ax_m 
			val_y = mdy + ay_m
			shrink_factor = lp_proximal_mapping(np.sqrt(val_x**2 + val_y**2), params.alpha_f/params.beta_f, params.lp)
			vx_m = val_x*shrink_factor
			vy_m = val_y*shrink_factor
			ax_m = ax_m + mdx - vx_m
			ay_m = ay_m + mdy - vy_m
		## vf/vm minimization (positivity and and f-m relation so that F is not large where M is small, mostly means F<=M)
		if params.beta_fm > 0:
			vf = f + af
			vm = m + am
			if params.pyramid_eps > 0: # (m,f) constrained to convex pyramid-like shape to force f<=const*m where const = 1/eps
				vm, vf = project2pyramid(vm, vf, params.pyramid_eps)
			else: # just positivity and m in [0,1]
				vf[vf < 0] = 0
				vm[vm < 0] = 0 
				vm[vm > 1] = 1
			# lagrange multiplier (dual var) update
			af = af + f - vf
			am = am + m - vm

		## F,M step
		rhs1 = rhs_f + params.beta_f*(Dx.T @ (vx-ax) + Dy.T @ (vy-ay)) + params.beta_fm*(vf-af) # f-part of RHS
		rhs2 = rhs_m + params.beta_f*(Dx.T @ (vx_m-ax_m) + Dy.T @ (vy_m-ay_m)).flatten('F') + params.beta_fm*(vm-am) # m-part of RHS
		def estimateFM_cg_Ax(fmfun0):
			fmfun = np.reshape(fmfun0, (-1,4),'F')
			xf = fmfun[:,:Fshape[2]] 
			xm = fmfun[:,-1]
			Fe[idx_f,idy_f,idz_f] = xf.flatten('F')
			Me[idx_m,idy_m] = xm
			HF = fH[:,:,np.newaxis]*fft2(Fe,axes=(0,1))
			bHM = B*(np.real(ifft2(fH*fft2(Me,axes=(0,1)),axes=(0,1)))[:,:,np.newaxis])
			yf = np.real(ifft2(HT3*(HF - fft2(bHM,axes=(0,1))),axes=(0,1)))
			yf = params.gamma*np.reshape(yf[idx_f,idy_f,idz_f],(-1,Fshape[2]),'F')
			ym = np.real(ifft2(HT*fft2(np.sum(B*(bHM - np.real(ifft2(HF,axes=(0,1)))),2),axes=(0,1)),axes=(0,1)))
			ym = params.gamma*ym[idx_m,idy_m]
			if params.lambda_R > 0: 
				ym = ym + params.lambda_R*(Rn @ xm) # mask regularizers
			res = np.c_[yf,ym] + beta_tv4*(DTD @ fmfun) + params.beta_fm*fmfun # common regularizers/identity terms
			return res.flatten('F')
		A = scipy.sparse.linalg.LinearOperator((4*f.shape[0],4*f.shape[0]), matvec=estimateFM_cg_Ax)
		fm, info = scipy.sparse.linalg.cg(A, np.c_[rhs1,rhs2].flatten('F'), np.c_[f,m].flatten('F'), params.cg_tol, params.cg_maxiter)
		fm = np.reshape(fm, (-1,4),'F')
		f = fm[:, :Fshape[2]]
		m = fm[:, -1]

		ff = f.flatten()
		df = ff-f_old.flatten()
		dm = m-m_old
		rel_diff2_f = (df @ df)/(ff @ ff)
		rel_diff2_m = (dm @ dm)/(m @ m)
		
		if params.visualize:
			f_img = ivec3(f, Fshape); m_img = ivec3(m, Fshape[:2])
			imshow_nodestroy(get_visim(H,f_img,m_img,I), 600/np.max(I.shape))
		if params.verbose:
			print("FM: iter={}, reldiff=({}, {})".format(iter, np.sqrt(rel_diff2_f), np.sqrt(rel_diff2_m)))	

		if rel_diff2_f < rel_tol2 and rel_diff2_m < rel_tol2:
			break

	f_img = ivec3(f, Fshape)
	m_img = ivec3(m, Fshape[:2])
	return f_img,m_img


def estimateH_gpu(Ic, Bc, Mc, Fc, params=None):
	## Estimate H in FMO equation I = H*F + (1 - H*M)B, where * is convolution
	## dimensions: batch, channels, H, W
	## Hmask represents a region in which computations are done
	if params is None:
		params = Params()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)
	if len(Ic.shape) == 3:
		I = torch.from_numpy(Ic[:,:,:,np.newaxis]).float().to(device).permute([3,2,0,1])
		B = torch.from_numpy(Bc[:,:,:,np.newaxis]).float().to(device).permute([3,2,0,1])
		M = torch.from_numpy(Mc[:,:,np.newaxis,np.newaxis]).float().to(device).permute([3,2,0,1])
		F = torch.from_numpy(Fc[:,:,:,np.newaxis]).float().to(device).permute([3,2,0,1])
	elif len(Ic.shape) == 4: # valid input
		I = torch.from_numpy(Ic).float().to(device)
		B = torch.from_numpy(Bc).float().to(device)
		M = torch.from_numpy(Mc).float().to(device)
		F = torch.from_numpy(Fc).float().to(device)
	else:
		print('Not valid input')
		return
	BS = I.shape[0]

	H = torch.zeros((BS,1,I.shape[2],I.shape[3])).float().to(device)
	v_lp = torch.zeros_like(H).float().to(device)
	a_lp = torch.zeros_like(H).float().to(device)
	rgnA = torch.arange(1,I.shape[2]*I.shape[3]+1).float().to(device)
	hsize = I.shape[2:]
	
	iF = fft_gpu(psfshift_gpu(F, hsize, device))
	iFconj = cconj(iF)
	iM = fft_gpu(psfshift_gpu(M, hsize, device))
	iMconj = cconj(iM)
	Fgb = fft_gpu(I-B)
	Fbgb = fft_gpu(B*(I-B))

	## precompute RHS for the 'h' subproblem
	rhs_const = params.gamma*ifft_gpu(cmul(iFconj,Fgb) - cmul(iMconj, Fbgb)).sum(1,True)

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
			FH = fft_gpu(hfun)
			Fh_BMh = cmul(iF, FH) - fft_gpu(B*ifft_gpu(cmul(iM, FH)))
			term = cmul(iFconj,Fh_BMh) - cmul(iMconj, fft_gpu( B*ifft_gpu(Fh_BMh)))
			return params.gamma*ifft_gpu(term).sum(1,True) + params.beta_h*hfun

		H, info = cg_batch(A_bmm, rhs, X0=H, rtol=params.cg_tol, maxiter=params.cg_maxiter)
		rel_diff2 = ((H - H_old) ** 2).sum([2,3])/( H ** 2).sum([2,3])

		if params.visualize:
			bi = iter % BS
			imshow_nodestroy(get_visim(H[bi,0,:,:].data.cpu().detach().numpy(),Fc[bi,:,:,:].transpose(1,2,0),Mc[bi,0,:,:],Ic[bi,:,:,:].transpose(1,2,0)), 600/np.max(I.shape))
		if params.verbose:
			print("H: iter={}, reldiff={}".format(iter, torch.sqrt(torch.max(rel_diff2))))	
		if (rel_diff2 < rel_tol2).all():
			break

	return H.data.cpu().detach().numpy()[:,0,:,:].transpose(1,2,0)

def fft_gpu(inp):
	return torch.rfft(inp, signal_ndim=2, normalized=False, onesided=False)

def ifft_gpu(inp):
	return torch.irfft(inp, signal_ndim=2, normalized=False, onesided=False)

def proj2simplex_gpu(Y, rgnA):
	## euclidean projection of a batch of y to a simplex defined as x>=0 and sum(x(:)) = 1
	Yf = torch.reshape(Y, (Y.shape[0],-1))
	X = torch.sort(Yf, dim=-1, descending=True)[0]
	temp = (torch.cumsum(X, dim=-1)-1)/rgnA
	inds = torch.argmax( (X > temp).float()*rgnA ,dim=1)
	X = torch.reshape(Yf - temp[(torch.arange(0,X.shape[0]),inds)].unsqueeze(1), Y.shape)
	X[X < 0] = 0
	return X

def cmul(arr1, arr2):
	re1 = arr1[:,:,:,:,0]
	im1 = arr1[:,:,:,:,1]
	re2 = arr2[:,:,:,:,0]
	im2 = arr2[:,:,:,:,1]
	return torch.stack([re1 * re2 - im1 * im2, re1 * im2 + im1 * re2], dim = -1)

def cconj(arr):
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
