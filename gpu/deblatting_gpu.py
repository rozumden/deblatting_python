import numpy as np
from scipy import sparse

from utils import *
from vis import *
from deblatting import *
from gpu.torch_cg import *

import torch

def estimateFMH_gpu(Ic,Bc,Mc=None,Fc=None,params=None):
	## Estimate F,M,H in FMO equation I = H*F + (1 - H*M)B, where * is convolution
	if params is None:
		params = Params()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)
	I = torch.from_numpy(Ic).float().to(device)
	B = torch.from_numpy(Bc).float().to(device)
	BS = I.shape[0]
	if Mc is None:
		Mc = np.ones(Ic.shape[:2])
	if Fc is None:
		Fc = np.ones((Mc.shape[0],Mc.shape[1],Ic.shape[1]))
	M = torch.ones((BS,1,Mc.shape[0],Mc.shape[1])).float().to(device)
	F = torch.ones((BS,3,Mc.shape[0],Mc.shape[1])).float().to(device)

	H = 0
	params.maxiter = 1
	params.do_fit = 0 # not implemented
	
	rel_tol2 = params.rel_tol_h**2
	stateh = StateH()
	statefm = StateFM()
	stateh.device = device
	statefm.device = device

	## blind loop, iterate over estimateFM and estimateH
	for iter in range(params.loop_maxiter):
		H_old = H
		H, stateh = estimateH_gpu(I, B, M, F, state=stateh, params=params)
		F, M, statefm = estimateFM_gpu(I, B, H, M, F, state=statefm, params=params)
		rel_diff2 = ((H - H_old) ** 2).sum([2,3])/( H ** 2).sum([2,3])
		if params.visualize:
			bi = iter % BS
			Fcpu = F[bi,:,:,:].data.cpu().detach().numpy().transpose(1,2,0)
			Mcpu = M[bi,0,:,:].data.cpu().detach().numpy()
			imshow_nodestroy(get_visim(H[bi,0,:,:].data.cpu().detach().numpy(),Fcpu,Mcpu,Ic[bi,:,:,:].transpose(1,2,0)), 600/np.max(I.shape))
			# pdb.set_trace()
		if params.verbose:
			print("FMH: iter={}, reldiff_h={}".format(iter+1, torch.sqrt(torch.max(rel_diff2))))	
		if (rel_diff2 < rel_tol2).all():
			break

	return H, F, M

def estimateFM_gpu(Ic, Bc, Hc, Mc=None, Fc=None, state=None, params=None):
	## Estimate F,M in FMO equation I = H*F + (1 - H*M)B, where * is convolution
	## M is suggested to be specified to know approximate object size, at least as an array of zeros, for speed-up
	if params is None:
		params = Params()
	params.lambda_R = 0 # not implemented
	if type(Ic) != np.ndarray: ## already in GPU
		I = Ic; B = Bc; H = Hc
		device = state.device
	else:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print(device)
		I = torch.from_numpy(Ic).float().to(device)
		B = torch.from_numpy(Bc).float().to(device)
		H = torch.from_numpy(Hc).float().to(device)
	BS = I.shape[0]

	if Mc is None:
		if Fs is not None:
			M = torch.zeros((BS,1,Fc.shape[0],Fc.shape[1])).float().to(device)
		else:
			M = torch.zeros((BS,1,I.shape[2],I.shape[3])).float().to(device)
	else:
		if type(Mc) == np.ndarray:
			M = torch.zeros((BS,1,Mc.shape[0],Mc.shape[1])).float().to(device)
		else:
			M = Mc

	if Fc is None:
		F = torch.zeros((BS,3,M.shape[2],M.shape[3])).float().to(device)
	else:
		if type(Fc) == np.ndarray:
			F = torch.zeros((BS,3,Fc.shape[0],Fc.shape[1])).float().to(device)
		else:
			F = Fc

	Fshape = F.shape[2:]+(3,)

	## init
	Mask4 = None
	if state is not None:
		vx = state.vx; vy = state.vy; ax = state.ax; ay = state.ay
		vx_m = state.vx_m; vy_m = state.vy_m; ax_m = state.ax_m; ay_m = state.ay_m
		vf = state.vf; af = state.af; vm = state.vm; am = state.am 
		Mask4 = state.Mask4
	if Mask4 is None:
		Mask = psfshift_gpu(1-0*M,I.shape[2:],device) ## to limit the size of mask support
		Mask4 = torch.cat((Mask,Mask,Mask,Mask),1)
		vx = torch.zeros_like(F).float().to(device) 
		vy = torch.zeros_like(F).float().to(device) 
		ay = torch.zeros_like(F).float().to(device)
		ax = torch.zeros_like(F).float().to(device) ## v=Df splitting due to TV and its assoc. Lagr. mult.
		vx_m = torch.zeros_like(M).float().to(device)
		vy_m = torch.zeros_like(M).float().to(device)
		ay_m = torch.zeros_like(M).float().to(device)
		ax_m = torch.zeros_like(M).float().to(device) ## v_m=Dm splitting due to TV (m-part) and its assoc. Lagr. mult.
		af = torch.zeros_like(F).float().to(device)
		vf = torch.zeros_like(F).float().to(device) ## vf=f splitting due to positivity and f=0 outside mask constraint
		am = torch.zeros_like(M).float().to(device)
		vm = torch.zeros_like(M).float().to(device) ## vm=m splitting due to mask between [0,1]

	fH = fft_gpu(H) # precompute FT
	HT = cconj(fH) 
	## precompute const RHS for 'f/m' subproblem
	rhs_f = params.gamma*ifft_gpu(cmul(HT,fft_gpu(I-B)))
	rhs_m = -params.gamma*ifft_gpu(cmul(HT,fft_gpu((B*(I-B)).sum(1,True))))
		
	beta_tv4 = torch.from_numpy(np.repeat(params.beta_f, Fshape[2]+1)).float().to(device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
	rel_tol2 = params.rel_tol_f**2
	## ADMM loop
	for iter in range(params.maxiter):
		fdx, fdy = gradient_gpu(F)
		mdx, mdy = gradient_gpu(M)

		F_old = F; M_old = M
		## dual/auxiliary var updates, vx/vy minimization (splitting due to TV regularizer)
		if params.alpha_f > 0 and params.beta_f > 0:
			val_x = fdx + ax
			val_y = fdy + ay
			shrink_factor = lp_proximal_mapping_gpu(torch.sqrt(val_x**2 + val_y**2), params.alpha_f/params.beta_f, params.lp, device) # isotropic "TV"
			vx = val_x*shrink_factor
			vy = val_y*shrink_factor
			ax = ax + fdx - vx ## 'a' step
			ay = ay + fdy - vy 
			## vx_m/vy_m minimization (splitting due to TV regularizer for the mask)
			val_x = mdx + ax_m 
			val_y = mdy + ay_m
			shrink_factor = lp_proximal_mapping_gpu(torch.sqrt(val_x**2 + val_y**2), params.alpha_f/params.beta_f, params.lp, device)
			vx_m = val_x*shrink_factor
			vy_m = val_y*shrink_factor
			ax_m = ax_m + mdx - vx_m
			ay_m = ay_m + mdy - vy_m
		## vf/vm minimization (positivity and and f-m relation so that F is not large where M is small, mostly means F<=M)
		if params.beta_fm > 0:
			vf = F + af
			vm = M + am
			if params.pyramid_eps > 0: # (m,f) constrained to convex pyramid-like shape to force f<=const*m where const = 1/eps
				vm, vf = project2pyramid_gpu(vm, vf, params.pyramid_eps, device)
			else: # just positivity and m in [0,1]
				vf[vf < 0] = 0
				vm[vm < 0] = 0 
				vm[vm > 1] = 1
			# lagrange multiplier (dual var) update
			af = af + F - vf
			am = am + M - vm

		## F,M step
		rhs1 = rhs_f + psfshift_gpu(params.beta_f*(gradientT_gpu(vx-ax,0) + gradientT_gpu(vy-ay,1)) + params.beta_fm*(vf-af),I.shape[2:],device) # f-part of RHS
		rhs2 = rhs_m + psfshift_gpu(params.beta_f*(gradientT_gpu(vx_m-ax_m,0) + gradientT_gpu(vy_m-ay_m,1)) + params.beta_fm*(vm-am),I.shape[2:],device) # m-part of RHS
		rhs = torch.cat((rhs1,rhs2),1)*Mask4
		def F_bmm(fmfun0):
			Fe = fmfun0[:,:Fshape[2],:,:]
			Me = fmfun0[:,-1:,:,:]
			HF = cmul(fH, fft_gpu(Fe))
			bHM = B*ifft_gpu(cmul(fH,fft_gpu(Me)))
			yf = params.gamma*ifft_gpu(cmul(HT, (HF - fft_gpu(bHM))))
			ym = params.gamma*ifft_gpu(cmul(HT,fft_gpu((B*(bHM - ifft_gpu(HF))).sum(1,True))))
			fmx, fmy = gradient_gpu(ipsfshift_gpu(fmfun0, F.shape))
			DTD = psfshift_gpu(gradientT_gpu(fmx,0) + gradientT_gpu(fmy,1),I.shape[2:],device)
			res = torch.cat((yf,ym),1) + beta_tv4*DTD + params.beta_fm*fmfun0 # common regularizers/identity terms
			res = res * Mask4 
			return res	
		FM = psfshift_gpu(torch.cat((F,M),1), I.shape[2:], device)
		FM, info = cg_batch(F_bmm, rhs, X0=FM, rtol=params.cg_tol, maxiter=params.cg_maxiter)
		fm = ipsfshift_gpu(FM, F.shape)
		F = fm[:,:Fshape[2],:,:]
		M = fm[:,-1:,:,:]
		if state is not None:
			continue

		rel_diff2_f = ((F - F_old) ** 2).sum([1,2,3])/( F ** 2).sum([1,2,3])
		rel_diff2_m = ((M - M_old) ** 2).sum([1,2,3])/( M ** 2).sum([1,2,3])

		if params.visualize:
			bi = iter % BS
			Fcpu = F[bi,:,:,:].data.cpu().detach().numpy().transpose(1,2,0)
			Mcpu = M[bi,0,:,:].data.cpu().detach().numpy()
			imshow_nodestroy(get_visim(Hc[bi,0,:,:],Fcpu,Mcpu,Ic[bi,:,:,:].transpose(1,2,0)), 600/np.max(I.shape))
			# imshow(F[0,:,:,:].data.cpu().detach().numpy().transpose(1,2,0))
		if params.verbose:
			print("FM: iter={}, reldiff=({}, {})".format(iter, torch.sqrt(torch.max(rel_diff2_f)), torch.sqrt(torch.max(rel_diff2_m))))	

		if (rel_diff2_f < rel_tol2).all() and (rel_diff2_m < rel_tol2).all():
			break

	if state is None:
		return F, M
	else:
		state.vx = vx; state.vy = vy; state.ax = ax; state.ay = ay
		state.vx_m = vx_m; state.vy_m = vy_m; state.ax_m = ax_m; state.ay_m = ay_m
		state.vf = vf; state.af = af; state.vm = vm; state.am = am 
		state.Mask4 = Mask4
		return F,M,state

def estimateH_gpu(Ic, Bc, Mc, Fc, state=None, params=None):
	## Estimate H in FMO equation I = H*F + (1 - H*M)B, where * is convolution
	## dimensions: batch, channels, H, W
	## Hmask represents a region in which computations are done
	if params is None:
		params = Params()
	if type(Ic) != np.ndarray: ## already in GPU
		I = Ic; B = Bc; M = Mc; F = Fc
		device = state.device
	else:
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

	H = None
	if state is not None:
		H = state.H
		v_lp = state.v_lp
		a_lp = state.a_lp
		rgnA = state.rgnA

	if H is None:
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
		
		if state is not None:
			continue

		rel_diff2 = ((H - H_old) ** 2).sum([2,3])/( H ** 2).sum([2,3])

		if params.visualize:
			bi = iter % BS
			imshow_nodestroy(get_visim(H[bi,0,:,:].data.cpu().detach().numpy(),Fc[bi,:,:,:].transpose(1,2,0),Mc[bi,0,:,:],Ic[bi,:,:,:].transpose(1,2,0)), 600/np.max(I.shape))
		if params.verbose:
			print("H: iter={}, reldiff={}".format(iter, torch.sqrt(torch.max(rel_diff2))))	
		if (rel_diff2 < rel_tol2).all():
			break

	if state is None:
		return H
	else:
		state.a_lp = a_lp
		state.v_lp = v_lp
		state.H = H
		state.rgnA = rgnA
		return H, state

def gradient_gpu(inp):
	Dx = inp - torch.cat((inp[:,:,1:,:], 0*inp[:,:,-1:,:]), 2)
	Dy = inp - torch.cat((inp[:,:,:,1:], 0*inp[:,:,:,-1:]), 3)
	return Dx, Dy

def gradientT_gpu(inp,out_type=2):
	Dx = inp - torch.cat((inp[:,:,:1,:], inp[:,:,:-1,:]), 2)
	Dy = inp - torch.cat((inp[:,:,:,:1], inp[:,:,:,:-1]), 3)
	if out_type == 0:
		return Dx
	elif out_type == 1:
		return Dy
	else:
		return Dx, Dy

def fft_gpu(inp):
	return torch.rfft(inp, signal_ndim=2, normalized=False, onesided=False)

def ifft_gpu(inp):
	return torch.irfft(inp, signal_ndim=2, normalized=False, onesided=False)

def project2pyramid_gpu(M, F, eps, device):
	## projection of (m,f) values to feasible "pyramid"-like intersection of convex sets (roughly all positive, m<=1, m>=f)
	maxiter = 10 # number of whole cycles, 0=only positivity
	mf = torch.cat((M,F),1).permute(0,2,3,1)
	N = mf.shape[3] # number of conv sets
	Z = torch.zeros(mf.shape+(N,)).float().to(device) # auxiliary vars (sth like projection residuals)
	normal = np.array([-1,eps]) / np.sqrt(1 + eps**2) # dividing plane normal vector (for projection to oblique planes)

	for iter in range(N*maxiter+1): # always end with projection to the first set
		mf_old = mf
		idx = np.mod(iter,N) # set index
		if idx == 0: # projection to f>0, 0<m<1
			mf += Z[:,:,:,:,0]
			mf[mf < 0] = 0 # f,m > 0
			mf[:,:,:,0][mf[:,:,:,0] > 1] = 1 # m < 1
		else: # one of the oblique sets
			mf += Z[:,:,:,:,idx]
			W = mf[:,:,:,idx]*eps > mf[:,:,:,0] # points outside of C_idx			
			proj = mf[:,:,:,0][W] * normal[0] + mf[:,:,:,idx][W] * normal[1] # projection to normal direction
			mf[:,:,:,0][W] -= (normal[0] * proj)
			mf[:,:,:,idx][W] -= (normal[1] * proj)
		Z[:,:,:,:,idx] = mf_old + Z[:,:,:,:,idx] - mf # auxiliaries
	return mf[:,:,:,:1].permute(0,3,1,2), mf[:,:,:,1:].permute(0,3,1,2)

def lp_proximal_mapping_gpu(val_norm, amount, p, device):
	## helper function for vector version of soft thresholding for l1 (lp) minimization
	shrink_factor = torch.zeros_like(val_norm).float().to(device)
	if p == 1: ## soft thresholding
		nz = (val_norm > amount)
		shrink_factor[nz] = (val_norm[nz]-amount) / val_norm[nz]
	else:
		raise Exception('not implemented!')
	return shrink_factor

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
	Hr = torch.roll(H, shifts=shift, dims=(2,3))
	Hc = Hr[:,:,:hsize[2], :hsize[3]]
	return Hc
