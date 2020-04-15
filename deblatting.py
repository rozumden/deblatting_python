import cv2
import numpy as np
from numpy.fft import fft2, ifft2
import scipy.sparse.linalg
from scipy import sparse

import pdb
from utils import *

class Params:
	def __init__(self):
		## universal parameters
		self.loop_maxiter = 100 # max number of (F,M)/H blind loop alternations  
		self.maxiter = 10 # max number of outer iterations
		self.cg_maxiter = 25 # max number of inner CG iterations ('h' subproblem)
		self.rel_tol = 2e-3 # relative between iterations difference for outer ADMM loop
		self.cg_tol = 1e-5 # tolerance for relative residual of inner CG iterations ('h' subproblem)
		self.lp = 1 # exponent of the Lp regularizer sum |h|^p or TV |Df|^p, allowed values are 0, 1
		self.gamma = 1.0 # data term weight
		## parameters for H estimation
		self.alpha_h = 1.0 # Lp regularizer weight
		self.beta_h = 1e3*self.alpha_h
		self.sum1 = True # force sum(H)=1 constraint (via beta_h), takes precedence over lp
		## parameters for F,M estimation
		self.alpha_f = 2e-12 # F,M total variation regularizer weight
		self.beta_f = 10*self.alpha_f # splitting vx/vy=Df due to the TV regularizer
		self.lambda_T = 0 # template L2 term weight
		self.lambda_R = 0 # mask rotation symmetry weight term, lambda_R*|R*m-m|^2 where R is approx rotational averaging, e.g. 1e-2
		## parameters for sub-frame F,M estimation
		self.alpha_cross_f = 2^-12 # cross-image (in 3-dim) image TV regularizer weight 
		self.beta_cross_f = 10*self.alpha_cross_f # splitting vc=D_cross*f due to cross-image TV regularizer
		## visualization parameters 
		self.verbose = True

class StateH:
	def __init__(self):
		self.H = []
		self.a_lp = []
		self.v_lp = []


def estimateFM_motion(I, B, H, M, F=None, F_T=0, M_T=0, oHmask=None, state=None, params=None):
	## Estimate F,M in FMO equation I = H*F + (1 - H*M)B, where * is convolution
	## M is needed to know approximate object size, can be array of zeros
	## F_T, M_T - template for F and M
	if params is None:
		params = Params()
	if oHmask is None:
		Hmask = np.ones(I.shape[:2]).astype(bool)
	if F is None:
		F = np.zeros((M.shape[0],M.shape[1],3))

	Fshape = F.shape
	f = vec3(F)
	m = M.flatten()
	if F_T != 0:
		F_T = vec3(F_T)
	if M_T != 0:
		M_T = M_T.flatten()
	Me = np.zeros(M.shape)
	Fe = np.zeros(F.shape)

	idx_f, idy_f, idz_f = psfshift_idx(F.shape, I.shape)
	idx_m, idy_m = psfshift_idx(M.shape, I.shape[:2])

	## init
	Dx, Dy = createDerivatives0(Fshape)
	
	pdb.set_trace()

	DTD = (Dx.T @ Dx) + (Dy.T @ Dy)
	vx = np.zeros((Dx.shape[0],Fshape[2]))
	vy = np.zeros((Dy.shape[0],Fshape[2]))
	ax = 0; ay = 0 ## v=Df splitting due to TV and its assoc. Lagr. mult.
	vx_m = np.zeros((Dx.shape[0],1))
	vy_m = np.zeros((Dy.shape[0],1))
	ax_m = 0; ay_m = 0 ## v_m=Dm splitting due to TV (m-part) and its assoc. Lagr. mult.
	
	f = 0; af = 0 ## vf=f splitting due to positivity and f=0 outside mask constraint
	vm = 0; am = 0 ## vm=m splitting due to mask between [0,1]
	if params.lambda_R > 0:
		Rn = createRnMatrix(Fshape[:2])
		Rn = Rn @ Rn - Rn.T - Rn + sparse.eye(Rn.shape)

	pdb.set_trace()

	fdx = Dx @ f
	fdy = Dy @ f
	mdx = Dx*m
	mdy = Dy*m
	rel_tol2 = params.rel_tol**2

	return F,M

def estimateH_motion(oI, oB, F, M, oHmask=None, state=None, params=None):
	## Estimate H in FMO equation I = H*F + (1 - H*M)B, where * is convolution
	## Hmask represents a region in which computations are done
	if oI.shape != oB.shape:
		raise Exception('Shapes must be equal!')
	if params is None:
		params = Params()
	if oHmask is None:
		Hmask = np.ones(I.shape[:2]).astype(bool)
		I = oI
		B = oB
	else: ## speed-up by padding and ROI
		pads = np.ceil( (np.array(M.shape)-1)/2 ).astype(int)
		rmin, rmax, cmin, cmax = boundingBox(oHmask, pads)
		I = oI[rmin:rmax,cmin:cmax,:]
		B = oB[rmin:rmax,cmin:cmax,:]
		Hmask = oHmask[rmin:rmax,cmin:cmax]

	H = np.zeros((np.count_nonzero(Hmask),))

	v_lp = 0 ## init 
	a_lp = 0 
	hsize = Hmask.shape

	iF = fft2(psfshift(F, hsize),axes=(0,1))
	iM = fft2(psfshift(M, hsize),axes=(0,1))
	Fgb = fft2(I-B,axes=(0,1))
	Fbgb = fft2(B*(I-B),axes=(0,1))
	iM3 = np.repeat(iM[:, :, np.newaxis], 3, axis=2)

	## precompute RHS for the 'h' subproblem
	rhs_const = np.sum(np.real(ifft2(np.conj(iF)*Fgb-np.conj(iM3)*Fbgb,axes=(0,1))),2)
	rhs_const = params.gamma*rhs_const[Hmask]

	He = np.zeros(hsize)
	rel_tol2 = params.rel_tol**2
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
				v_lp[~temp] -= params.alpha_h/params.beta_h
			elif params.lp == 0:
				v_lp[v_lp <= np.sqrt(2*params.alpha_h/params.beta_h)] = 0
			a_lp = a_lp + H - v_lp

		rhs = rhs_const + params.beta_h*(v_lp-a_lp)

		def estimateH_cg_Ax(hfun):
			He[Hmask] = hfun
			FH = fft2(He,axes=(0,1))
			Fh = iF*np.repeat(FH[:, :, np.newaxis], 3, axis=2) ## apply forward conv (->RGB image, summation over angles)
			BMh = B*np.repeat(np.real(ifft2(iM*FH,axes=(0,1)))[:, :, np.newaxis], 3, axis=2)
			Fh_BMh = Fh - fft2(BMh,axes=(0,1))
			res = np.sum(np.real(ifft2(np.conj(iF)*Fh_BMh - np.conj(iM3)*fft2(B*np.real(ifft2(Fh_BMh,axes=(0,1))),axes=(0,1)),axes=(0,1))),2)
			res = params.gamma*res[Hmask] + (params.beta_h)*hfun
			return res

		A = scipy.sparse.linalg.LinearOperator((H.shape[0],H.shape[0]), matvec=estimateH_cg_Ax)
		H, info = scipy.sparse.linalg.cg(A, rhs, H, params.cg_tol, params.cg_maxiter)

		Diff = (H - H_old)
		rel_diff2 = (Diff @ Diff)/(H @ H)

		if params.verbose:
			if False:
				He[Hmask] = H
				He /= np.max(He)
				imshow(He)
			if False: ## calculate cost
				FH = fft2(He,axes=(0,1))
				FH3 = np.repeat(FH[:, :, np.newaxis], 3, axis=2)
				Fh = iF*FH3
				BMh = B*np.real(ifft2(iM3*FH3,axes=(0,1)));
				err = np.sum((np.real(ifft2(Fh,axes=(0,1)))-BMh-(I-B))**2)
				cost = params.gamma/2*err + params.alpha_h*np.sum(np.abs(H)**params.lp)
				print("H: iter={}, reldiff={}, err={}, cost={}".format(iter, np.sqrt(rel_diff2), err, cost))	
			else:
				print("H: iter={}, reldiff={}".format(iter, np.sqrt(rel_diff2)))	
			# pdb.set_trace()
		
		if rel_diff2 < rel_tol2:
			break

	oHe = np.zeros(oHmask.shape)
	oHe[oHmask] = H

	return oHe

def createDerivatives0(sz):
	N_in = sz[0] * sz[1]
	N_out = (sz[0]+1) * (sz[1]+1)
	idx_in = np.reshape( range(N_in), sz[:2])
	idx_out = np.reshape( range(N_out), np.array(sz[:2])+1)
	## x direction
	v1 = np.ones((sz[0], sz[1]-1))
	v2 = np.ones((sz[0],1))
	inds = idx_out[:-1,np.r_[:(idx_out.shape[1]-1), 1:idx_out.shape[1]]]
	index = idx_in[:, np.r_[0,:(idx_in.shape[1]-1), 1:idx_in.shape[1], idx_in.shape[1]-1] ]
	values = np.hstack((v2,-v1,v1,-v2))
	Dx = sparse.csc_matrix((values.flatten(),(inds.flatten(),index.flatten())), shape=(N_out, N_in))
	## y direction
	v1 = np.ones((sz[0]-1, sz[1]))
	v2 = np.ones((1,sz[1]))
	inds = idx_out[np.r_[:(idx_out.shape[0]-1), 1:idx_out.shape[0]], :-1]
	index = idx_in[np.r_[0,:(idx_in.shape[0]-1), 1:idx_in.shape[0], idx_in.shape[1]-1],:]
	values = np.vstack((v2,-v1,v1,-v2))
	Dy = sparse.csc_matrix((values.flatten(),(inds.flatten(),index.flatten())), shape=(N_out, N_in))
	return Dx, Dy

def proj2simplex(Y):
	## euclidean projection of y (arbitrarily shaped but treated as a single vector) to a simplex defined as x>=0 and sum(x(:)) = 1
	## based on "Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application"; Weiran Wang et al; 2013 (arXiv:1309.1541)
	Yf = Y.flatten()
	X = -np.sort(-Yf) ## descend sort
	temp = (np.cumsum(X)-1)/np.array(range(1,len(X)+1))
	X = np.reshape(Yf - temp[np.nonzero(X > temp)[0][-1]], Y.shape)
	X[X < 0] = 0
	return X

def psfshift(H, usize):
	## PSFSHIFT Moves PSF center to origin and extends the PSF to be the same size as image (for use with FT). ipsfshift does the reverse.
	hsize = H.shape
	usize = usize[:2]
	if len(hsize) > 2:
		Hp = np.zeros((usize[0],usize[1],hsize[2]))
		Hp[:hsize[0],:hsize[1],:] = H ## pad zeros
	else:
		Hp = np.zeros((usize[0],usize[1]))
		Hp[:hsize[0],:hsize[1]] = H ## pad zeros
	shift = tuple((-np.ceil( (np.array(hsize[:2])+1)/2 )+1).astype(int))
	Hr = np.roll(Hp, shift, axis=(0,1))
	return Hr

def ipsfshift(H, hsize):
	## IPSFSHIFT Performs the inverse of 'psfshift' + crops PSF to desired size.
	shift = tuple((np.ceil((np.array(hsize[:2])+1)/2) - 1).astype(int))
	Hr = np.roll(H, shift, axis=(0,1))
	Hc = Hr[:hsize[0], :hsize[1], :]
	return Hc

def psfshift_idx(small, sz_large):
	## variant of psfshift intended for repeated use in a loop (subsequent calls are faster)
	## determines index pairing between 'small' and 'large' images, i.e. if large = psfshift(small, sz_large) then large[idx,idy] = small[mask_small]
	if type(small) == tuple: ## it is shape
		temp = np.reshape(np.array(range(1, np.prod(small)+1)), small).astype(int)
	else: ## it is array
		temp = np.zeros(small.shape).astype(int)
		temp[small] = np.array(range(1, np.count_nonzero(small)+1))
	temp = psfshift(temp, sz_large).astype(int)
	if len(temp.shape) == 2:
		idx,idy = np.nonzero(temp)
		temp_idx = temp[idx,idy]
		pos = np.zeros(temp_idx.shape).astype(int)
		pos[temp_idx-1] = range(len(temp_idx))
		idx = idx[pos]
		idy = idy[pos]
		return idx, idy
	elif len(temp.shape) == 3:
		idx,idy,idz = np.nonzero(temp)
		temp_idx = temp[idx,idy,idz]
		pos = np.zeros(temp_idx.shape).astype(int)
		pos[temp_idx-1] = range(len(temp_idx))
		idx = idx[pos]
		idy = idy[pos]
		idz = idz[pos]
		return idx, idy, idz

def vec3(I):
	return np.reshape(I, (I.shape[0]*I.shape[1], I.shape[2]))

def ivec3(I, ishape):
	return np.reshape(I, ishape)