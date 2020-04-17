import cv2
import numpy as np
from numpy.fft import fft2, ifft2
import scipy.sparse.linalg
from scipy import sparse

import pdb
from utils import *

class Params:
	def __init__(self): ## Parameters with which users can experiment are marked by #!
		## universal parameters
		self.loop_maxiter = 100 #! max number of (F,M)/H blind loop alternations  
		self.maxiter = 10 #! max number of outer iterations
		self.cg_maxiter = 25 # max number of inner CG iterations ('h' subproblem)
		self.rel_tol = 2e-3 # relative between iterations difference for outer ADMM loop
		self.cg_tol = 1e-5 # tolerance for relative residual of inner CG iterations 
		self.lp = 1 # exponent of the Lp regularizer sum |h|^p or TV |Df|^p, allowed values are 0, 1
		self.gamma = 1.0 # data term weight
		## parameters for H estimation
		self.alpha_h = 1.0 # Lp regularizer weight
		self.beta_h = 1e3*self.alpha_h
		self.sum1 = True # force sum(H)=1 constraint (via beta_h), takes precedence over lp
		## parameters for F,M estimation
		self.alpha_f = 2e-12 #! F,M total variation regularizer weight
		self.lambda_T = 1e-3 #! template L2 term weight, influence: 1e-3 soft, 1e-2 strong, 1e-1 very strong
		self.lambda_R = 0*1e-3 #! mask rotation symmetry weight term, lambda_R*|R*m-m|^2 where R is approx rotational averaging, similar values as *_T
		self.beta_f = 10*self.alpha_f # splitting vx/vy=Df due to the TV regularizer
		self.beta_fm = 1e-3 # splitting vf=f and vm=m due to (F,M) in C constraint where C is prescribed convex set given by positivity and F-M relation, penalty weight
		self.pyramid_eps = 1 # inverse slope of the f<=m/eps constraing for each channel. eps=0 means no constraint (only m in [0,1], f>0), eps=1 means f<=m etc
		## parameters for sub-frame F,M estimation TODO
		self.alpha_cross_f = 2^-12 #! cross-image (in 3-dim) image TV regularizer weight 
		self.beta_cross_f = 10*self.alpha_cross_f # splitting vc=D_cross*f due to cross-image TV regularizer
		## visualization parameters 
		self.verbose = True #!


def estimateFMH(oI,oB,M=None,F=None,oHmask=None):
	## Estimate F,M,H in FMO equation I = H*F + (1 - H*M)B, where * is convolution
	if M is None:
		M = np.ones(oI.shape[:2])
	if F is None:
		F = np.ones((M.shape[0],M.shape[1],oI.shape[2]))
	if oHmask is None:
		Hmask = np.ones(oI.shape[:2]).astype(bool)
		oHmask = Hmask
		I = oI
		B = oB
	else: ## speed-up by padding and ROI
		pads = np.ceil( (np.array(M.shape)-1)/2 ).astype(int)
		rmin, rmax, cmin, cmax = boundingBox(oHmask, pads)
		I = oI[rmin:rmax,cmin:cmax,:]
		B = oB[rmin:rmax,cmin:cmax,:]
		Hmask = oHmask[rmin:rmax,cmin:cmax]

	H = np.zeros(I.shape[:2])
	params = Params()
	params.maxiter = 3
	params.verbose = False
	rel_tol2 = params.rel_tol**2
	stateh = StateH()
	statefm = StateFM()
	## blind loop, iterate over estimateFM and estimateH
	for iter in range(params.loop_maxiter):
		H_old = H

		H, stateh = estimateH(I, B, M, F, oHmask=Hmask, state=stateh, params=params)
		F, M, statefm = estimateFM(I, B, H, M, F, state=statefm, params=params)

		reldiff2 = np.sum((H_old - H)**2) / np.sum(H**2)

		if True:
			imshow(H/np.max(H),wkey=2)
			imshow(np.r_[np.repeat(M[:,:,np.newaxis], 3, axis=2),F], 2, 4)
			print("FMH: iter={}, reldiff_h={}".format(iter, np.sqrt(reldiff2)))	

		if reldiff2 < rel_tol2:
			break

	He = np.zeros(oI.shape[:2])
	He[oHmask] = H[Hmask]
	return He, F, M

def estimateFM(I, B, H, M=None, F=None, F_T=None, M_T=None, oHmask=None, state=None, params=None):
	## Estimate F,M in FMO equation I = H*F + (1 - H*M)B, where * is convolution
	## M is suggested to be specified to know approximate object size, at least as am array of zeros, for speed-up
	## F_T, M_T - template for F and M
	if params is None:
		params = Params()
	if oHmask is None:
		Hmask = np.ones(I.shape[:2]).astype(bool)
	if M is None:
		if F is not None:
			M = np.zeros(F.shape[:2])
		else:
			M = np.zeros(I.shape[:2])
	if F is None:
		F = np.zeros((M.shape[0],M.shape[1],3))

	Fshape = F.shape
	f = vec3(F)
	m = M.flatten()
	if F_T is not None:
		F_T = vec3(F_T)
	if M_T is not None:
		M_T = M_T.flatten()
	Me = np.zeros(I.shape[:2])
	Fe = np.zeros(I.shape)

	idx_f, idy_f, idz_f = psfshift_idx(F.shape, I.shape)
	idx_m, idy_m = psfshift_idx(M.shape, I.shape[:2])

	## init
	Dx = None
	if state is not None:
		Dx = state.Dx; Dy = state.Dy; DTD = state.DTD; Rn = state.Rn
		vx = state.vx; vy = state.vy; ax = state.ax; ay = state.ay
		vx_m = state.vx_m; vy_m = state.vy_m; ax_m = state.ax_m; ay_m = state.ay_m
		vf = state.vf; af = state.af; vm = state.vm; am = state.am 
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
		if params.lambda_R > 0: 
			RnA = createRnMatrix(Fshape[:2]).A
			Rn = RnA.T @ RnA - RnA.T - RnA + np.eye(RnA.shape[0])
			Rn = sparse.csc_matrix(Rn)

	fH = fft2(H,axes=(0,1)) # precompute FT
	HT = np.conj(fH) 
	HT3 = np.repeat(HT[:, :, np.newaxis], 3, axis=2)
	## precompute const RHS for 'f/m' subproblem
	rhs_f = np.real(ifft2(HT3*fft2(I-B,axes=(0,1)),axes=(0,1)))
	rhs_f = params.gamma*np.reshape(rhs_f[idx_f,idy_f,idz_f], (-1,Fshape[2]))
	if params.lambda_T > 0 and F_T is not None:
		rhs_f += (params.lambda_T*F_T) ## template matching term lambda_T*|F-F_T|  
	rhs_m = np.real(ifft2(HT*fft2(np.sum(B*(I-B),2),axes=(0,1)),axes=(0,1)))
	rhs_m = -params.gamma*rhs_m[idx_m,idy_m] 
	if params.lambda_T > 0 and M_T is not None:
		rhs_m += (params.lambda_T*M_T) ## template matching term lambda_T*|M-M_T|  
	
	beta_tv4 = np.repeat(params.beta_f, Fshape[2]+1)
	rel_tol2 = params.rel_tol**2
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
		rhs2 = rhs_m + params.beta_f*(Dx.T @ (vx_m-ax_m) + Dy.T @ (vy_m-ay_m)).flatten() + params.beta_fm*(vm-am) # m-part of RHS
		def estimateFM_cg_Ax(fmfun0):
			fmfun = np.reshape(fmfun0, (-1,4))
			xf = fmfun[:,:Fshape[2]] 
			xm = fmfun[:,-1]
			Fe[idx_f,idy_f,idz_f] = xf.flatten()
			Me[idx_m,idy_m] = xm
			HF = fH[:,:,np.newaxis]*fft2(Fe,axes=(0,1))
			bHM = B*(np.real(ifft2(fH*fft2(Me,axes=(0,1)),axes=(0,1)))[:,:,np.newaxis])
			yf = np.real(ifft2(HT3*(HF - fft2(bHM,axes=(0,1))),axes=(0,1)))
			yf = params.gamma*np.reshape(yf[idx_f,idy_f,idz_f],(-1,Fshape[2]))
			ym = np.real(ifft2(HT*fft2(np.sum(B*(bHM - np.real(ifft2(HF,axes=(0,1)))),2),axes=(0,1)),axes=(0,1)))
			ym = params.gamma*ym[idx_m,idy_m]
			if params.lambda_T > 0 and F_T is not None:
				yf = yf + params.lambda_T*xf
			if params.lambda_T > 0 and M_T is not None:
				ym = ym + params.lambda_T*xm 
			if params.lambda_R > 0: 
				ym = ym + params.lambda_R*(Rn @ xm) # mask regularizers
			res = np.c_[yf,ym] + beta_tv4*(DTD @ fmfun) + params.beta_fm*fmfun # common regularizers/identity terms
			return res.flatten()
		A = scipy.sparse.linalg.LinearOperator((4*f.shape[0],4*f.shape[0]), matvec=estimateFM_cg_Ax)
		fm, info = scipy.sparse.linalg.cg(A, np.c_[rhs1,rhs2].flatten(), np.c_[f,m].flatten(), params.cg_tol, params.cg_maxiter)
		fm = np.reshape(fm, (-1,4))
		f = fm[:, :Fshape[2]]
		m = fm[:, -1]

		ff = f.flatten()
		df = ff-f_old.flatten()
		dm = m-m_old
		rel_diff2_f = (df @ df)/(ff @ ff)
		rel_diff2_m = (dm @ dm)/(m @ m)
		
		if params.verbose:
			# f_img = ivec3(f, Fshape); m_img = ivec3(m, Fshape[:2])
			# imshow(np.r_[np.repeat(m_img[:,:,np.newaxis], 3, axis=2),f_img], 1, 6)
			# pdb.set_trace()
			if False: # calculate cost, ot fully implemented for all terms, e.g. lambda_R
				Fe[idx_f,idy_f,idz_f] = ff
				Me[idx_m,idy_m] = m
				err = np.sum(np.reshape(np.real(ifft2(fH[:,:,np.newaxis]*fft2(Fe,axes=(0,1)),axes=(0,1)))-B*np.real(ifft2(fH*fft2(Me)))[:,:,np.newaxis]-(I-B), (-1,1))**2)
				cost = (params.gamma/2)*err + params.alpha_f*np.sum(np.sqrt(fdx**2+fdy**2))
				cost = cost + params.alpha_f*np.sum(np.sqrt(mdx**2+mdy**2)) 
				if F_T is not None:
					cost = cost + params.lambda_T*np.sum((f-F_T)**2)/2 
				if M_T is not None:
					cost = cost + np.sum(params.lambda_T*(m-M_T)**2)/2
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
		return f_img,m_img,state

def estimateH(oI, oB, M, F, oHmask=None, state=None, params=None):
	## Estimate H in FMO equation I = H*F + (1 - H*M)B, where * is convolution
	## Hmask represents a region in which computations are done
	if oI.shape != oB.shape:
		raise Exception('Shapes must be equal!')
	if params is None:
		params = Params()
	if oHmask is None:
		Hmask = np.ones(oI.shape[:2]).astype(bool)
		oHmask = Hmask
		I = oI
		B = oB
	else: ## speed-up by padding and ROI
		if state is None:
			pads = np.ceil( (np.array(M.shape)-1)/2 ).astype(int)
			rmin, rmax, cmin, cmax = boundingBox(oHmask, pads)
			I = oI[rmin:rmax,cmin:cmax,:]
			B = oB[rmin:rmax,cmin:cmax,:]
			Hmask = oHmask[rmin:rmax,cmin:cmax]
		else:
			I = oI; B = oB; Hmask = oHmask
			
	H = None
	if state is None:
		v_lp = 0 ## init 
		a_lp = 0 
	else:
		H = state.H
		v_lp = state.v_lp
		a_lp = state.a_lp

	if H is None:
		H = np.zeros((np.count_nonzero(Hmask),))

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
				v_lp[~temp] -= (params.alpha_h/params.beta_h)
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
	if state is None:
		return oHe
	else:
		state.a_lp = a_lp
		state.v_lp = v_lp
		state.H = H
		return oHe, state

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
	index = idx_in[np.r_[0,:(idx_in.shape[0]-1), 1:idx_in.shape[0], idx_in.shape[0]-1],:]
	values = np.vstack((v2,-v1,v1,-v2))
	Dy = sparse.csc_matrix((values.flatten(),(inds.flatten(),index.flatten())), shape=(N_out, N_in))
	return Dx, Dy

def createRnMatrix(img_sz, angles=None):
	if angles is None:
		angles = np.array([16, 25, 78, 152])/180*np.pi # selected set of angles
	img_sz = np.array(img_sz[:2])
	idx2, idx1 = np.meshgrid(range(img_sz[1]), range(img_sz[0]))
	offset = (img_sz + 1)/2  # offset between indices and coordinates 
	idx = np.c_[idx1.T.flatten()+1, idx2.T.flatten()+1] - offset
	idx_out = np.zeros((0,)).astype(int)
	idx_in = np.zeros((0,)).astype(int)
	for ki in range(len(angles)): # no antialiasing, NN 'interpolation'
		ca = np.cos(angles[ki]) 
		sa = np.sin(angles[ki])
		R = np.array([[ca, -sa], [sa, ca]]) # rotates by angle; R.' rotates by -angle
		res = np.round(idx @ R + offset).astype(int) # input indices (rotated)
		keep = np.all(np.logical_and(res >= 1, res <= img_sz),1) # crop to image dimensions
		idx_out = np.r_[idx_out, np.nonzero(keep)[0]] # recalculate to linear indices
		res -= 1
		idx_in = np.r_[idx_in, res[keep,0]*img_sz[0] + res[keep,1] ]

	values = np.repeat(1/len(angles),idx_in.shape[0])
	R = sparse.csc_matrix((values,(idx_out,idx_in)), shape=(np.prod(img_sz), np.prod(img_sz)))
	
	temp = np.sum(R,1)
	ww = np.nonzero(temp > 0)[0]
	R[ww,:] = R[ww,:] / temp[ww] # averaging
	return R

def project2pyramid(m, f, eps):
	## projection of (m,f) values to feasible "pyramid"-like intersection of convex sets (roughly all positive, m<=1, m>=f)
	maxiter = 10 # number of whole cycles, 0=only positivity

	mf = np.concatenate((m[:,np.newaxis],f),1)
	N = mf.shape[1] # number of conv sets
	Z = np.zeros((mf.shape[0],mf.shape[1],N)) # auxiliary vars (sth like projection residuals)
	normal = np.array([-1,eps]) / np.sqrt(1 + eps**2) # dividing plane normal vector (for projection to oblique planes)

	for iter in range(N*maxiter+1): # always end with projection to the first set
		mf_old = mf
		idx = np.mod(iter,N) # set index
		if idx == 0: # projection to f>0, 0<m<1
			mf += Z[:,:,0]
			mf[mf < 0] = 0 # f,m > 0
			mf[mf[:,0] > 1,0] = 1 # m < 1
		else: # one of the oblique sets
			mf += Z[:,:,idx]
			W = mf[:,idx]*eps > mf[:,0] # points outside of C_idx			
			proj = (np.c_[mf[W,0],mf[W,idx]] @ normal) # projection to normal direction
			mf[W,0] -= (normal[0] * proj)
			mf[W,idx] -= (normal[1] * proj)
		Z[:,:,idx] = mf_old + Z[:,:,idx] - mf # auxiliaries

	m = mf[:,0]
	f = mf[:,1:]
	return m, f

def lp_proximal_mapping(val_norm, amount, p):
	## helper function for vector version of soft thresholding for l1 (lp) minimization
	shrink_factor = np.zeros(val_norm.shape)
	if p == 1: ## soft thresholding
		nz = (val_norm > amount)
		shrink_factor[nz] = (val_norm[nz]-amount) / val_norm[nz]
	elif p == 1/2: # see eg "Computing the proximity operator of the lp norm..., Chen et al, IET Signal processing, 2016"
		nz = val_norm > 3/2*(val_norm)**(2/3)
		shrink_factor[nz] = (2/3*val_norm[nz]*(1+np.cos(2/3*np.acos(-3**(3/2)/4*amount*val_norm[nz]**(-3/2)))))/(val_norm[nz])
	else:
		raise Exception('not implemented!')
	return shrink_factor

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

class StateH:
	def __init__(self):
		self.H = None
		self.a_lp = 0
		self.v_lp = 0

class StateFM:
	def __init__(self):
		self.vx = 0
		self.vy = 0
		self.ax = 0
		self.ay = 0
		self.vx_m = 0
		self.vy_m = 0
		self.ax_m = 0
		self.ay_m = 0
		self.vf = 0
		self.af = 0
		self.vm = 0
		self.am = 0
		self.Dx = None
		self.Dy = None
		self.DTD = None
		self.Rn = None