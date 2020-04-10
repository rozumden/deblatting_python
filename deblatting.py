import cv2
import numpy as np
from numpy.fft import fft2, ifft2

import pdb
from utils import *

class Params:
	def __init__(self):
		self.gamma = 1 # data term weight
		self.alpha = 1 # Lp regularizer weight
		self.beta_lp = 1e2*self.alpha
		self.beta_pos = 0 # splitting v_pos=h due to positivity constraint, penalty weight
		self.lp = 1 # exponent of the Lp regularizer sum |h|^p, allowed values are 0, 1/2, 1, 2
		self.maxiter = 30 # max number of outer iterations
		self.rel_tol = (2e-3)**2 # relative between iterations difference for outer ADMM loop
		self.cg_maxiter = 100 # max number of inner CG iterations ('h' subproblem)
		self.cg_tol = 1e-5 # tolerance for relative residual of inner CG iterations ('h' subproblem)
		if self.lp == 2:
			self.beta_lp = self.alpha # naturally applies Lp=2 in the inner CG function
		

def estimateH_motion(I, B, F, M, Hmask=None, H=None):
	if I.shape != B.shape:
		raise Exception('Shapes must be equal!')
	if Hmask == None:
		Hmask = np.ones(I.shape[:2]).astype(bool)
	## TODO: speed-up by padding and ROI
	params = Params()

	H = np.zeros((np.count_nonzero(Hmask),1))

	v_lp = 0; a_lp = 0; v_pos = 0; a_pos = 0 ## init 
	hsize = Hmask.shape

	## precompute RHS for the 'h' subproblem
	iF = fft2(psfshift(F, hsize))
	iM = fft2(psfshift(M, hsize))
	Fgb = fft2(I-B)
	Fbgb = fft2(B*(I-B))
	iM3 = np.repeat(iM[:, :, np.newaxis], 3, axis=2)

	rhs_const = np.sum(np.real(ifft2(np.conj(iF)*Fgb-np.conj(iM3)*Fbgb)),2)
	rhs_const = params.gamma*rhs_const[Hmask]

	He = np.zeros(hsize)
	## ADMM loop
	for iter in range(params.maxiter):
		H_old = H
		## 'v_lp' minimization
		if params.lp != 2 and params.beta_lp > 0 and params.alpha > 0: # eliminates L2
			v_lp = H + a_lp
			if params.lp == 1:
				temp = v_lp < params.alpha/params.beta_lp
				v_lp[temp] = 0 
				v_lp[~temp] -= params.alpha/params.beta_lp
			elif params.lp == .5:
				temp = v_lp <= 3/2*(params.alpha/params.beta_lp)**(2/3)
				v_lp[temp] = 0
				v_lp[~temp] = 2/3*v_lp[~temp]*(1+np.cos(2/3*np.acos(-3**(3/2)/4*params.alpha/params.beta_lp*v_lp[~temp]**(-3/2))))
			elif params.lp == 0:
				v_lp[v_lp <= sqrt(2*params.alpha/params.beta_lp)] = 0
			a_lp = a_lp + H - v_lp

		## 'v_pos' minimization
		if params.beta_pos > 0:
			v_pos = H + a_pos
			v_pos[v_pos < 0] = 0 ## infinity penalty for negative 'h' values
			v_pos[v_pos > 1] = 1 ## infinity penalty for 'h' values over 1 
			a_pos = a_pos + h - v_pos;
		
		## 'h' minimization
		rhs = rhs_const + params.beta_pos*(v_pos-a_pos) 
		if params.lp != 2:  ## this term is zero for lp=2
			rhs += params.beta_lp*(v_lp-a_lp)

		pdb.set_trace()

	return H

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