import numpy as np
from gpu.deblatting_gpu import *
from deblatting_pw import *

def get_fast_params():
	params = Params()
	params.visualize = False
	params.verbose = False
	params.lambda_R = 0
	params.loop_maxiter = 100
	params.maxiter = 50
	return params

def deblatting_oracle_runner(img,bgrr,debl_dim,gt_traj):
	params = get_fast_params()
	nsplits = gt_traj.shape[1]
	Hso = np.zeros(img.shape[:2]+(nsplits,))

	for ni in range(nsplits): 
		if ni < nsplits-1:
			pars = np.array([gt_traj[:,ni], gt_traj[:,ni+1]-gt_traj[:,ni]]).T
		else:
			pars = np.array([gt_traj[:,ni], gt_traj[:,ni]-gt_traj[:,ni-1]]).T
		Hso[:,:,ni] = renderTraj(pars, Hso[:,:,ni])
	Hso /= Hso.sum()

	Fs,Ms = estimateFM_pw(img,bgrr,Hso,np.zeros(tuple(debl_dim)+(nsplits,)), params=params)
	rgba_tbd3d = np.concatenate((Fs, Ms),2)
	return rgba_tbd3d, Hso

def deblatting_single_runner(imr,bgrr,nsplits,debl_dim):
	params = get_fast_params()
	dI = imr.transpose(2,0,1)[np.newaxis,:,:,:]
	dB = bgrr.transpose(2,0,1)[np.newaxis,:,:,:]
	M0 = np.zeros(debl_dim)
	H,F,M = estimateFMH_gpu(dI, dB, M0, params=params)
	return H,F,M

def deblatting_runner(imr,bgrr,nsplits,debl_dim):
	params = get_fast_params()
	dI = imr.transpose(2,0,1)[np.newaxis,:,:,:]
	dB = bgrr.transpose(2,0,1)[np.newaxis,:,:,:]
	M0 = np.zeros(debl_dim)
	H,F,M = estimateFMH_gpu(dI, dB, M0, params=params)
	Fc = F.cpu().numpy()[0].transpose(1,2,0)
	Mc = M.cpu().numpy()[0].transpose(1,2,0)
	Hc = H.cpu().numpy()[0,0]
	Hc /= Hc.sum()
	Hf,pars = psffit(Hc,True)
	Fc,Mc = estimateFM(imr,bgrr,Hf,Mc[:,:,0], params=params)
	mynorm = np.linalg.norm(pars[:,1])
	if mynorm < 1:
		red_nsplits = 1
	else:
		pcd = nsplits
		while pcd % 2 == 0 and pcd > 0: pcd = pcd // 2 
		red_nsplits = pcd*int(np.min([nsplits/pcd, np.max([1,2**int(np.log2(mynorm))])]))
	Hs = psfsplit(Hc,red_nsplits)
	Fs,Ms = estimateFM_pw(imr,bgrr,Hs,np.zeros(tuple(debl_dim)+(red_nsplits,)), params=params)
	inds = np.repeat(range(red_nsplits), int(nsplits/red_nsplits))
	Hs = Hs[:,:,inds]
	Fs = Fs[:,:,:,inds]
	Ms = Ms[:,:,:,inds]
	est_hs_tbd = np.zeros(imr.shape+(nsplits,))
	est_hs_tbd3d = np.zeros(imr.shape+(nsplits,))
	est_traj = np.zeros((2,nsplits))
	timestamps = np.linspace(0,1,nsplits)
	for ki in range(nsplits): 
		Hsc = Hs[:,:,ki]/np.sum(Hs[:,:,ki])
		est_hs_tbd[:,:,:,ki] = fmo_model(bgrr,Hsc,Fc,Mc)
		est_hs_tbd3d[:,:,:,ki] = fmo_model(bgrr,Hsc,Fs[:,:,:,ki],Ms[:,:,0,ki])
		est_traj[:,ki] = pars[:,0] + timestamps[ki]*pars[:,1]
	rgba_tbd = np.repeat(np.concatenate((Fc, Mc[:,:,None]),2)[...,None], nsplits, 3)
	rgba_tbd3d = np.concatenate((Fs, Ms),2)
	return est_hs_tbd, est_hs_tbd3d, rgba_tbd, rgba_tbd3d, est_traj[[1,0]], Hs
