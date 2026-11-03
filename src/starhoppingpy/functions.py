import os
import glob
import numpy as np
import numpy
import matplotlib.pyplot as plt
import scipy.io as sio
from . import basic
import time

from astropy.io import fits
from matplotlib.colors import LogNorm
from scipy.stats import rankdata
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate
from scipy import ndimage


#cube.ndim = 3
#image.ndim = 2

def remove_median_in_region(cube, ww):

    cubenew = cube.copy()  # copy of cube
    if cubenew.ndim == 3:
        num = np.shape(cubenew)[0] # number of images
        for i in range(0,num,1): # loops through the number if images
            img = cubenew[i,:,:] # each image is saved as img
            img -= np.median(img[ww])  # removes median in a region xsimg
            cubenew[i,:,:] = img # saves each image in the cube with median removed in every image 
    elif cubenew.ndim == 2:
        cubenew -= np.median(cubenew[ww])  # removes median in a region xsimg
    else: print('We have a problem with the input cube')

    return cubenew


def remove_bgnd_slope(cube):

    cubenew = cube.copy()  # copy of cube
    if cubenew.ndim == 3:
        num = np.shape(cubenew)[0] # number of images
        for i in range(0,num,1): # loops through the number if images
            img = cubenew[i,:,:] # each image is saved as img
            bgnd= basic.fit_poly2d(img, degree=1)
            img -= bgnd  # removes median in a region xsimg
            cubenew[i,:,:] = img # saves each image in the cube with median removed in every image 
    elif cubenew.ndim == 2:
        cubenew -= basic.fit_poly2d(cubenew,degree=1)  # removes median in a region xsi
    else:
        print('We have a problem with the input cube')

    return cubenew

def sublcomb1dm(im1, imc): # linear combines im1 cube and subtract linear combination of imc
    e = imc
    ez = np.shape(e)[0]
    dz = np.shape(e)[1]
    a = im1
    med0 = 0
    s = np.tile(np.median(e,axis=1),(dz,1)).T
    e-=s
    p = (ez,ez)
    am = np.zeros(p)
    for m in range(0,ez,1):
        am[:,m] = np.sum(np.tile(e[m,:],(ez,1))*e,axis=1)
    bm = np.tile(a,(ez,1))
    bm = np.sum(e*bm,axis=1)
    u, w, v = np.linalg.svd(am, full_matrices=True)
    cr, _, _, _ = np.linalg.lstsq(am,bm,rcond=None)
    T = np.sum((np.tile(cr,(dz,1)).T)*e,axis=0)
    sub = T+med0
    return sub,cr




def sublcomb1d(im1, imc, verbose=False, ncomp=None, nomedsub=False):
    """
    Linearly combine reference frames to approximate a single image using
    least squares fitting.

    Parameters:
    im1 (np.array): The target vector.
    imc (np.array): The reference matrix (first dimension as different frames).
    verbose (bool): If True, print additional debug information.
    ncomp (int): Number of images to use in linear cutoff by coefficient magnitude.
    nomedsub (bool): If True, do not subtract the median from the target vector.

    Returns:
    np.array: The subtraction result.
    dict: Dictionary containing coefficients and optional messages.
    """
    n, m = imc.shape
    res = {}

    a = im1.copy()
    if not nomedsub:
        med0 = np.median(a)
        a -= med0
    else:
        med0 = 0

    # Remove median from each row of the reference matrix
    medians = np.median(imc, axis=1, keepdims=True)
    e = imc - medians

    # Construct matrix AM for the minimization
    AM = np.dot(e, e.T)
    
    # Construct vector BM
    BM = np.dot(e, a)

    # Solve using Singular Value Decomposition
    U, s, Vt = np.linalg.svd(AM, full_matrices=False)
    s_inv = np.linalg.pinv(np.diag(s))
    cr = np.dot(np.dot(Vt.T, s_inv), np.dot(U.T, BM))
    
    # Apply coefficients to compute the linear combination
    if ncomp is not None and ncomp < n:
        idx_largest = np.argsort(np.abs(cr))[-ncomp:]
        T = np.dot(cr[idx_largest], e[idx_largest, :])
    else:
        T = np.dot(cr, e)

    sub = T + med0
    
    if verbose:
        print('Ref old rms:', np.std(im1))
        print('Ref new rms:', np.std(im1 - sub))
        print('----------------')
    
    res['coeff'] = cr
    return sub, cr

def zscalediff(img1, img2, wm, results=None):
    img1 = np.ascontiguousarray(img1)
    img2 = np.ascontiguousarray(img2)
    # Start the clock
    #start_overall = time.time()

    # Print the shapes of img1 and img2 to verify their sizes
    #print(f"Shape of img1: {img1.shape}")
    #print(f"Shape of img2: {img2.shape}")

    # Calculate and print the sparseness of the mask
    total_elements = wm.size
    true_elements = np.sum(wm)
    sparseness = true_elements / total_elements
    #print(f"Total elements in mask: {total_elements}")
    #print(f"Number of True elements in mask: {true_elements}")
    #print(f"Sparseness of mask (fraction of True elements): {sparseness:.4f}")

    # Extract the elements where the mask is True as 1D arrays for computation
    start_time = time.time()
    b = img2[wm].flatten()  # Ensures b is 1D
    a = img1[wm].flatten()  # Ensures a is 1D
    end_time = time.time()
    #print(f"Time for extracting and flattening data: {end_time - start_time:.6f} seconds")

    # Perform the efficient element-wise operations for dot products
    start_time = time.time()
    denominator = np.einsum('i,i->', b, b)  # Efficient sum of squares
    numerator = np.einsum('i,i->', a, b)    # Efficient dot product
    fac = numerator / denominator
    end_time = time.time()
    #print(f"Time for computing dot products: {end_time - start_time:.6f} seconds")

    # Calculate the difference and retain the shape of img1
    start_time = time.time()
    diff = img1 - fac * img2
    end_time = time.time()
    #print(f"Time for computing difference: {end_time - start_time:.6f} seconds")

    # Optionally store results
    if results is not None:
        results['fac'] = fac
        results['mimg'] = fac * img2

    # Overall elapsed time
    end_overall = time.time()
    #print(f"Total elapsed time: {end_overall - start_overall:.6f} seconds")

    return diff

# will use combine rgns w1 and w2, but will make sure the counts are treated the same.
def zscalediff2(img1o,img2o,w1,w2,results=None):

    img1 = img1o.copy()
    img2 = img2o.copy()
        
    rgnfac = rgn_fac(img1, w1, w2)
    
    img1[w1] /= rgnfac
    img2[w1] /= rgnfac

    wm = w1 | w2

    b = img2[wm]
    a = img1[wm]

    fac = np.sum(a*b)/np.sum(b*b)

    diff=img1o-fac*img2o
    
    if results is not None:
        results['fac']=fac
        results['mimg']=fac*img2o
    
    return diff


#for each image in simc gets the inx of the best matches rimc
def best_refs(simc, rimc, wm, numref, rmsa=None):
    ginx=basic.get_mask

    if simc.ndim == 2:
        simc = np.expand_dims(simc, axis=0)  # Convert 2D simc to 3D with one element

    if simc.ndim == 3 and rimc.ndim == 3:
        nums = simc.shape[0]  # number of science images
        numr = rimc.shape[0]  # number of ref images
        wb=ginx(simc[0],60,120)

        rms = np.zeros((numr, nums)) + 1e30
        rms0 = np.zeros((numr, nums)) + 1e30
        best_indices_array = np.zeros((nums, numref), dtype=int)
        con = np.zeros(nums)
        
        for i in range(nums):
            imgsci = simc[i]
            con[i] = 1/np.std(imgsci[wb])
            
            for j in range(numr):
                imgref = rimc[j]
                diffa = zscalediff(imgsci, imgref, wm)
                rms[j, i] = np.std(diffa[wm]) / np.std(imgsci[wm])
                rms0[j, i] = np.std(diffa[wb])
                
            best_indices_array[i] = np.argsort(rms[:, i])[:numref]

        print("Finished cross match rms calcs.")
        print(rms[:, i])
        print(rms[best_indices_array[i], i])
        #input('press any key...')
        
        if rmsa is not None:
            rmsa['rat'] = rms
            rmsa['abs'] = rms0
            rmsa['con'] = con
                        
        return best_indices_array

    else:
        print('Invalid shapes for input arrays.')
        return None

def good_refs(simc, rimc, wm, numref, rmsa=None):
    ginx = basic.get_mask  # Assuming basic.get_mask is a predefined function

    if simc.ndim == 2:
        simc = np.expand_dims(simc, axis=0)  # Convert 2D simc to 3D with one element

    if simc.ndim == 3 and rimc.ndim == 3:
        nums = simc.shape[0]  # number of science images
        numr = rimc.shape[0]  # number of ref images
        wb = ginx(simc[0], 60, 120)

        rms = np.zeros((numr, nums)) + 1e30
        rms0 = np.zeros((numr, nums)) + 1e30
        best_indices_array = np.zeros((numr, nums), dtype=int)
        truncation_points = np.full(nums, numref, dtype=int)  # Initialize truncation points
        con = np.zeros(nums)
          
        for i in range(nums):
            imgsci = simc[i]
            con[i] = 1 / np.std(imgsci[wb])

            for j in range(numr):
                imgref = rimc[j]
                diffa = zscalediff(imgsci, imgref, wm)  # Assuming zscalediff is a predefined function
                rms[j, i] = np.std(diffa[wm]) / np.std(imgsci[wm])
                rms0[j, i] = np.std(diffa[wb])

            sorted_indices = np.argsort(rms[:, i])
            best_indices_array[:, i] = sorted_indices  # Store all indices

            print(f"Image {i} - Sorted Indices: {sorted_indices[:numref]}")
            print(f"Image {i} - Sorted RMS values: {rms[sorted_indices[:numref], i]}")

            # Determine the truncation point for this row
            trunc_point = np.argmax(rms[sorted_indices[:numref], i] >= 0.9)
            if trunc_point == 0 and rms[sorted_indices[0], i] < 0.9:
                trunc_point = numref
            truncation_points[i] = trunc_point

        # Find the smallest truncation point
        #jmin = truncation_points.min()
        jcut = np.median(truncation_points)
        
        # Truncate best_indices_array based on the smallest truncation point found
        best_indices_array = best_indices_array[:jcut, :]

        for i in range(nums):
            print(f"Image {i} - Final Truncated Indices: {best_indices_array[:, i]}")
            print(f"Image {i} - Final Truncated RMS values: {rms[best_indices_array[:, i], i]}")

        if rmsa is not None:
            rmsa['rat'] = rms
            rmsa['abs'] = rms0
            rmsa['con'] = con

        return best_indices_array

    else:
        print('Invalid shapes for input arrays.')
        return None

def good_refs2(simc, rimc, wm, numref, rmsa=None):
    ginx = basic.get_mask  # Assuming basic.get_mask is a predefined function


    if simc.ndim == 2:
        simc = np.expand_dims(simc, axis=0)  # Convert 2D simc to 3D with one element

    if simc.ndim == 3 and rimc.ndim == 3:
        nums = simc.shape[0]  # number of science images
        numr = rimc.shape[0]  # number of ref images
        wb = ginx(simc[0], 60, 120)

        rms = np.zeros((numr, nums)) + 1e30
        rms0 = np.zeros((numr, nums)) + 1e30
        best_indices_array = np.zeros((numr, nums), dtype=int)
        truncation_points = np.full(nums, numref, dtype=int)  # Initialize truncation points
        con = np.zeros(nums)
          
        for i in range(nums):
            imgsci = simc[i]
            con[i] = 1 / np.std(imgsci[wb])

            for j in range(numr):
                imgref = rimc[j]
                diffa = zscalediff(imgsci, imgref, wm)  # Assuming zscalediff is a predefined function
                rms[j, i] = np.std(diffa[wm]) / np.std(imgsci[wm])
                rms0[j, i] = np.std(diffa[wb])

            sorted_indices = np.argsort(rms[:, i])
            best_indices_array[:, i] = sorted_indices  # Store all indices

            print(f"Image {i} - Sorted Indices: {sorted_indices[:numref]}")
            print(f"Image {i} - Sorted RMS values: {rms[sorted_indices[:numref], i]}")

            rms_thres = np.min(rms[:, i])*1.2
    
            # Determine the truncation point for this row
            trunc_point = np.argmax(rms[sorted_indices[:numref], i] >= rms_thres)
            if trunc_point == 0 and rms[sorted_indices[0], i] < rms_thres:
##                trunc_point = numref
                trunc_point = max(0, min(numref, rms.shape[0] - 1))
            truncation_points[i] = trunc_point
            print(f"Image {i} - Trunc Point: {trunc_point}, RMS at Trunc Point: {rms[sorted_indices[trunc_point], i]}")

        # Find the smallest truncation point
        #jmin = truncation_points.min()
        jcut = int(np.median(truncation_points))
        # Truncate best_indices_array based on the median truncation point
        best_indices_array = best_indices_array[:jcut, :]

        for i in range(nums):
            print(f"Image {i} - Final Truncated Indices: {best_indices_array[:, i]}")
            print(f"Image {i} - Final Truncated RMS values: {rms[best_indices_array[:, i], i]}")

        if rmsa is not None:
            rmsa['rat'] = rms
            rmsa['abs'] = rms0
            rmsa['con'] = con

        return best_indices_array

    else:
        print('Invalid shapes for input arrays.')
        return None
    
def good_refs_mask(simc, rimc, wm0, numref, rmsa=None, mask=None, pas=None):
    if pas is None:
        pas = np.zeros(simc.shape[0])  # Default to no rotation if pas not provided
        
    if mask is None:
        mask = np.ones_like(simc[0], dtype=bool)  # Default mask of True values

    ginx = basic.get_mask  # Assuming basic.get_mask is a predefined function


    if simc.ndim == 2:
        simc = np.expand_dims(simc, axis=0)  # Convert 2D simc to 3D with one element

    if simc.ndim == 3 and rimc.ndim == 3:
        nums = simc.shape[0]  # number of science images
        numr = rimc.shape[0]  # number of ref images
        wb = ginx(simc[0], 60, 120)

        rms = np.zeros((numr, nums)) + 1e30
        rms0 = np.zeros((numr, nums)) + 1e30
        best_indices_array = np.zeros((numr, nums), dtype=int)
        truncation_points = np.full(nums, numref, dtype=int)  # Initialize truncation points
        con = np.zeros(nums)
          
        for i in range(nums):
            maskr = ndimage.rotate(mask,-pas[i],reshape=False) > 0.5

            wm = wm0 | maskr
            imgsci = simc[i]
            con[i] = 1 / np.std(imgsci[wb])

            for j in range(numr):
                imgref = rimc[j]
                diffa = zscalediff(imgsci, imgref, wm)  # Assuming zscalediff is a predefined function
                rms[j, i] = np.std(diffa[wm]) / np.std(imgsci[wm])
                rms0[j, i] = np.std(diffa[wb])

            sorted_indices = np.argsort(rms[:, i])
            best_indices_array[:, i] = sorted_indices  # Store all indices

            print(f"Image {i} - Sorted Indices: {sorted_indices[:numref]}")
            print(f"Image {i} - Sorted RMS values: {rms[sorted_indices[:numref], i]}")

            rms_thres = np.min(rms[:, i])*2.0
    
            # Determine the truncation point for this row
            trunc_point = np.argmax(rms[sorted_indices[:numref], i] >= rms_thres)
            if trunc_point == 0 and rms[sorted_indices[0], i] < rms_thres:
                trunc_point = numref
            truncation_points[i] = trunc_point
            print(f"Image {i} - Trunc Point: {trunc_point}, RMS at Trunc Point: {rms[sorted_indices[trunc_point], i]}")

        # Find the smallest truncation point
        #jmin = truncation_points.min()
        jcut = int(np.median(truncation_points))
        # Truncate best_indices_array based on the median truncation point
        best_indices_array = best_indices_array[:jcut, :]

        for i in range(nums):
            print(f"Image {i} - Final Truncated Indices: {best_indices_array[:, i]}")
            print(f"Image {i} - Final Truncated RMS values: {rms[best_indices_array[:, i], i]}")

        if rmsa is not None:
            rmsa['rat'] = rms
            rmsa['abs'] = rms0
            rmsa['con'] = con

        return best_indices_array

    else:
        print('Invalid shapes for input arrays.')
        return None
    
#Swastik found an error in this one. Only last science image considered.
def get_index_rms00(numref, wm, *inputdata):

    if len(inputdata) == 4:

        nums = np.shape(inputdata[2]["imc"])[0]  # number of science images
        numr = np.shape(inputdata[3]["imc"])[0]  # number of ref images

        wsci = inputdata[0]
        wref = inputdata[1]

        rms = np.zeros((numr, nums)) + 1e30

        for i in range(nums):
            if i % 10 == 0:
                print("Image: ", i)
            imgsci = wsci[i, :]
            for j in range(numr):
                imgref = wref[j, :, :]
                diffa = zscalediff(imgsci, imgref, wm)
                rms[j, i] = np.std(diffa[wm]) / np.std(imgsci[wm])
                if str(inputdata[2]['lambdas'][i]) != str(inputdata[3]['lambdas'][j]):
                    rms[j, i] = 100000000
                if str(inputdata[2]['stokes'][i]) != str(inputdata[3]['stokes'][j]):
                    rms[j, i] = 100000000
                if str(inputdata[2]['polfilts'][i]) != str(inputdata[3]['polfilts'][j]):
                    rms[j, i] = 100000000
                print(inputdata[2]['polfilts'][i], inputdata[3]['polfilts'][j])
            oa = np.argsort((rms[:, i]))  # sort the rms from smallest to largest to use the best 16 as references

        print("Finished cross match rms calcs.")

    elif len(inputdata) == 2:

        numr = np.shape(inputdata[1])[0]  # number of ref images
        wref = inputdata[1]

        rms = [None] * numr

        imgscimed = inputdata[0]  # ONE MEDIAN SCIENCE IMAGE ONLY
        for j in range(numr):
            imgref = wref[j, :, :]
            diffa = zscalediff(imgscimed, imgref, wm)
            rms[j] = np.std(diffa[wm]) / np.std(imgscimed[wm])  # saves rms of the diff between each image and the median in region
#           print(j, rms[j])

        oa = np.argsort((rms))  # sort the rms from smallest to largest to use the best 16 as references
    else:
        print('Problema con rms')

    oc = oa[0:numref]  # select the best 16 ref

    return oc

#swastik's fixed version below.
def get_index_rms(numref, wm, *inputdata):

	if len(inputdata)==4:

		nums = np.shape(inputdata[2]["imc"])[0] # number of science images
		numr = np.shape(inputdata[3]["imc"])[0] # number of ref images

		wsci=inputdata[0]
		wref=inputdata[1]

		rms = np.zeros((numr,nums))+1e30
		best_indices_array = np.zeros((nums, numref), dtype=int)  # New change


		for i in range(0,nums,1):
			if i%10 ==0:print("Image: ", i)
			imgsci = wsci[i,:]
			for j in range(0,numr,1):
				imgref = wref[j,:,:]
				diffa = zscalediff(imgsci,imgref,wm)
				rms[j,i] = np.std(diffa[wm])/np.std(imgsci[wm])
				if str(inputdata[2]['lambdas'][i])!=str(inputdata[3]['lambdas'][j]):
					rms[j,i] = 100000000
#				if str(inputdata[2]['stokes'][i])!=str(inputdata[3]['stokes'][j]):
#					rms[j,i] = 100000000
#				if str(inputdata[2]['polfilts'][i])!=str(inputdata[3]['polfilts'][j]):
#					rms[j,i] = 100000000

			oa = np.argsort((rms[:,i]))[0:numref] # ordena las rsma de menor a mayor para ocupar los mejor 16 como referencia
			best_indices_array[i] = oa
		oc = best_indices_array
		print("Finished cross match rms calcs.")

	elif len(inputdata)==2:

		numr = np.shape(inputdata[1])[0] # number of ref images
		wref=inputdata[1]

		rms = [None] * numr

		imgscimed = inputdata[0]    #ONE MEDIAN SCIENCE IMAGE ONLY
		for j in range(0,numr,1):
			imgref = wref[j,:,:]
			diffa = zscalediff(imgscimed,imgref,wm)
			rms[j] = np.std(diffa[wm])/np.std(imgscimed[wm])  # saves rms of the diff between each image and the median in region 
#			print(j,rms[j])
	
		oa = np.argsort((rms))  # ordena las rsma de menor a mayor para ocupar los mejor 16 como referencia
		oc = oa[0:numref] # selecciona las mejor 16 ref
		
	else: print('Problema con rms')

	
	return oc

# numref = 16 # numero de referencias a seleccionar
def loci_operation(image, cubeR_submed, wm):

	simg = image
	ref = cubeR_submed
	ns1=(simg.shape[0]*simg.shape[1])
	ns2=(ref.shape[0],ref.shape[1]*ref.shape[2])
	wm1d = wm.reshape(ns1)
	simg1d = simg.reshape(ns1)
	ref1d = ref.reshape(ns2)
	#junk,cf = sublcomb1dm(simg1d[wm1d],ref1d[:,wm1d])
	junk,cf = sublcomb1d(simg1d[wm1d],ref1d[:,wm1d],nomedsub=True)
	refnew = basic.lcomb(cubeR_submed,cf)

	return refnew

def loci2(image, cubeR_submed, w1, w2):

    #rgnfac = np.std(image[w1 != 0]) / np.std(image[w2 != 0])
    rgnfac = rgn_fac(image, w1, w2)
    
    simg = image.copy()
    ref = cubeR_submed.copy()

    simg[w1] /= rgnfac
    ref[:,w1] /= rgnfac

    wm = w1 | w2
    
    ns1=(simg.shape[0]*simg.shape[1])
    ns2=(ref.shape[0],ref.shape[1]*ref.shape[2])
    wm1d = wm.reshape(ns1)
    simg1d = simg.reshape(ns1)
    ref1d = ref.reshape(ns2)
    #junk,cf = sublcomb1dm(simg1d[wm1d],ref1d[:,wm1d])
    junk,cf = sublcomb1d(simg1d[wm1d],ref1d[:,wm1d],nomedsub=True)
    refnew = basic.lcomb(cubeR_submed,cf)
    #refnew[w1] *= rgnfac

    return refnew

def rgn_fac(image, w1, w2):
    # Calculate RMS for each masked region
    rms1 = np.std(image[w1 != 0])
    rms2 = np.std(image[w2 != 0])
    
    # Calculate the number of pixels in each mask
    npix1 = np.sum(w1 != 0)
    npix2 = np.sum(w2 != 0)
    
    # Calculate the product of RMS and number of pixels for each mask
    p1 = rms1*rms1*npix1
    p2 = rms2*rms2*npix2
        
    #print(rms1, rms2)
    #print(npix1, npix2) 
    #print(p1, p2) 
    #print(p1/p2)
    return p1/p2

#def loci2(simg, refs, w1, w2):
#    rgnfac = np.std(simg[w1 != 0]) / np.std(simg[w2 != 0])
#
#    simg[w1] /= rgnfac
#    refs[:,w1] /= rgnfac
#
#    wm1d = w1.ravel()  # Convert w1 to 1D index mask
#    simg1d = simg.ravel()
#    ref1d = refs.reshape((refs.shape[0], -1))  # Reshape refs to 2D array

    # Perform combined subtraction and normalization
#    junk, cf = sublcomb1d(simg1d[wm1d], ref1d[:, wm1d], nomedsub=True)
#    refnew = basic.lcomb(refs, cf)
#
#    refnew[w1] *= rgnfac
#    simg[w1] *= rgnfac
#    refs[:,w1] *= rgnfac
#
#    return refnew


