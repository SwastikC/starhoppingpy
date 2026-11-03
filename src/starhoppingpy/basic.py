import os
import glob
import numpy as np
import numpy
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import rankdata
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import scipy.io as sio

def fit_poly2d(z, degree=2, regularization=0.1):
    
    #z is a 2D array, like an image.
    x = np.linspace(0, 1, z.shape[0])
    y = np.linspace(0, 1, z.shape[1])
    xx, yy = np.meshgrid(x, y, indexing='ij')
    z_flat = z.ravel()

    # Calculate median and MAD for outlier detection
    median_z = np.median(z_flat)
    mad_z = np.median(np.abs(z_flat - median_z))
    lower_bound = median_z - 2 * mad_z * 1.4826  # 1.4826 factor to make it equivalent to std deviation
    upper_bound = median_z + 2 * mad_z * 1.4826

    # Print median, MAD, bounds
    print(f"Median of Z: {median_z}, MAD of Z: {mad_z}")
    print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

    # Mask to identify values within two MADs from the median
    inlier_mask = (z_flat >= lower_bound) & (z_flat <= upper_bound)
    z_inliers = z_flat[inlier_mask]
    xx_inliers = xx.ravel()[inlier_mask]
    yy_inliers = yy.ravel()[inlier_mask]

    # Print information about inliers
    print(f"Total number of points: {z_flat.size}, Number of inliers: {z_inliers.size}")

    # Prepare the matrix of powers for inliers
    cols = [np.ones(z_inliers.shape)]
    for d in range(1, degree + 1):
        cols.extend([xx_inliers**d, yy_inliers**d])
        for i in range(1, d):
            cols.append((xx_inliers**i) * (yy_inliers**(d-i)))

    A = np.vstack(cols).T
    identity = regularization * np.eye(A.shape[1])  # Regularization term

    # Solve the linear least squares with regularization for inliers
    coeffs, _, _, _ = np.linalg.lstsq(A.T @ A + identity, A.T @ z_inliers, rcond=None)

    # Print the polynomial coefficients
    print(f"Polynomial Coefficients: {coeffs}")

    # Reconstruct the fitted surface over the entire grid
    poly_surface = np.zeros_like(z)
    index = 0
    for d in range(degree + 1):
        for i in range(d + 1):
            poly_surface += coeffs[index] * np.power(xx, i) * np.power(yy, d - i)
            index += 1

    return poly_surface

def compute_annulus_values(cube, coord):
    # Create meshgrid coordinates to construct annulus on
#    print(cube.shape)

    x = np.arange(0, cube.shape[-1])
    y = np.arange(0, cube.shape[-2])
    xm, ym = np.meshgrid(x, y)
    sz =  np.shape(cube)[1] # number of pixels 
    cc = (sz-1)/2.0 # calculates position of center

    param=(cc,cc,coord[0],coord[1],0,360) # coord_center_x, coord_center_y, inner_radius, outer_radius, start_angle, end_angle

    # Create empty arrays for x- and y-coordinates
    coord_x_tot = np.array([])
    coord_y_tot = np.array([])
#    print(coord_x_tot,coord_y_tot)
    # Make sure param is a list of tuples
    if type(param) == tuple:
        param = [param]

    for param_sel in param:
        coord_center_x, coord_center_y, inner_radius, outer_radius, start_angle, end_angle = param_sel
#        print(coord_center_x, coord_center_y, inner_radius, outer_radius, start_angle, end_angle)
        start_angle = np.mod(start_angle, 360)  # mod returns remainder of division e.g. 7/5 = 2
        end_angle = np.mod(end_angle, 360)

        # Of each pixel calculate radius and angle in range [0, 360)
        radius = np.sqrt((xm - coord_center_x)**2 + (ym - coord_center_y)**2)
        angle = np.mod(np.rad2deg(np.arctan2(ym - coord_center_y, xm - coord_center_x)), 360)
        # Select pixels that satisfy provided requirements
        if start_angle < end_angle:
            coord_y, coord_x = np.nonzero(np.logical_and(np.logical_and(radius >= inner_radius, radius < outer_radius),
                                                         np.logical_and(angle >= start_angle, angle < end_angle)))
            
        else:
            coord_y1, coord_x1 = np.nonzero(np.logical_and(np.logical_and(radius >= inner_radius, radius < outer_radius),
                                                           np.logical_and(angle >= start_angle, angle < 360)))
            coord_y2, coord_x2 = np.nonzero(np.logical_and(np.logical_and(radius >= inner_radius, radius < outer_radius),
                                                           np.logical_and(angle >= 0, angle < end_angle)))
            coord_y, coord_x = np.hstack([coord_y1, coord_y2]), np.hstack([coord_x1, coord_x2])
        # Append coordinates to final coordinate arrays
        coord_x_tot = np.append(coord_x_tot, coord_x).astype(int)
        coord_y_tot = np.append(coord_y_tot, coord_y).astype(int)
    # Determine values
    values_annulus = cube[..., coord_y_tot, coord_x_tot]
    # Create map with annulus coordinates
    frame_annulus = np.zeros(cube.shape[-2:], dtype = np.float32)
    frame_annulus[coord_y_tot, coord_x_tot] = 1.0

    return values_annulus, frame_annulus



def get_mask(cube, r1, r2):
    # Create meshgrid coordinates to construct annulus on
#    print(cube.shape)

    x = np.arange(0, cube.shape[-1])
    y = np.arange(0, cube.shape[-2])
    xm, ym = np.meshgrid(x, y)
    sz =  np.shape(cube)[1] # number of pixels 
    cc = (sz-1)/2.0 # calculates position of center

    param=(cc,cc,r1,r2,0,360) # coord_center_x, coord_center_y, inner_radius, outer_radius, start_angle, end_angle

    # Create empty arrays for x- and y-coordinates
    coord_x_tot = np.array([])
    coord_y_tot = np.array([])
#    print(coord_x_tot,coord_y_tot)
    # Make sure param is a list of tuples
    if type(param) == tuple:
        param = [param]

    for param_sel in param:
        coord_center_x, coord_center_y, inner_radius, outer_radius, start_angle, end_angle = param_sel
#        print(coord_center_x, coord_center_y, inner_radius, outer_radius, start_angle, end_angle)
        start_angle = np.mod(start_angle, 360)  # mod returns remainder of division e.g. 7/5 = 2
        end_angle = np.mod(end_angle, 360)

        # Of each pixel calculate radius and angle in range [0, 360)
        radius = np.sqrt((xm - coord_center_x)**2 + (ym - coord_center_y)**2)
        angle = np.mod(np.rad2deg(np.arctan2(ym - coord_center_y, xm - coord_center_x)), 360)
        # Select pixels that satisfy provided requirements
        if start_angle < end_angle:
            coord_y, coord_x = np.nonzero(np.logical_and(np.logical_and(radius >= inner_radius, radius < outer_radius),
                                                         np.logical_and(angle >= start_angle, angle < end_angle)))
            
        else:
            coord_y1, coord_x1 = np.nonzero(np.logical_and(np.logical_and(radius >= inner_radius, radius < outer_radius),
                                                           np.logical_and(angle >= start_angle, angle < 360)))
            coord_y2, coord_x2 = np.nonzero(np.logical_and(np.logical_and(radius >= inner_radius, radius < outer_radius),
                                                           np.logical_and(angle >= 0, angle < end_angle)))
            coord_y, coord_x = np.hstack([coord_y1, coord_y2]), np.hstack([coord_x1, coord_x2])
        # Append coordinates to final coordinate arrays
        coord_x_tot = np.append(coord_x_tot, coord_x).astype(int)
        coord_y_tot = np.append(coord_y_tot, coord_y).astype(int)
    # Determine values
    values_annulus = cube[..., coord_y_tot, coord_x_tot]
    # Create map with annulus coordinates
    frame_annulus = np.zeros(cube.shape[-2:], dtype = int)
    frame_annulus[coord_y_tot, coord_x_tot] = 1

    return frame_annulus.astype(bool)


def fixpix(imgs, badpix, npix=8, weight=False, noise=False, sigma=None, dc=0, silent=False):
    import numpy as np
    from scipy.ndimage import median_filter
    from astropy.io import fits

    """
    Fix bad pixels in imgs using the badpix mask.
    
    Parameters:
      imgs: a 2D image or 3D image cube (if 3D, axis 0 indexes images)
      badpix: a mask array of the same shape as imgs; 0 indicates a bad pixel,
              nonzero indicates a good pixel.
      npix: number of adjacent good pixels used for correction (default = 8)
      weight: if True, weight adjacent pixels by inverse of distance when averaging.
      noise: if True, replace bad pixels with random noise (using sigma and dc).
      sigma: standard deviation for noise (if noise is True)
      dc: DC level for noise (if noise is True)
      silent: if True, suppress debug output.
    
    Returns:
      outimgs: the corrected image(s) with bad pixels fixed.
      
    Debugging: This function prints input shapes, key statistics, and progress
    information, and it saves an initial badpix map as well as intermediate corrected
    images every 100,000 bad pixels processed.
    """

    #Debug- print input variables.
    #print("Debug: Starting fixpix")
    #print(f"imgs shape: {imgs.shape}, ndim: {imgs.ndim}")
    #print(f"badpix shape: {badpix.shape}, ndim: {badpix.ndim}")
    #print(f"npix: {npix}, weight: {weight}, noise: {noise}, sigma: {sigma}, dc: {dc}, silent: {silent}")
    
    outimgs = imgs.copy()
    
    # Determine if imgs is 2D or 3D.
    if imgs.ndim == 2:
        nimg = 1
        shape2d = imgs.shape
    elif imgs.ndim == 3:
        nimg = imgs.shape[0]
        shape2d = imgs.shape[1:]
    else:
        raise ValueError("Input image must be 2D or 3D.")
    
    nrows, ncols = shape2d
    #print(f"Number of images: {nimg}, image shape: {shape2d}")
    #print("Badpix summary: min =", np.min(badpix), "max =", np.max(badpix), 
    #      "unique values =", np.unique(badpix))
    
    # Save the initial badpix map (without header) for debugging.
    fits.writeto("initial_badpix.fits", badpix, overwrite=True)
    #print("Saved initial badpix map to 'initial_badpix.fits'")
    
    for j in range(nimg):
        if imgs.ndim == 3:
            img = imgs[j, :, :].copy()
            bp = badpix[j, :, :]
        else:
            img = imgs.copy()
            bp = badpix
        print(f"Processing image {j}: shape {img.shape}")
        bp_flat = bp.flatten(order='F')
        wbp = np.where(bp_flat == 0.0)[0]  # indices of bad pixels in Fortran order
        nbad = wbp.size
        if nbad/bp_flat.size > 0.1:
            fits.writeto(f"badpix{j}.fits", bp, overwrite=True)
            fits.writeto(f"img{j}.fits", img, overwrite=True)
            print('More than 10% are bad pixels. Please check. Stopping here.')
            import sys
            sys.exit()
            
        print(f"Image {j}: found {nbad} bad pixels (flattened length: {bp_flat.size})")
        
        for count, idx in enumerate(wbp):
            if count % 10000 == 0:
                print(f"  Processing bad pixel {count}/{nbad}")
            # Save intermediate image every 100,000 pixels.
            #if count % 100000 == 0 and count > 0:
            #    intermediate_fname = f"intermediate_flat_{j}_{count}.fits"
            #    fits.writeto(intermediate_fname, img, overwrite=True)
            #    print(f"    Saved intermediate file: {intermediate_fname}")
            # Convert flat (Fortran order) index to 2D coordinates:
            # For Fortran order, row = index % nrows, col = index // nrows.
            x = int(idx % nrows)
            y = int(idx // nrows)
            
            if noise:
                newval = np.random.randn() * sigma + dc
                img[x, y] = newval
            else:
                dd = 2
                found = False
                # Expand search window until at least npix good pixels are found.
                while not found:
                    x0 = max(0, x - dd)
                    x1 = min(nrows - 1, x + dd)
                    y0 = max(0, y - dd)
                    y1 = min(ncols - 1, y + dd)
                    sub_bp = bp[x0:x1+1, y0:y1+1]
                    sub_img = img[x0:x1+1, y0:y1+1]
                    good_idx = np.where(sub_bp != 0.0)
                    ngood = good_idx[0].size
                    if ngood < npix:
                        dd += 2
                        if dd > max(nrows, ncols):
                            found = True
                    else:
                        found = True
                if ngood < npix:
                    continue
                good_rows = good_idx[0] + x0
                good_cols = good_idx[1] + y0
                dists = np.sqrt((good_rows - x)**2 + (good_cols - y)**2)
                good_values = sub_img[good_idx]
                order = np.argsort(dists)
                dists_sorted = dists[order]
                values_sorted = good_values[order]
                unique_dists = np.unique(dists_sorted)
                numu = unique_dists.size
                if numu == 0:
                    newval = img[x, y]
                elif numu == 1:
                    idx1 = 0
                    idx2 = 0
                elif numu == 2:
                    idx1 = 0
                    idx2 = np.where(dists_sorted == unique_dists[1])[0][0]
                else:
                    idx1 = np.where(dists_sorted == unique_dists[1])[0][0]
                    idx2 = np.where(dists_sorted == unique_dists[-1])[0][0]
                if numu >= 1:
                    newval = (values_sorted[idx1] + values_sorted[idx2]) / 2.0
                else:
                    newval = img[x, y]
                img[x, y] = newval
        if nimg == 1:
            outimgs = img
        else:
            outimgs[j, :, :] = img
    return outimgs



def radial_profile(data, center):
    y,x = np.indices((data.shape)) # first determine radii of all pixels
    r = np.sqrt((x-center[0])**2+(y-center[1])**2)
    ind = np.argsort(r.flat) # get sorted indices
    sr = r.flat[ind] # sorted radii
    sim = data.flat[ind] # image values sorted by radii
    ri = sr.astype(np.int32) # integer part of radii (bin size = 1)
    # determining distance between changes
    deltar = ri[1:] - ri[:-1] # assume all radii represented
    rind = np.where(deltar)[0] # location of changed radius
    nr = rind[1:] - rind[:-1] # number in radius bin
    csim = np.cumsum(sim, dtype=np.float64) # cumulative sum to figure out sums for each radii bin
    tbin = csim[rind[1:]] - csim[rind[:-1]] # sum for image values in radius bins
    radialprofile = tbin/nr # the answer
    return radialprofile


def median_rms_calc(diff,sci):
    sz =  np.shape(sci)[1]
    cc = (sz-1)/2.0
    param =(cc,cc,6,10,0,360)
    xsimg, ysimg = compute_annulus_values(sci, param)
    xdimg, ydimg = compute_annulus_values(diff, param)
    medsci = np.median(xsimg)
    medref = np.median(xdimg)
    median =medref/medsci
    rms = np.std(xdimg)/np.std(xsimg)
    return median,rms


def map4(sci,ref,center):
    sz =  np.shape(sci)[1]
    cc = (sz-1)/2.0
    param =(cc,cc,6,10,0,360)   #(cen_x,cen_y,innerrad,outerrad,startann,endann)
    xsimg, ysimg = compute_annulus_values(sci, param)
    xrimg, yrimg = compute_annulus_values(ref, param)
    param1 =(cc,cc,60,90,0,360)
    xwm0, ywm0 = compute_annulus_values(sci, param1)
    xwm1, ywm1 = compute_annulus_values(ref, param1)
    simg = sci.copy()
    simg-= np.median(xwm0)
    rimg = ref.copy()
    rimg-= np.median(xwm1)
    fs = np.median(xsimg)
    fr = np.median(xrimg)
    fs = np.median(simg[ysimg==1])
    fr = np.median(rimg[ysimg==1])
    x = rimg[ysimg==1]
    temp4 = rimg/fr*fs
    diff2 = simg-temp4
    mimg = temp4
    xdiff, ydiff = compute_annulus_values(diff2, param1)
    diff2 -= np.median(xdiff)
    return simg,mimg,diff2
    
def anglediff(a1,a2):
    b1 = a1%360  
    b2 = a2%360
    off1 = (360-b2)+b1
    off2 = (360-b1)+b2
    off3 = abs(b1-b2)
    val = (min([off1,off2,off3]))
    return val
    
def pdf_creator():
    imagelist = glob.glob('test*.jpeg')
    imagelist = np.sort(imagelist)
    from fpdf import FPDF  # optional dependency
    pdf = FPDF()

    for image in imagelist:
    	pdf.add_page()
    	pdf.image(image,x=50,y=50)
    pdf.output("radprof.pdf", "F")
	
def diskmap(img):
    temp = img
    temp[temp<385]=0
#    plt.imshow(a,vmin=0,vmax=1000)
#    plt.colorbar()
    return temp
    
def rmsmap(img,xc,yc):
    a = img
    b = np.shape(a)
    sigma = robust_sigma(a, zero=0)
    sz1 = b[0]
    sz2 = b[1]
    map = np.zeros(b)*sigma
    orad=sz1/2-10
    dia=5
    box = 15
    res = 3
    s = int(box/2)
    print(s)
    for x in range(0,sz1-res,res):
        for y in range(0,sz2-res,res):
            if x-s >= 0 :
                xl = x-s 
            else:
                xl = 0
            if y-s >= 0 :
                yl = y-s 
            else:
                yl = 0
            if x+s <= sz1:
                xh = x+s-1
            if y+s <= sz2:
                yh = y+s-1
            rms = robust_sigma(a[yl:yh,xl:xh])
            map[y:y+res,x:x+res] = rms
    map[map < sigma] = sigma
        

    return map

def snmap(img,xc,yc):
    a = img
    b = np.shape(a)
    sigma = robust_sigma(a, zero=0)
    sz1 = b[0]
    sz2 = b[1]
    map = np.zeros(b)*sigma
    orad=sz1/2-10
    dia=5
    box = 15
    res = 3
    s = int(box/2)
    print(s)
    for x in range(0,sz1-res,res):
        for y in range(0,sz2-res,res):
            if x-s >= 0 :
                xl = x-s 
            else:
                xl = 0
            if y-s >= 0 :
                yl = y-s 
            else:
                yl = 0
            if x+s <= sz1:
                xh = x+s-1
            if y+s <= sz2:
                yh = y+s-1
            rms = robust_sigma(a[yl:yh,xl:xh])
            map[y:y+res,x:x+res] = rms
    map[map < sigma] = sigma
        

    return map

def robust_sigma(in_y, zero=0):
   """
   Calculate a resistant estimate of the dispersion of
   a distribution. For an uncontaminated distribution,
   this is identical to the standard deviation.

   Use the median absolute deviation as the initial
   estimate, then weight points using Tukey Biweight.
   See, for example, Understanding Robust and
   Exploratory Data Analysis, by Hoaglin, Mosteller
   and Tukey, John Wiley and Sons, 1983.

   .. note:: ROBUST_SIGMA routine from IDL ASTROLIB.

   :History:
       * H Freudenreich, STX, 8/90
       * Replace MED call with MEDIAN(/EVEN), W. Landsman, December 2001
       * Converted to Python by P. L. Lim, 11/2009

   Examples
   --------
   >>> result = robust_sigma(in_y, zero=1)

   Parameters
   ----------
   in_y: array_like
       Vector of quantity for which the dispersion is
       to be calculated

   zero: int
       If set, the dispersion is calculated w.r.t. 0.0
       rather than the central value of the vector. If
       Y is a vector of residuals, this should be set.

   Returns
   -------
   out_val: float
       Dispersion value. If failed, returns -1.

   """
   # Flatten array
   y = in_y.reshape(in_y.size, )

   eps = 1.0E-20
   c1 = 0.6745
   c2 = 0.80
   c3 = 6.0
   c4 = 5.0
   c_err = -1.0
   min_points = 3

   if zero:
       y0 = 0.0
   else:
       y0 = numpy.median(y)

   dy    = y - y0
   del_y = abs( dy )

   # First, the median absolute deviation MAD about the median:

   mad = numpy.median( del_y ) / c1

   # If the MAD=0, try the MEAN absolute deviation:
   if mad < eps:
       mad = numpy.mean( del_y ) / c2
   if mad < eps:
       return 0.0

   # Now the biweighted value:
   u  = dy / (c3 * mad)
   uu = u*u
   q  = numpy.where(uu <= 1.0)
   count = len(q[0])
   if count < min_points:
       print ("ROBUST_SIGMA: This distribution is TOO WEIRD! Returning", c_err)
       return c_err

   numerator = numpy.sum( (y[q]-y0)**2.0 * (1.0-uu[q])**4.0 )
   n    = y.size
   den1 = numpy.sum( (1.0-uu[q]) * (1.0-c4*uu[q]) )
   siggma = n * numerator / ( den1 * (den1 - 1.0) )

   if siggma > 0:
       out_val = numpy.sqrt( siggma )
   else:
       out_val = 0.0

   return out_val


    
def zscalediffnomedsub(img1,img2,wm,*rms):
#Image 1 : Total intensity image
#Image 2: Pol image
#Wm : location in which you want to compute the values in a given annulus. Zero for regions outside the annulus. one for values inside the annulus.

    b = img2
    c = (np.shape(b))
    xc = (c[0]-1)/2
    yc = (c[1]-1)/2
    pos = (np.ndarray.flatten(wm))
    if wm.any() == 0:
        a = np.ndarray.flatten(img1)
        b = np.ndarray.flatten(b)
    else:
        
        img1a = np.ndarray.flatten(img1)
        a = img1a[pos==1]
        b = np.ndarray.flatten(b)
        b = b[pos==1]
    meda = np.median(a)   
    a = a
    medb = np.median(b)
    b = b
    fac = np.sum(a*b)/np.sum(b*b)
    mimg = img2*fac
    diff=img1-mimg
    diff = np.ndarray.flatten(diff)
    diff=diff[pos==1]
    mincnt =  np.std(diff)/10.0
    if rms!=0:
        rms = np.std(diff)/np.std(img1)
    return rms
    
def sublcomb1d(im1, imc):
    e = imc
    ez = np.shape(e)[0] # number of images?
    dz = np.shape(e)[1] # number of pixel?
    a = im1

    med0 = np.median(a)
    a -= np.median(a)
    s = np.tile(np.median(e,axis=1),(dz,1)).T      # numpy.tile(A, reps): Construct an array by repeating A the number of times given by reps.
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
    
def lcomb(rimc,cf):
    sz = np.shape(rimc)
    sz1 = len(sz)
    temp = rimc*0
    
    if sz1==3:
        a=[sz[1],sz[2],sz[0]]
        b=[sz[1],sz[2]]
        temp = np.zeros(a,dtype=float, order='C')
        out = np.zeros(b,dtype=float, order='C')
        for i in range(0,sz[0],1): 
            temp[:,:,i] = rimc[i,:,:]*cf[i]
        out = np.sum(temp,axis=2)
    else:
        print("write the code for frames :(")
    return out
    
def sublcomb1dm(im1, imc):
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
    
def rms_map(img,wid,sp,n,m):
    #SP - STarting Point
    #N - ENding point
    #width of the annulus
    # M --> tIMES THE STANDARD DEVIATION
    sz =  np.shape(img)[1]
    cc = (sz-1)/2.0
    map = np.zeros([sz,sz])
    r = int((n-sp)/wid)
    cp =sp
    for i in range(1,r+1,1):
        w1 =(cc,cc,cp,cp+wid,0,360)
        xwm0, ywm0 = compute_annulus_values(img,w1)
        map[ywm0==1]=m*np.std(xwm0)
        cp =cp+wid
        print(cp)
    return map


def get_region(img, rin, rout, sz=None, xc=None, yc=None, dd=None, show=None, mask=None, pa0=None, pa1=None, amax=None, bgnd=None, getmask=None):

    if bgnd is not None:
        med = np.median(img)
        std = np.std(img)
        lev = med + 2 * std
        return np.where(img < lev)
  
    if sz is None:
        sz = img.shape[0]

    if xc is None:
        xc = (sz - 1) / 2

    if yc is None:
        yc = (sz - 1) / 2

    if rin is None:
        rin = 5.0

    if rout is None:
        rout = sz / 2 * 1.5

    if dd is None:
        y, x = np.ogrid[-yc:sz-yc, -xc:sz-xc]
        dd = np.sqrt(x**2 + y**2)

    if pa0 is not None:
        pp = np.arctan2(y, x) * 180 / np.pi

        if pa1 is None:
            pa1 = pa0 + 360

        angle_mask = ((pp >= pa0) & (pp < pa1)) if amax is None else (1 - ((pp >= pa0) & (pp < pa1)))
        mask = (dd >= rin) & (dd < rout) & angle_mask

    else:
        mask = (dd >= rin) & (dd < rout)

    w = np.where(mask)
   
    if show:
        temp = img.copy()
        temp[w] += temp.max() * 2

        print('center=', xc, yc)

        plt.imshow(temp, cmap='gray', interpolation='nearest')
        plt.show()
   
    if getmask is not None:
        return mask

    else:
        return w




def bscentrd(img, xgess, ygess, INFO=0):
    """
    Compute the "center of mass" centroid coordinates of a stellar object.

    Parameters
    ----------
    img : 2D numpy array
        Input image.
    xgess, ygess : float
        Approximate x and y stellar center guesses (input follows NumPy convention).
    INFO : int, optional
        If set to 1, prints additional debugging information.

    Returns
    -------
    x_final, y_final : float
        Computed centroid coordinates (x, y). Returns (-1, -1) if unable to centroid.
    """

    import numpy as np
    from scipy.ndimage import uniform_filter
    from skimage.transform import resize

    print(f"Initial guess: x={xgess}, y={ygess}, pixel value={img[int(ygess), int(xgess)]}")

    WIDTH = 7
    EXPAND = 10
    MAX_REPOS = 5  # Maximum repositioning attempts
    reposition_attempts = 0

    while True:
        reposition_attempts += 1
        xguess, yguess = int(round(xgess)), int(round(ygess))

        # Extract a region around the guessed centroid (original image)
        im1 = img[int(yguess - WIDTH / 2): int(yguess + WIDTH / 2),
                  int(xguess - WIDTH / 2): int(xguess + WIDTH / 2)].astype(float)

        # Expand and smooth the image
        im2 = resize(im1, (WIDTH * EXPAND, WIDTH * EXPAND), order=1, mode='reflect', anti_aliasing=True)
        im3 = uniform_filter(im2, size=int(WIDTH / 2))
        im3 = uniform_filter(im3, size=int(WIDTH / 2))

        # Estimate background
        av_val = np.mean(im3)
        bak = np.mean(im3[im3 < av_val])

        if INFO:
            print(f"Background Value: {bak}")

        # Threshold the image
        im4 = (im3 - bak) > 0
        W, C = WIDTH * EXPAND, WIDTH * EXPAND // 2

        # ***** Find peak near center *****
        prevmax, curmax, inarow = 0.0, 0.0, 0
        peak_x_expanded, peak_y_expanded = 0.0, 0.0
        i = 0

        while peak_x_expanded == 0 and i < C / 2:
            i_low, i_high = max(0, C - i), min(W, C + i + 1)
            sub_region = im4[i_low:i_high, i_low:i_high]
            curmax = np.max(sub_region)

            if curmax == prevmax:
                inarow += 1
            else:
                prevmax, inarow = curmax, 0

            if inarow == 5:
                indices = np.nonzero(im4 == curmax)
                if indices[0].size > 0:
                    peak_y_expanded, peak_x_expanded = indices[0][0], indices[1][0]

            i += 1

        if peak_x_expanded == 0:
            print("Unable to Centroid Star: No peak found.")
            return -1, -1

        # Convert expanded coordinates to original image coordinates
        peak_x = xgess + (peak_x_expanded - C) / EXPAND
        peak_y = ygess + (peak_y_expanded - C) / EXPAND

        # Repositioning logic
        step_size = 0.1  # Reduce step size for smoother convergence
        tolerance = max(3, WIDTH // 2)  # Adaptive stopping criterion

        if (abs(peak_x - xgess) > tolerance or abs(peak_y - ygess) > tolerance) and reposition_attempts < MAX_REPOS:
            xgess += step_size * (peak_x - xgess)
            ygess += step_size * (peak_y - ygess)
            print(f"Repositioning to x={xgess}, y={ygess}")
            continue  # Retry with updated guess
        else:
            break  # Stop repositioning if within tolerance

    # ***** Compute center of mass *****
    sttot = np.sum(im1)
    ycen = np.sum(np.arange(WIDTH) * np.sum(im1, axis=1)) / sttot  # IDL X-axis → NumPy Y-axis
    xcen = np.sum(np.arange(WIDTH) * np.sum(im1, axis=0)) / sttot  # IDL Y-axis → NumPy X-axis

    # Compute final centroid coordinates
    x_final = xgess - WIDTH / 2 + xcen
    y_final = ygess - WIDTH / 2 + ycen

    # Final Peak Check in the Original Image
    print(f"Final Centroid at ({x_final:.2f}, {y_final:.2f})")
    if 1 <= x_final < img.shape[1] - 1 and 1 <= y_final < img.shape[0] - 1:
        neighbors = img[int(y_final) - 1:int(y_final) + 2, int(x_final) - 1:int(x_final) + 2]  # 3×3 patch
        neighbors_flat = neighbors.flatten()
        neighbors_flat = np.delete(neighbors_flat, 4)  # Remove center pixel
        peak_value = img[int(y_final), int(x_final)]  # Peak value from original image
        differences = peak_value - neighbors_flat  # Difference from neighbors
        print(f"Peak Value: {peak_value:.2f}")
        print(f"Mean Diff from Neighbors: {np.mean(differences):.3f}")
        print(f"Std Dev of Differences: {np.std(differences):.3f}")

    return  x_final, y_final  # Return in (Y, X) order
    

def esopar(header, keyword):
    """
    Extract a parameter value from an ESO-style FITS header.
    Prints all partial matches while returning only the exact match if found.

    Parameters
    ----------
    header : astropy.io.fits.Header
        The FITS header as an Astropy `Header` object.
    keyword : str
        The keyword to search for.

    Returns
    -------
    str
        The extracted parameter value as a string, or an empty string if not found.
    """
    from astropy.io import fits

    keyword = keyword.strip()  # Ensure no leading/trailing spaces

    # Exact match: return the value
    if keyword in header:
        value = str(header[keyword]).strip()
        return value

    # Partial matches: find all keys that contain the keyword as a substring
    matches = [key for key in header.keys() if keyword in key]

    if matches:
        print(f"Partial matches for '{keyword}':")
        for match in matches:
            print(f"  {match} = {header[match]}")
    
    return ''  # Return empty string if exact match not found
