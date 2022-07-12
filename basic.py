import os
import glob
import numpy as np
import numpy
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import rankdata
from scipy.ndimage import gaussian_filter
import statsmodels
from scipy import ndimage, misc
import scipy.io as sio
import numpy

def compute_annulus_values(cube, param):
    # Create meshgrid coordinates to construct annulus on
#    print(cube.shape)
    x = np.arange(0, cube.shape[-1])
    y = np.arange(0, cube.shape[-2])
    xm, ym = np.meshgrid(x, y)

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
        start_angle = np.mod(start_angle, 360)
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
        coord_x_tot = np.append(coord_x_tot, coord_x).astype(np.int)
        coord_y_tot = np.append(coord_y_tot, coord_y).astype(np.int)
    # Determine values
    values_annulus = cube[..., coord_y_tot, coord_x_tot]
    # Create map with annulus coordinates
    frame_annulus = np.zeros(cube.shape[-2:], dtype = np.float32)
    frame_annulus[coord_y_tot, coord_x_tot] = 1.0

    return values_annulus, frame_annulus



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
    print("SNMAP is incorrect")
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

def zscalediff(img1,img2,wm,*rms):
#Image 1 : Total intensity image
#Image 2: Pol image
#Wm : location in which you want to compute the values in a given annulus. Zero for regions outside the annulus. one for values inside the annulus.

    b = img2
    c = (np.shape(b))
    xc = (c[0]-1)/2
    yc = (c[1]-1)/2
    pos = (np.ndarray.flatten(wm))
    if wm.any() == 0:
        a = img1a
        b = b
    else:
        
        img1a = np.ndarray.flatten(img1)
        a = img1a[pos==1]
        b = np.ndarray.flatten(b)
        b = b[pos==1]
    meda = np.median(a)   
    a = a-meda
    medb = np.median(b)
    b = b-medb
    fac = np.sum(a*b)/np.sum(b*b)
    mimg = img2*fac+medb
    diff=img1-mimg
    diff = np.ndarray.flatten(diff)
    diff=diff[pos==1]
    mincnt =  np.std(diff)/10.0
    if rms!=0:
        rms = np.std(diff)/np.std(img1)
    return rms
    
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
        a = img1a
        b = b
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
    ez = np.shape(e)[0]
    dz = np.shape(e)[1]
    a = im1

    med0 = np.median(a)
    a -= np.median(a)
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
