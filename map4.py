import os
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
import glob
import numpy as np
from fpdf import FPDF 

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
	
	
