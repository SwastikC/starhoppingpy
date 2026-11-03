### All procs list #############

### files
# getfilename(pattern, num): get filename which is closest match to pattern + number
# getfilename('dustkapscatmat_*.inp', 100) -> dustkapscatmat_1.18e+02.inp


### strings
# getwildpart(pattern, string): get part of string which matches the wildcard part
# getwildpart('dustkapscatmat_*.inp', 'dustkapscatmat_2.40e+02.inp') -> 2.40e+02
# a2str([1,2,3]) -> '1, 2, 3'


#### Imports #############################
from astropy.io import fits
from astropy.constants import c, h, k_B
from astropy.modeling.models import BlackBody
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import astropy.units as u
from scipy import ndimage as nd
from scipy.ndimage import rotate, median_filter, uniform_filter
from astropy.stats import mad_std
from collections import OrderedDict
from datetime import datetime
import glob
import numpy as np
import re
import sys
import os
import shutil

### Misc ########################


#get the var name (instead of value)
def zname(variable, scope):
    return [name for name, val in scope.items() if val is variable][0]

#create a copy of the file  with a name like filename_22feb2023_150216.txt, where 22feb2023 represents the last modified date.
def savecopy(filename): ### NOT TESTED !!!!!!!!!!!!!!!!!!!!!!!!! ###########
    # Get the last modified time of the file
    mod_time = os.path.getmtime(filename)
    mod_date = datetime.fromtimestamp(mod_time)
    
    # Format the date as DDmonYYYY_HHMMSS
    date_suffix = mod_date.strftime('%d%b%Y_%H%M%S').lower()
    
    # Split the file name into its base and extension
    base, ext = os.path.splitext(filename)
    
    # Construct the new file name with the date suffix
    new_filename = f"{base}_{date_suffix}{ext}"
    
    # Copy the file to the new name
    shutil.copy2(filename, new_filename)

    
def info(var_list):
    for v in var_list:
        print(f"Val: {v}, Type: {type(v)}, Shape: {np.shape(v)}", flush=True)


# when given a non-uniform set of varnames in a text file:
# rad1, rad2, rad3, mass, rdisk1, rdisk2, rdisk3, hs
# and a list of values
# prams = [1,2,3,10,100,200,300,1000]
# Make array/dict assignments so that solo vars
#        like 'mass' above is also triplicated like:
# {'rad': [1, 2, 3], 'mass': [10, 10, 10], 'rdisk': [100, 200, 300], 'hs': [1000, 1000, 1000]}


def assn_var_sets(array):
    var_names = load('param_names.save').tolist()
    
    # create ordered dictionaries to store variable names and values
    var_dict = OrderedDict()
    for name in var_names:
        var_dict[name] = None
    
    # assign variable values based on variable names
    for i, val in np.ndenumerate(array):
        name = var_names[i[0]]
        var_dict[name] = val
    
    group_names = [x.rstrip('0123456789') for x in var_names]
    # get groups of variable names and their corresponding values
    var_groups = {}
    group_size = max([int(xs[-1]) if xs[-1].isnumeric() else 0 for xs in var_names])
    for name in var_names:
        group_name = name.rstrip('0123456789')
        ginx = int(name[-1])-1 if name[-1].isnumeric() else 0 #make sure ginx > 0 for solo vars
        if group_name not in var_groups:
            var_groups[group_name] = [var_dict[name]] * group_size
        var_groups[group_name][ginx] = var_dict[name]
    
    return var_groups


### plotting ###################


def zplot(x, y, legend=None, ylog=False, xlog=False, xtit='', ytit='', yinv=False, dotsize=3, xmin=None):
    """
    Plots variable 'x' on the X-axis against variable 'y' on the Y-axis, supporting multiple datasets.
    
    Parameters:
    x (iterable): Data for the X-axis. Can be a 1D or list of 1D arrays.
    y (iterable): Data for the Y-axis. Can be a 1D or list of 1D arrays.
    legend (iterable): Labels for the legend corresponding to each x, y pair.
    ylog (bool): If True, sets the Y-axis to logarithmic scale.
    xtit (str): Title for the X-axis.
    ytit (str): Title for the Y-axis.
    yinv (bool): If True, inverts the Y-axis.
    """

    plt.figure(figsize=(8, 6))  # Specifies the figure size
    
    # Determine if x and y are lists of arrays (2D)
    if isinstance(x[0], np.ndarray) or isinstance(y[0], np.ndarray):
        for i, (xi, yi) in enumerate(zip(x, y)):
            # Check if legend is provided and has enough entries
            if legend and i < len(legend):
                plt.plot(xi, yi, marker='o', label=legend[i], ms=dotsize)
            else:
                plt.plot(xi, yi, marker='o', ms=dotsize)
    else:
        # Plot single dataset
        plt.plot(x, y, marker='o', label=legend[0] if legend else None)
        
    if ylog:
        plt.yscale('log')

    if xlog:
        plt.xscale('log')
    
    plt.xlabel(xtit)
    plt.ylabel(ytit)
    if yinv:
        plt.gca().invert_yaxis()

    if legend:
        plt.legend()

    plt.grid(True)
    plt.xlim(xmin=xmin, xmax=None)
    plt.show()

def zplot2(x, y, legend=None, ylog=False, xlog=False, xtit='', ytit='', yinv=False, 
           dotsize=3, xmin=None, yerr=None, yerrn=None, xerr=None, xerrn=None, linestyle=None):
    """
    Enhanced plotting function with internal helper functions to improve readability.
    """
    
    def prepare_error_bars(err, errn):
        """Prepare positive and negative error bars."""
        if is_error_zero_or_none(err):
            return None
        return [errn if errn is not None else err, err]
    
    def is_error_zero_or_none(err):
        """Check if the error array is None or filled with zeros."""
        return err is None or np.all(np.asarray(err) == 0)
    
    def get_current_setting(value, index, default=None):
        """Get current setting from list or return a single value."""
        return value[index] if value is not None and isinstance(value, list) else value

    plt.figure(figsize=(8, 6))

    
    # Check if 'x' and 'y' contain multiple datasets.
    # If true, it indicates that 'x' and 'y' are lists (or similar iterables) of arrays, each array representing a separate dataset.
    if isinstance(x[0], np.ndarray) or isinstance(y[0], np.ndarray):
        # If 'x' and 'y' contain multiple datasets, iterate through each dataset.
        # This index 'i' is used for accessing corresponding attributes (like error bars, dot sizes, etc.) for each dataset.
        for i, (xi, yi) in enumerate(zip(x, y)):
            current_dotsize = get_current_setting(dotsize, i, 3)
            current_linestyle = get_current_setting(linestyle, i, 'None')
            current_xerr = prepare_error_bars(get_current_setting(xerr, i), get_current_setting(xerrn, i))
            current_yerr = prepare_error_bars(get_current_setting(yerr, i), get_current_setting(yerrn, i))
            
            plt.errorbar(xi, yi, xerr=current_xerr, yerr=current_yerr, fmt='o',
                         linestyle=current_linestyle, label=get_current_setting(legend, i),
                         ms=current_dotsize, capsize=5)
    else:
        plt.errorbar(x, y, xerr=prepare_error_bars(xerr, xerrn), yerr=prepare_error_bars(yerr, yerrn), fmt='o',
                     linestyle=linestyle if linestyle else 'None', label=legend[0] if legend else None,
                     ms=dotsize if isinstance(dotsize, (int, float)) else dotsize[0], capsize=5)

    if ylog:
        plt.yscale('log')
    if xlog:
        plt.xscale('log')
    
    plt.xlabel(xtit)
    plt.ylabel(ytit)

    if yinv:
        plt.gca().invert_yaxis()

    if legend:
        plt.legend()
        
    plt.grid(False)  # This line turns off the grid display
    
    if xmin is not None:
        plt.xlim(xmin=xmin)
    plt.show()

### printing ###################

def format_limits(median, upper_limit, lower_limit):
    formatted_median = "{:.3g}".format(median)
    
    upper_percentage = ((upper_limit - median) / median) * 100
    formatted_upper_limit = "{:.2g}%".format(upper_percentage)
    
    lower_percentage = ((median - lower_limit) / median) * 100
    formatted_lower_limit = "{:.2g}%".format(lower_percentage)
    
    return formatted_median, formatted_upper_limit, formatted_lower_limit


### distributions ###############

def asym_dist(x0, el_x, eu_x, n_values):
    """
    Generate a distribution of values with asymmetric Gaussian uncertainties around x0.
    
    Parameters:
    x0 (float): Central value of x.
    el_x (float): Standard deviation for the Gaussian distribution on the lower side.
    eu_x (float): Standard deviation for the Gaussian distribution on the upper side.
    n_values (int): Number of values to generate.
    
    Returns:
    tuple: A tuple containing the distribution of generated values and the number of generated values.
    """
    # Calculate the number of values for each side based on a 50/50 split or any other desired ratio
    n_lower = n_values // 2
    n_upper = n_values - n_lower
    
    # Generate lower side values
    lower_distribution = x0 - np.abs(np.random.normal(0, el_x, n_lower))
    
    # Generate upper side values
    upper_distribution = x0 + np.abs(np.random.normal(0, eu_x, n_upper))
    
    # Combine both distributions
    combined_distribution = np.concatenate([lower_distribution, upper_distribution])
    
    return combined_distribution, len(combined_distribution)

### arrays and dicts ######################

def a2str(array) :
    #example:  a2str([1,2,3]) -> '1, 2, 3'
    x1=array.tolist()
    x = ['{:.3e}'.format(item) for item in x1]
    return ', '.join(x)

def str2a(string):
    #convert a comma-separated string list into an array of strings.
    #ex: a2str('aa, bb, cc') -> ['aa','bb','cc']
    return [x.strip() for x in string.split(',')]

def str2a2(string,str=0):
    """
    Convert a delimiter-separated string list into an array of strings.
    Automatically recognizes common delimiters (space, comma, semicolon, tab, newline).
    
    Parameters:
    string (str): A string containing delimiter-separated values.
    
    Returns:
    list: A list of strings split based on the detected delimiter.
    """
    import re
    
    # Try to detect the delimiter by using regular expressions to find common delimiters
    delimiter = re.search(r'[,;\s\t\n]', string)
    if delimiter:
        delimiter = delimiter.group()
    else:
        # Default to comma if no common delimiter is detected
        delimiter = ','

    if str == 0:
        return np.array([float(x.strip()) for x in string.split(delimiter)])
    else:
        return [x.strip() for x in string.split(delimiter)]


#this version allows sideways or transpose table printing
#this one works but the col spacing was too big and had no default formatting.
#Named this _old
#New version is named 'forprint', but still testing. So I keep this here.
def forprint_old(*arrays, filename=None, transpose=False):
    # Determine the maximum width of each column/row
    max_widths = []
    for row_or_col in (zip(*arrays) if transpose else arrays):
        max_val = max([len(str(val)) for val in row_or_col])
        max_widths.append(max_val)

    # Create the output string
    output = ''
    for i in range(max([len(a) for a in arrays])):
        for j, a in enumerate((zip(*arrays) if transpose else arrays)):
            if i < len(a):
                val = a[i]
                if isinstance(val, (int, float)):
                    val = f'{val:.5g}'
                    if len(val) > max_widths[j]:
                        val = val[:max_widths[j]-3] + '...'
                    output += str(val).rjust(max_widths[j]) + '  '
                else:
                    if transpose:
                        output += str(val).rjust(max_widths[j]) + '  '
                    else:
                        output += str(val).ljust(max_widths[j]) + '  '
            else:
                output += " " * max_widths[j] + '  '
        output += '\n'

    if filename:
        with open(filename, 'a') as f:
            f.write(output)
    else:
        print(output)

def forprint(*arrays, filename=None, transpose=False):
    # Transpose the arrays if needed
    arrays = list(zip(*arrays)) if transpose else arrays

    # Determine the maximum width of each column
    max_widths = []
    for col in arrays:
        max_widths.append(max([len(f'{val:.5g}') if isinstance(val, (int, float)) else len(str(val)) for val in col]))

    # Create the output string
    output = ''
    for i in range(max([len(a) for a in arrays])):
        row = []
        for j, col in enumerate(arrays):
            if i < len(col):
                val = col[i]
                if isinstance(val, (int, float)):
                    val = f'{val:.5g}'  # Format numbers with 5 significant digits
                    row.append(val.rjust(max_widths[j]))  # Right-align numbers
                else:
                    row.append(str(val).ljust(max_widths[j]))  # Left-align strings
        # Join the row elements with exactly two spaces between them
        output += '  '.join(row) + '\n'

    # Output to file or print
    if filename:
        with open(filename, 'a') as f:
            f.write(output)
    else:
        print(output)
        

def bracket(arr,delta=1e-5):
    # repeat each element of the original array twice
    arr_new = np.repeat(arr, 2)

    # multiply every other element by 1.01 and the remaining elements by 0.99
    arr_new[1::2] *= 1+delta
    arr_new[::2] *= 1-delta

    return arr_new



### dictionaries ################

#gets all groups of dict elements where the given keys match
#as groups of indices (not values).
#example: [ [1,5,6], [4,7,0], [9, 2, 13], ....]
def keygroups(dict1, keys):
    groups = []
    checked_indices = set()

    # Check if keys exist in dict1 and have the same length
    key_lengths = {key: len(dict1[key]) for key in keys if key in dict1}
    if len(set(key_lengths.values())) != 1:
        print("Keys must have the same length in the dictionary.")
        return groups

    max_length = key_lengths[keys[0]]  # Use the length of the first key

    for i in range(max_length):
        if i in checked_indices:
            continue
        
        current_group = [i]
        checked_indices.add(i)

        for j in range(i + 1, max_length):
            match = True
            for key in keys:
                if dict1[key][i] != dict1[key][j]:
                    match = False
                    break
            
            if match:
                current_group.append(j)
                checked_indices.add(j)

        if len(current_group) > 1:  # Only add valid groups with more than one index
            groups.append(current_group)

    return groups

# groups are arrays of indices where the dict is the same for some keys
# this func returns the values of the first group only, thus the unique value combos
def firstkeyvals(dictionary, keys, groups):
    values = []
    for group in groups:
        first_index = group[0]
        value = {key: dictionary[key][first_index] for key in keys}
        values.append(value)
    return values

def wherekeys(dictionary, keyvalues):
    matching_indices = []
    # Determine the length of the arrays based on the first key's values
    array_length = len(dictionary[next(iter(dictionary))])
    
    for i in range(array_length):
        match = True
        for key in keyvalues:
            # Check if the current index i is within bounds
            if i >= len(dictionary[key]):
                match = False
                break
            if dictionary[key][i] != keyvalues[key]:
                match = False
                break
        if match:
            matching_indices.append(i)
    
    return matching_indices


def where2xy(w, sz):
    """
    Converts a flattened index array (from np.where) into (x, y) coordinates.

    Parameters
    ----------
    w : array_like
        1D array of indices (e.g., from np.where).
    sz : int
        Width of the 2D array (number of columns).

    Returns
    -------
    xcors : np.ndarray
        X-coordinates of the input indices.
    ycors : np.ndarray
        Y-coordinates of the input indices.
    xmin, ymin : int
        Minimum x and y values in the coordinate list.
    xmax, ymax : int
        Maximum x and y values in the coordinate list.
    """
    if w.size == 0:
        return np.array([]), np.array([]), None, None, None, None  # Return empty if no points

    xcors = w % sz  # X-coordinates
    ycors = w // sz  # Y-coordinates

    xmin, ymin = np.min(xcors), np.min(ycors)
    xmax, ymax = np.max(xcors), np.max(ycors)

    return xcors, ycors, xmin, ymin, xmax, ymax


### files ######################
        
# get filename which is closest match to pattern + number
def getfilename(pattern, num, numpart=0):
    # example:
    # getfilename('dustkapscatmat_*.inp', 100) -> dustkapscatmat_1.18e+02.inp
    # if numpart == 1 then returns the part which matches the number
    
    #path = "./"
    gsA1= []
    gssA1= []
    filesA1= []
    
    for files in glob.glob(pattern):
        x = getwildpart(pattern,files)
        gsA1.append(float(x)) 
        gssA1.append(x) 
        filesA1.append(files)

    delta = 1e-20    
    diffA1 = np.abs((np.array(gsA1)-num)/(num+delta))
    #print(gsA1)
    #print(diffA1)
    minx = np.argmin(diffA1)

    if numpart == 1:
        return gssA1[minx]
    else:
        return filesA1[minx]

### strings ######################

def getwildpart(pattern, string):
    # example:
    # getwildpart('dustkapscatmat_*.inp', dustkapscatmat_2.40e+02.inp) -> 2.40e+02
    
    inx0 = re.search(r'\*', pattern).end()
    #index of end of wildcard match
    inx1 = len(string)-len(pattern[inx0:])
    return string[inx0-1:inx1]


### images #######################

# extract a subimage from an image
def imgext(img,xc=0,yc=0,szn=0):
    
    class MyError(Exception):
        pass

    if xc == 0:
        xc = (img.shape[0]-1)/2.0
    if yc == 0:
        yc = (img.shape[1]-1)/2.0

    xcf = int(xc) #int typing yields floor 
    ycf = int(yc)

    fimg = img
    if xc-xcf !=0 or yc-ycf !=0:
        fimg = nd.shift(fimg, (xc-xcf, yc-ycf), order=3 )
        #zdisp(fimg)

    dx = 0
    if xc-xcf !=0:
        dx = 1

    dy = 0
    if yc-ycf !=0:
        dy = 1
        
    if szn % 2 == 0:
        raise MyError('The parameter SZN has to be odd')
    hn = int((szn-1)/2)

    #print( xcf,ycf,hn)
    #print( xcf-hn+dx, xcf+hn+dx+1, ycf-hn+dy, ycf+hn+dy+1)

    #remember array assignment excludes ending index, thus +1 
    fimg = fimg[ xcf-hn+dx: xcf+hn+dx+1, ycf-hn+dy: ycf+hn+dy+1]

    
    return fimg


#### photometry  ###############################################

#wave is in microns

def zp(wave):
#zp in Janskys
    if not isinstance(wave, np.ndarray):
        wave = np.array(wave)
        
    # pysynphot is optional and needs the CDBS reference files; set PYSYN_CDBS to their location.
    os.environ.setdefault('PYSYN_CDBS', os.path.expanduser('~/trds/'))
    import pysynphot as S
    bb = S.BlackBody(9600.0)
    bb.convert('Jy')
    bb.convert('Micron')
    bb_flux = bb.sample(wave)
    hflux = bb.sample(np.array(2.22))
    hzpflux = 670.0 * u.Jy
    zps = (bb_flux * hzpflux / hflux).value

    return zps

def mag2flux(wave, mag):
#flux in Janskys
    if not isinstance(mag, np.ndarray):
        mag = np.array(mag)

    flux = zp(wave)*10**(-0.4*mag)
    
    return flux

#### polarimetry ###############################################

# Make Q-phi and U-phi images
def imphi(qname,uname):
    
    qmod = (fits.open(qname))[0].data
    umod = (fits.open(uname))[0].data
    x = np.arange(0, qmod.shape[-1])
    y = np.arange(0, qmod.shape[-2])
    xm, ym = np.meshgrid(x, y)
    x_center = 0.5*x[-1]
    y_center = 0.5*y[-1]
    rotation_angle = 0
    phi = np.arctan2((x_center - xm), (ym - y_center)) + np.deg2rad(rotation_angle)
    frame_U = umod
    frame_Q = qmod

    frame_Q_phi = -frame_Q * np.cos(2*phi) - frame_U * np.sin(2*phi)
    frame_U_phi = frame_Q * np.sin(2*phi) - frame_U * np.cos(2*phi) 

    writefits('Q_phi_model.fits',frame_Q_phi)
    writefits('U_phi_model.fits',frame_U_phi)
    
    return

################################################################
# get region as indices within annulus with center xc, yc and radii r1->r2
def whereback(arr, rmsfrac=None):
    import numpy as np
    
    arr = np.asarray(arr)  # Ensure arr is a NumPy array

    if rmsfrac is not None:
        nonzero = (arr != 0)
        if np.any(nonzero):
            tagval = np.std(arr[nonzero]) * rmsfrac
        else:
            tagval = 1e-30
    else:
        tagval = 1e-30

    # Use np.flatnonzero to get flat indices.
    w = np.flatnonzero((arr > -2 * tagval) & (arr < 2 * tagval))
    return w


def getfreq(img_in, fwhm, unsharp=None, low=None, high=None, subsmooth=None):
    """
    Process an image to extract frequency information via Gaussian filtering.
    
    Parameters
    ----------
    img_in : array_like
        Input image.
    fwhm : float
        Parameter used for filtering.
    unsharp : optional, default None
        If provided (non-empty), then output is computed as img_in - dull.
    low : optional, default None
        If not provided or empty, defaults to fwhm * 2.
    high : optional, default None
        If not provided or empty, defaults to fwhm / 2.
    subsmooth : optional, default None
        If provided (non-empty), then output is computed as
        filter_image(img_in - dull, fwhm_gaussian=high).
    
    Returns
    -------
    out : ndarray
        The processed image.
        
    Debug:
      Writes intermediate FITS files:
         - debug_in0.fits: in0 (input as double)
         - debug_sharp.fits: sharp = in0 - filter_image(in0, high)
         - debug_dull.fits: dull = filter_image(in0, low)
         - debug_out_pre_unsharp.fits: initial out computed as in0 - sharp - dull (or via subsmooth)
         - debug_out_post_unsharp.fits: out after applying unsharp (if provided)
         - debug_out_final.fits: out after zeroing out pixels from wback
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from astropy.io import fits

    # Helper: apply a Gaussian filter; convert FWHM to sigma.
    def filter_image(img, fwhm_gaussian):
        sigma = fwhm_gaussian / 2.35482
        return gaussian_filter(img, sigma=sigma)

    # Get indices of non-zero pixels in the input (for later zeroing).
    wback = whereback(img_in)
    
    # Convert input to double precision.
    in0 = np.array(img_in, dtype=float)
    #fits.writeto("debug_in0.fits", in0, overwrite=True, output_verify='ignore')
    #print("Saved debug_in0.fits", flush=True)
    
    # Set defaults for low and high if they are not provided.
    if low is None or (hasattr(low, '__len__') and len(low) == 0):
        low = fwhm * 2.0
    if high is None or (hasattr(high, '__len__') and len(high) == 0):
        high = fwhm / 2.0

    # Compute the "sharp" image.
    sharp = in0 - filter_image(in0, fwhm_gaussian=high)
    #fits.writeto("debug_sharp.fits", sharp, overwrite=True, output_verify='ignore')
    #print("Saved debug_sharp.fits", flush=True)
    
    # Compute the "dull" image.
    dull = filter_image(in0, fwhm_gaussian=low)
    #fits.writeto("debug_dull.fits", dull, overwrite=True, output_verify='ignore')
    #print("Saved debug_dull.fits", flush=True)
    
    # Compute the initial output.
    if subsmooth is not None and (hasattr(subsmooth, '__len__') and len(subsmooth) != 0):
        out = filter_image(in0 - dull, fwhm_gaussian=high)
    else:
        out = in0 - sharp - dull
    #fits.writeto("debug_out_pre_unsharp.fits", out, overwrite=True, output_verify='ignore')
    #print("Saved debug_out_pre_unsharp.fits", flush=True)
    
    # If unsharp is provided, update the output.
    if unsharp is not None and (hasattr(unsharp, '__len__') and len(unsharp) != 0):
        out = in0 - dull
        #fits.writeto("debug_out_post_unsharp.fits", out, overwrite=True, output_verify='ignore')
        #print("Saved debug_out_post_unsharp.fits", flush=True)
    
    # Zero out pixels corresponding to wback.
    if wback.size > 0:  # Ensures wback is not empty before indexing
        out.flat[wback] = 0
    #fits.writeto("debug_out_final.fits", out, overwrite=True, output_verify='ignore')
    #print("Saved debug_out_final.fits", flush=True)
    
    return out



def getrgn_old_oct17_2024(img,xc=0,yc=0,r0=0,r=0, mask=False):

    if xc == 0:
        xc = (img.shape[0]-1)/2.0
    if yc == 0:
        yc = (img.shape[1]-1)/2.0
        
    sx = np.arange(img.shape[0])
    sy = np.arange(img.shape[1])
    x,y=np.meshgrid(sx,sy)                           # x and y are two 2D array with coords 
    dist=np.sqrt((x-xc)**2+(y-yc)**2)                # dist is 2D array giving dist from xc, yc
    if mask == True:
        rgn = np.logical_and(dist>=r0,dist<r)  #rgn is array of indices within annulus r1->r2                  
    else:
        rgn = np.where(np.logical_and(dist>=r0,dist<r))  #rgn is array of indices within annulus r1->r2
    return rgn

# Get region as a boolean mask within an annulus defined by r0 -> r around the image center (or specified center)
def getrgn(img, r0, r, xc=None, yc=None):
    """
    Returns a boolean mask of pixels within an annulus (r0 -> r) centered on (xc, yc).
    
    Parameters
    ----------
    img : 2D numpy array
        Input image.
    r0, r : float
        Inner and outer radii of the annulus.
    xc, yc : float, optional
        Center of the annulus. Defaults to the center of the image.
        
    Returns
    -------
    rgn : np.ndarray (bool)
        2D boolean mask of pixels within the annulus.
    """
    # If center coordinates are not provided, default to the center of the image
    if xc is None:
        xc = (img.shape[1] - 1) / 2.0
    if yc is None:
        yc = (img.shape[0] - 1) / 2.0

    # Create a grid of x and y coordinates
    y, x = np.indices(img.shape)
    
    # Calculate the distance of each point from the center (xc, yc)
    dist = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    # Return a boolean mask for pixels within the annulus (r0 <= dist < r)
    rgn = np.logical_and(dist >= r0, dist < r)
    
    return rgn

#Display an image with color bar
def zdisp_v2(img,zoom=1,cmap='magma',ax=False):

    #if imgX is a string load it as a FITS file name
    if isinstance(img, str) :
        from astropy.io import fits
        fimg = fits.open(img)[0].data
    else :
        fimg=img

    if zoom != 1:
        xc = int(fimg.shape[0]/2.)
        yc = int(fimg.shape[1]/2.)
        xs = int(xc*1.0/zoom)
        ys = int(yc*1.0/zoom)
        fimg = fimg[xc-xs:xc+xs,yc-ys:yc+ys]

    if ax==False:
        fig, pltax = plt.subplots(figsize=(10, 10))
    else:
        pltax = ax
        
    if cmap == 'seismic':
        vlim = np.max([abs(np.min(fimg)),np.max(fimg)])
        vmax = vlim
        vmin = -vlim
        pltax.imshow(fimg,origin="lower",cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax)
    else:
        pltax.imshow(fimg,origin="lower",cmap=plt.get_cmap(cmap))

    return

def zdisp_v3(img, zoom=1, label=None, cmap='magma', mask_radius=None, limfrac=1.0, show_axis_labels=False, show_axis_scale=False, ax=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = plt.gcf()  # Get the current figure if ax is not None
        
    # Load the FITS file if img is a string
    if isinstance(img, str):
        from astropy.io import fits
        fimg = fits.open(img)[0].data
    else:
        fimg = img

    # Apply zoom if zoom parameter is specified
    if zoom != 1:
        xc, yc = np.array(fimg.shape) / 2
        xs, ys = int(xc * (1.0 / zoom)), int(yc * (1.0 / zoom))
        fimg = fimg[int(xc) - xs:int(xc) + xs, int(yc) - ys:int(yc) + ys]

    if mask_radius is not None:
        # Create a circular mask and apply it to the image
        x, y = np.meshgrid(np.arange(fimg.shape[1]), np.arange(fimg.shape[0]))
        mask = (x - fimg.shape[1] // 2)**2 + (y - fimg.shape[0] // 2)**2 >= (mask_radius+0.5)**2
        fimg[~mask] = np.nanmin(fimg)
        
    # Determine vmin and vmax based on limfrac
    if limfrac == 0:
        med = np.median(fimg)
        std = np.std(fimg)                    
        vmax = med+4*std
        vmin = med-std
        vmin0 = med-std
    elif limfrac == 1:
        vmax = np.nanmax(fimg)
        #vmin = np.nanmin(fimg)  # Define vmin0 as vmin for consistency
        vmin = -vmax/2.0
        vmin0 = -vmax  # Define vmin0 as vmin for consistency
    else:
        vlim = np.nanmax(np.abs(fimg)) * limfrac
        vmax = vlim
        vmin = -vlim/2.0
        vmin0 = -vmax  # Define vmin0 as vmin for consistency
        

    # Display the image with the specified colormap and limits
    if cmap == 'seismic':
        im = ax.imshow(fimg, origin="lower", cmap=plt.get_cmap(cmap), vmin=vmin0, vmax=vmax)  # Use vmin instead of vmin0
        lcolor='black'
    else:
        im = ax.imshow(fimg, origin="lower", cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
        lcolor='lightgray'
        
    # Create a filled black circle
    circle = plt.Circle((fimg.shape[1] // 2, fimg.shape[0] // 2), mask_radius, color='purple', fill=True)
    #plt.gca().add_patch(circle)

    # Create a red circle with a smaller radius on top for the thick red circumference
    red_circle = plt.Circle((fimg.shape[1] // 2, fimg.shape[0] // 2), mask_radius-1, color='black', fill=True, lw=3)
    #plt.gca().add_patch(red_circle)
    ax.add_patch(circle)
    ax.add_patch(red_circle)
    
    # make an inside axis for the colorbar
    #cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.01, shrink=0.9)
    ax_pos = ax.get_position()
    # Specify the position and size of the colorbar axis
    cbar_ax_pos = [ax_pos.x0 + ax_pos.width * 0.9, ax_pos.y0, 0.01, ax_pos.height]  # Adjust these values as needed
    cbar_ax = fig.add_axes(cbar_ax_pos)
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelcolor=lcolor, labelsize=24)  # Adjust as needed

    if label != None:
        ax.text(0.05,0.95, label, color=lcolor, fontsize=36, transform=ax.transAxes)  # Adjust the fontsize as needed
    # Draw a 0.2" bar in the lower left
    #bar_length = 0.2 / 0.0125
    #ax.plot([10, 10 + bar_length], [10, 10], 'k-', lw=3, color='lightgray')
    #ax.text(10 + bar_length / 2, 12, '0.2"', color='lightgray', ha='center', fontsize=16)  # Adjust the fontsize as needed

    # Optionally show axis labels and scale
    if not show_axis_labels:
        ax.set_xticks([])
        ax.set_yticks([])
    if not show_axis_scale:
        ax.set_xlabel('')
        ax.set_ylabel('')

    return fig, ax
 

import matplotlib.pyplot as plt
from astropy.io import fits

def plot_fits_matrix(fits_files, nrows, ncols, labels, cmaps, zooms, limfracs, mask_radii, **kwargs):
    """
    Plots a matrix of FITS images.

    Parameters:
    fits_files (list of str): List of FITS file names.
    nrows (int): Number of rows in the plot matrix.
    ncols (int): Number of columns in the plot matrix.
    cmaps (str): String of color map codes for each image.
    zooms (list of float): List of zoom factors for each image.
    limfracs (list of float): List of limfrac values for each image.
    mask_radii (list of float): List of mask_radius values for each image.
    **kwargs: Additional keyword arguments for the zdisp_v3 function.

    Returns:
    fig (Figure): Matplotlib Figure object.
    """

    if nrows * ncols != len(fits_files):
        raise ValueError("Number of rows times number of columns must equal the number of FITS files.")
    if len(cmaps) != len(fits_files) or len(zooms) != len(fits_files) or len(limfracs) != len(fits_files) or len(mask_radii) != len(fits_files):
        raise ValueError("Length of cmaps, zooms, limfracs, and mask_radii must equal the number of FITS files.")

    cmap_dict = {'m': 'magma', 's': 'seismic', 'r': 'RdBu_r'}

    fig, axs = plt.subplots(nrows, ncols, figsize=(10 * ncols, 10 * nrows))
    axs = axs.ravel()
    plt.subplots_adjust(left=0.01, right=0.97, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)

    for i, (fits_file, cmap_code, zoom, limfrac, mask_radius, label) in enumerate(zip(fits_files, cmaps, zooms, limfracs, mask_radii, labels)):
        cmap = cmap_dict.get(cmap_code)
        if cmap is None:
            raise ValueError(f"Invalid color map code: {cmap_code}")

        # Load the FITS file
        img = fits.open(fits_file)[0].data

        # Call the zdisp_v3 function with the specified color map and additional arguments
        zdisp_v3(img, label=label, cmap=cmap, zoom=zoom, limfrac=limfrac, mask_radius=mask_radius, ax=axs[i], **kwargs)

        if i == 0:
            bar_length = 0.2 / 0.0125
            axs[i].plot([10, 10 + bar_length], [10, 10], 'k-', lw=3, color='lightgray')
            axs[i].text(10 + bar_length / 2, 12, '0.2"', color='lightgray', ha='center', fontsize=16)  # Adjust the fontsize as needed

    return fig  # Return the Figure object

def img_range(fimg1, cmap='magma',limfrac=1.0):
    vlim = np.max([abs(np.min(fimg1)),np.max(fimg1)])*limfrac
    vmax = vlim

    if cmap == 'seismic':
        vmin = -vlim
    else:
        vmin = -vlim/2

    return vmin, vmax
    
def zdisp(img,zoom=1,cmap='magma',limfrac=1.0):

    #if imgX is a string load it as a FITS file name
    if isinstance(img, str) :
        from astropy.io import fits
        fimg = fits.open(img)[0].data
    else :
        fimg=img

    if zoom != 1:
        xc = int(fimg.shape[0]/2.)
        yc = int(fimg.shape[1]/2.)
        xs = int(xc*1.0/zoom)
        ys = int(yc*1.0/zoom)
        fimg = fimg[xc-xs:xc+xs,yc-ys:yc+ys]

    vmin, vmax = img_range(fimg, cmap, limfrac) 

    plt.figure(figsize=(10,10))
    
    plt.imshow(fimg,origin="lower",cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax)

    return

 
#Display 2 images with color bar
def zdisp2(img1, img2, zoom=1,cmap='magma'):

    #if imgX is a string load it as a FITS file name
    if isinstance(img1, str) :
        fimg1 = fits.open(img1)[0].data
        fimg2 = fits.open(img2)[0].data
    else :
        fimg1=img1
        fimg2=img2

    if zoom != 1:
        xc = int(fimg1.shape[0]/2.)
        yc = int(fimg1.shape[1]/2.)
        xs = int(xc*1.0/zoom)
        ys = int(yc*1.0/zoom)
        fimg1 = fimg1[xc-xs:xc+xs,yc-ys:yc+ys]
        fimg2 = fimg2[xc-xs:xc+xs,yc-ys:yc+ys]


    f, ax = plt.subplots(1, 2, figsize=(10,10),layout="constrained")
    #plt.rcParams['figure.constrained_layout.use'] = True
    
    vlim = np.max([abs(np.min(fimg1)),np.max(fimg1)])
    vmax = vlim

    if cmap == 'seismic':
        vmin = -vlim
    else:
        vmin = -vlim/2

    ax[0].axis('off')
    ax[0].imshow(fimg1,origin="lower",cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax)
    ax[1].axis('off')
    ax[1].imshow(fimg2,origin="lower",cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax)
    #plt.colorbar(im, ax[1])

    #f.get_layout_engine().set(hspace=0.01, wspace=0.01)

    plt.show()
    
    return

#Display 3 images horizontal layout
#samescale makes sure that all images use the same dynamic range.
#limfrac = sets limits to values shown as fraction of max
def zdisp3(img1, img2, img3, zoom=1,cmap='magma',samescale=False, limfrac=1.0):

    #if imgX is a string load it as a FITS file name
    if isinstance(img1, str) :
        fimg1 = fits.open(img1)[0].data
        fimg2 = fits.open(img2)[0].data
        fimg3 = fits.open(img3)[0].data
    else :
        fimg1=img1
        fimg2=img2
        fimg3=img3

        
    if zoom != 1:
        xc = int(fimg1.shape[0]/2.)
        yc = int(fimg1.shape[1]/2.)
        xs = int(xc*1.0/zoom)
        ys = int(yc*1.0/zoom)
        fimg1 = fimg1[xc-xs:xc+xs,yc-ys:yc+ys]
        fimg2 = fimg2[xc-xs:xc+xs,yc-ys:yc+ys]
        fimg3 = fimg3[xc-xs:xc+xs,yc-ys:yc+ys]

    f, ax = plt.subplots(1, 3, figsize=(10,10),layout="constrained")
    #plt.rcParams['figure.constrained_layout.use'] = True

#    def img_range(fimg1, cmap=cmap,limfrac=limfrac):
#        vlim = np.max([abs(np.min(fimg1)),np.max(fimg1)])*limfrac
#        vmax = vlim
#
#        if cmap == 'seismic':
#            vmin = -vlim
#        else:
#            vmin = -vlim/2
#
#        return vmin, vmax

    vmin, vmax = img_range(fimg1,cmap,limfrac) 

    ax[0].axis('off')
    ax[0].imshow(fimg1,origin="lower",cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax)
    ax[1].axis('off')
    vmin, vmax = img_range(fimg2,cmap,limfrac) if not samescale else (vmin, vmax)
    ax[1].imshow(fimg2,origin="lower",cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax)
    ax[2].axis('off')
    vmin, vmax = img_range(fimg3,cmap,limfrac) if not samescale else (vmin, vmax)
    ax[2].imshow(fimg3,origin="lower",cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax)

    plt.show()
    
    return f, ax


def zdispcb3(img1, img2, img3, zoom=1,cmap='magma',samescale=False, limfrac=1.0):

    #if imgX is a string load it as a FITS file name
    if isinstance(img1, str) :
        fimg1 = fits.open(img1)[0].data
        fimg2 = fits.open(img2)[0].data
        fimg3 = fits.open(img3)[0].data
    else :
        fimg1=img1
        fimg2=img2
        fimg3=img3

        
    if zoom != 1:
        xc = int(fimg1.shape[0]/2.)
        yc = int(fimg1.shape[1]/2.)
        xs = int(xc*1.0/zoom)
        ys = int(yc*1.0/zoom)
        fimg1 = fimg1[xc-xs:xc+xs,yc-ys:yc+ys]
        fimg2 = fimg2[xc-xs:xc+xs,yc-ys:yc+ys]
        fimg3 = fimg3[xc-xs:xc+xs,yc-ys:yc+ys]

    f = plt.figure(figsize=(10, 2.7))

    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], height_ratios=[1])

    ax1 = plt.subplot(gs[0])
    vmin, vmax = img_range(fimg1,cmap) 
    ax1.axis('off')
    ax1.imshow(fimg1,origin="lower",cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax)

    ax2 = plt.subplot(gs[1])
    vmin, vmax = img_range(fimg2,cmap) if not samescale else (vmin, vmax)
    ax2.axis('off')
    ax2.imshow(fimg2,origin="lower",cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax)

    ax3 = plt.subplot(gs[2])
    vmin, vmax = img_range(fimg3,cmap) if not samescale else (vmin, vmax)
    ax3.axis('off')
    ax3.imshow(fimg3,origin="lower",cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax)

    # Add colorbar to the right
    cax = plt.subplot(gs[3])
    plt.colorbar(ax3.imshow(fimg3,origin="lower",cmap=plt.get_cmap(cmap),vmin=vmin,vmax=vmax), cax=cax)

    plt.show()
    
    return


# x_center and y_center will be the center of the gaussian, theta will be the rotation angle
# sigma_x and sigma_y will be the stdevs in the x and y axis before rotation
# x_size and y_size give the size of the frame 

def gauss2d(xc=0, yc=0, fwhm=3, theta=0, xsigma = 0, ysigma = 0, sz=0, xs=101, ys=101):

    sx = xsigma
    sy = ysigma
    x0 = xc
    y0 = yc

    if sz !=0: 
        xs = sz
        ys = sz
        
    if sx == 0:
        sx = fwhm
        sy = fwhm
        
    if x0 == 0:
        x0 = (xs-1)/2.0
    if y0 == 0:
        y0 = (ys-1)/2.0

    theta = 2*np.pi*theta/360
    x = np.arange(0,xs, 1, float)
    y = np.arange(0,ys, 1, float)
    y = y[:,np.newaxis]

    # rotation
    a=np.cos(theta)*x -np.sin(theta)*y
    b=np.sin(theta)*x +np.cos(theta)*y
    a0=np.cos(theta)*x0 -np.sin(theta)*y0
    b0=np.sin(theta)*x0 +np.cos(theta)*y0

    return np.exp(-(((a-a0)**2)/(2*(sx**2)) + ((b-b0)**2) /(2*(sy**2))))

#write an image to a fits file
def writefits(fname, img, header=None):
    """
    Write an image to a FITS file, optionally including a FITS header.

    Parameters:
      fname : str - Filename for the FITS file.
      img : numpy.ndarray - Image data to write.
      header : optional - FITS header to include. If None, a default header is created.

    Returns: None
    """
    from astropy.io import fits
    hdu = fits.PrimaryHDU(img, header=header if header else fits.Header())
    hdu.writeto(fname, overwrite=True, output_verify='silentfix+ignore')

    
def readfits(filename, header=False):
    import io
    from astropy.io import fits

    """
    Read a FITS file that may be compressed with UNIX compress (.Z).
    If the file ends with ".Z" (case-insensitive), it will be decompressed
    using the unlzw3 package and read from a BytesIO stream.
    Otherwise, it is read normally using astropy.io.fits.

    Parameters
    ----------
    filename : str
        Path to the FITS file (can be .fits or .fits.Z).
    header : bool, optional
        If True, returns both the image and the header. Default is False.

    Returns
    -------
    img : ndarray
        The image data from the FITS file.
    hdr : fits.Header (only if header=True)
        The FITS header, returned only if requested.
    """
    if filename.lower().endswith('.z'):
        from unlzw3 import unlzw
        with open(filename, 'rb') as f:
            compressed_data = f.read()
        uncompressed_data = unlzw(compressed_data)
        file_like = io.BytesIO(uncompressed_data)
        img, hdr = fits.getdata(file_like, header=True)
    else:
        img, hdr = fits.getdata(filename, header=True)

    return (img, hdr) if header else img




def gethdrfits(filename):
    """ 
    Efficiently reads only the header from a FITS or FITS.Z file.

    Parameters:
      filename : str - Path to the FITS or compressed FITS.Z file.

    Returns:
      fits.Header - Header of the FITS file.
    """
    import io
    import os
    from astropy.io import fits

    if filename.lower().endswith('.z'):  # Handle compressed .fits.Z files
        from unlzw3 import unlzw
        with open(filename, 'rb') as f:
            compressed_data = f.read()  # Read the full compressed file
        uncompressed_data = unlzw(compressed_data)  # Fully decompress
        file_like = io.BytesIO(uncompressed_data)
        return fits.getheader(file_like)
    
    else:  # Regular .fits file
        return fits.getheader(filename, memmap=True)  # Avoids loading full image



#save a variable to a file.
def save(fname,var):
    with open(fname, 'wb') as f:
        np.save(f, var)
        f.close()

    return

#save a variable to a file.
def savetxt(fname,var):

    with open(fname, 'a') as f:
        f.write(var)
        f.close()

    return

#load 'var' from file 'fname'
def load(fname):
    
    with open(fname, 'rb') as f:
        var = np.load(f,allow_pickle=True)
        f.close()
        
    return var

def nullrm(fname):
    # Read the file contents
    with open(fname, 'r') as file:
        contents = file.read()

    # Replace null characters with an empty string
    contents = re.sub("\x00", "", contents)

    # Split the file into lines
    lines = contents.split("\n")

    # Replace corrupted numbers with "0\n0"
    for i in range(len(lines)):
        line = lines[i]
        if line:
            try:
                float(line)
            except ValueError:
                lines[i] = "0\n0"

    # Write the new contents to the file
    with open(fname, 'w') as file:
        file.write("\n".join(lines))


########## image processing #########################
        

def bpmap(img, size=3, thresh=5):
# just finds outlier in a 2d array
    imgs = median_filter(img, size=size)
    temp = img-imgs
    bmap = (np.abs(temp) > thresh * np.std(temp)).astype(np.uint8)
    return bmap

def bpmapfin(img):
    bp1 = bpmap(img,3)
    bp2 = bpmap(img,13,15)+0.01
    bp2 = uniform_filter(bp2, size=5) > 0.1
    bmap = bp1 * (1-bp2)
    bmap[np.isnan(bmap)] = 1  # Replace NaNs with 1
    return bmap
    
def interp2d(img, x, y):
    """
    Perform cubic interpolation on a 2D image.

    Parameters
    ----------
    img : np.ndarray
        The input image (2D array).
    x : float
        X-coordinate for interpolation.
    y : float
        Y-coordinate for interpolation.

    Returns
    -------
    InterpResult
        A named tuple containing:
        - x (float): Clipped X-coordinate.
        - y (float): Clipped Y-coordinate.
        - interpolated_value (float): Interpolated pixel value.
    """
    import numpy as np
    from scipy.interpolate import RectBivariateSpline
    from collections import namedtuple

    res = namedtuple("res", ["x", "y", "interpolated_value"])

    h, w = img.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    spline = RectBivariateSpline(np.arange(h), np.arange(w), img, kx=3, ky=3)
    val = float(spline(y, x))

    #return res(x, y, interpolated_value)
    return val

#return an rms_map based on rms in annuli around center
def rmsmap_ann(img, xc=None, yc=None, annulus_width=2, max_radius=None):
    # Get image dimensions
    sz1, sz2 = img.shape
    
    # Set xc, yc to the center of the image if not provided
    if xc is None:
        xc = (sz1 - 1) / 2.0
    if yc is None:
        yc = (sz2 - 1) / 2.0
    
    # Calculate distance from each point in the image to the center (xc, yc)
    y, x = np.indices((sz1, sz2))
    dist_from_center = np.sqrt((x - xc)**2 + (y - yc)**2)
    
    # Set maximum radius if not specified
    if max_radius is None:
        max_radius = np.max(dist_from_center)
    
    # Prepare the map to store standard deviations
    std_map = np.zeros_like(img, dtype=float)
    
    # Calculate the standard deviation in each annulus
    current_radius = 0
    while current_radius < max_radius:
        mask = (dist_from_center >= current_radius) & (dist_from_center < current_radius + annulus_width)
        
        # Calculate standard deviation in the current annulus
        if np.any(mask):
            annulus_values = img[mask]
            annulus_std = np.std(annulus_values)
            std_map[mask] = annulus_std
        else:
            std_map[mask] = 0  # Or maintain the previous std value
        
        # Move to the next annulus
        current_radius += annulus_width

    return std_map

def dist_circle(sz, cc=None):
    """ Generate a matrix of distances from the center cc of a square matrix of size sz. """
    if cc is None:
        cc = (sz - 1) / 2.0
    y, x = np.ogrid[:sz, :sz]
    return np.sqrt((x - cc)**2 + (y - cc)**2)

########## astro #########################


def parangle(ha, dec, latitude, degree=False):
    r2d = 180.0 / np.pi
    d2r = np.pi / 180.0

    # Ensure ha is a numpy array for element-wise operations
    ha = np.array(ha)
    had = ha if degree else 15.0 * ha

    # Use the four-parts formula from spherical trigonometry to determine the parallactic angle
    pa = -r2d * np.arctan2(-np.sin(d2r * had),
                           np.cos(d2r * dec) * np.tan(d2r * latitude) -
                           np.sin(d2r * dec) * np.cos(d2r * had))

    # Return the parallactic angle
    return pa

def anglediff(a1, a2, amax=False):
    # Ensure a1 and a2 are numpy arrays
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    # Normalize angles to be within [0, 360) degrees
    a1 = np.mod(a1, 360)
    a2 = np.mod(a2, 360)

    # Calculate possible offsets
    off1 = np.mod((360 - a2) + a1, 360)
    off2 = np.mod((360 - a1) + a2, 360)
    off3 = np.abs(a1 - a2)

    # Find the smallest offset
    val = np.minimum(np.minimum(off1, off2), off3)

    # If amax is True, return the complement to 360
    if amax:
        val = np.mod(360 - val, 360)

    # Handle scalar input by returning a scalar value
    if np.isscalar(a1) and np.isscalar(a2):
        return val.item()  # Convert numpy scalar to a Python scalar
    else:
        return val

    
def radprof(image, dia=5, xc=None, yc=None, orad=None, med=True, rms=False, frms=False, robust=False, average=False, noclip=False):
    sz = image.shape
    if xc is None:
        xc = sz[1] // 2
    if yc is None:
        yc = sz[0] // 2
    if orad is None:
        orad = max(abs(xc - sz[1]), xc, abs(yc - sz[0]), yc) - 2 * dia

    # Creating a distance map from the center
    y, x = np.indices(sz)
    dd = np.sqrt((x - xc)**2 + (y - yc)**2)

    profile = np.zeros(int(orad))

    for j in range(int(orad)):
        rin = j - dia / 2.0
        rout = j + dia / 2.0
        if rin < 0:
            rin = 0
        
        mask = (dd > rin) & (dd <= rout) & (image != 0)
        if np.any(mask):
            data = image[mask]
            if med:
                profile[j] = np.median(data)
            if average:
                profile[j] = np.mean(data)
            if rms:
                if robust:
                    profile[j] = mad_std(data)  # MAD as a robust estimator of sigma
                else:
                    profile[j] = np.std(data)
            if frms:
                median_val = np.median(data)
                if median_val == 0:
                    profile[j] = 0
                else:
                    if robust:
                        profile[j] = mad_std(data) / median_val
                    else:
                        profile[j] = np.std(data) / median_val

    if rms and not noclip:
        rms0 = mad_std(image) if robust else np.std(image)
        profile = np.clip(profile, a_min=rms0 / 3, a_max=None)

    return profile
