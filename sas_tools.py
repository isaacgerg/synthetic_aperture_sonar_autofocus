import numpy as np
import scipy.stats
import ritsar.signal as sig
from PIL import Image

#--------------------------------------------------------------------------------------
#   Adapted from the autoFocus2 function originally written by Douglas Macdonald 
# as part of the RITSAR Python package located at https://github.com/dm6718/RITSAR and
# https://github.com/dm6718/RITSAR/blob/master/ritsar/imgTools.py.
#
#   We modified the phase gradient estimate to use the ML estimate from:
# Jakowatz, Charles V., and Daniel E. Wahl. "Eigenvector method for maximum-likelihood 
#   estimation of phase errors in synthetic-aperture-radar imagery." 
#   JOSA A 10.12 (1993): 2539-2546.
#
#   Assumes SLC azimuth is vertical dimension and range increases left to right 
# along the horizontal dimension.
#
# "np" is the numpy package.
# "sig" is the signal package from RITSAR.
#--------------------------------------------------------------------------------------
def pga(img, win = 'auto', win_params = [100, 0.5]):

    #Derive parameters
    npulses = int(img.shape[0])
    nsamples = int(img.shape[1])

    #Initialize loop variables
    img_af = 1.0*img
    max_iter = 30
    af_ph = 0
    rms = []

    #Compute phase error and apply correction
    for iii in range(max_iter):

        #Find brightest azimuth sample in each range bin
        index = np.argsort(np.abs(img_af), axis=0)[-1]

        #Circularly shift image so max values line up
        f = np.zeros(img.shape)+0j
        for i in range(nsamples):
            f[:,i] = np.roll(img_af[:,i], int(npulses/2-index[i]))

        if win == 'auto':
            #Compute window width
            s = np.sum(f*np.conj(f), axis = -1)
            s = 10*np.log10(s/s.max())
            # For first iteration, use all azimuth data
            if iii == 0:
                width = npulses
            # For second iteration, use half azimuth data
            elif iii == 1:
                width = npulses // 2
            #For all other iterations, use twice the 10 dB threshold
            else:
                width = np.sum(s>-10)
            window = np.arange(npulses/2-width/2,npulses/2+width/2)
        else:
            #Compute window width using win_params if win not set to 'auto'
            width = int(win_params[0]*win_params[1]**iii)
            window = np.arange(npulses/2-width/2,npulses/2+width/2)
            if width<5:
                break

        window = window.astype('int')
        #Window image
        g = np.zeros(img.shape)+0j
        g[window] = f[window]

        #Fourier Transform
        G = sig.ift(g, ax=0)

        # ML method
        phi_dot = np.angle(np.sum(np.conj(G[:-1, :]) * G[1:, :], axis=1)) 
        phi = np.concatenate([[0],  np.cumsum(phi_dot)])
        phi = np.unwrap(phi)

        #Remove linear trend
        t = np.arange(0,nsamples)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t,phi)
        line = slope*t + intercept
        phi = phi - line
        rms.append(np.sqrt(np.mean(phi**2)))

        if win == 'auto':
            if rms[iii]<0.01:
                break

        #Apply correction
        phi2 = np.tile(np.array([phi]).T,(1,nsamples))
        IMG_af = sig.ift(img_af, ax=0)
        IMG_af = IMG_af*np.exp(-1j*phi2)
        img_af = sig.ft(IMG_af, ax=0)

        #Store phase
        af_ph += phi

    print('number of iterations: {}'.format(iii+1))

    return(img_af, np.flip(af_ph), rms)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# C Schlick Rational Tone Mapping Operator
# targetBrightness is 0,1.
def schlick(L, targetBrightness = 0.3, medianFlag = True):
    L = np.squeeze(L)
    if np.iscomplex(L).sum() > 0:   
        L = np.abs(L)
    
    L = normalize(L.astype('float32'))

    # determine b
    if medianFlag:
        m = np.median(L[np.where(L>0)])
    else:        
        m = np.sqrt(np.sum(L**2) / (2*np.prod(L.shape)))
    #

    if np.isnan(m):
        return np.zeros_like(L)

    b = (targetBrightness - targetBrightness * m) / (m - targetBrightness * m)
    b = np.clip(b, 1, 99999999)

    # apply b
    L = (b*L) / ((b-1)*L + 1 + 1e-9)

    return L

#-----------------------------------------------------------------------------------------------------------------------------------------------------
def imwrite(mat, filename, normalize_data=True):
    if mat.ndim == 3:
        mat = mat[:,:,0:3]
        if normalize_data:
            mat = (normalize(mat)*255).astype('uint8')           

    else:  # grayscale
        if normalize_data:
            mat = (normalize(mat)*255).astype('uint8')
        else:
            mat = (mat*255).astype('uint8')
        #
    img = Image.fromarray(mat)
    img.save(filename)
    return

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Normalizes array to [0,1]
def normalize(arr):
    arr -= arr.min()
    arr /= (arr.max() + 1e-9)
    return arr
#-----------------------------------------------------------------------------------------------------------------------------------------------------
def get_fig_as_numpy(fig):
    import io
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # Open the image with PIL and convert to NumPy array
    image = Image.open(buf)
    image_array = np.array(image)
    
    # Close the BytesIO object
    buf.close()
    
    # Resize the image to 256x256 using PIL
    resized_image = Image.fromarray(image_array).resize((512, 512), Image.ANTIALIAS)
    resized_image_array = np.array(resized_image)
    
    return resized_image_array