"""
Synthetic Aperture Sonar (SAS) Processing Tools

This module provides tools for processing and analyzing Synthetic Aperture Sonar images,
including Phase Gradient Autofocus (PGA), tone mapping, and image handling utilities.

Key Functions:
    - pga: Phase Gradient Autofocus implementation
    - schlick: Schlick's rational tone mapping operator
    - imwrite: Image writing utility
    - normalize: Array normalization to [0,1] range
    - get_fig_as_numpy: Convert matplotlib figure to numpy array

References:
    - Original autoFocus2 function: https://github.com/dm6718/RITSAR
    - PGA ML estimation method: Jakowatz, Charles V., and Daniel E. Wahl.
      "Eigenvector method for maximum-likelihood estimation of phase errors in
      synthetic-aperture-radar imagery." JOSA A 10.12 (1993): 2539-2546.
    - Shadow PGA: Prater, et al. "SHADOW BASED PHASE GRADIENT AUTOFOCUS FOR
      SYNTHETIC APERTURE SONAR." 5th annual Institute of Acoustics SAS/SAR
      Conference. Lerici, Italy. 2023.
"""

import numpy as np
import scipy.stats
import ritsar.signal as sig
from PIL import Image, ImageFilter
from PIL.Image import Resampling
import io


def pga(img, win='auto', win_params=[100, 0.5], shadow_pga=False):
    """
    Phase Gradient Autofocus implementation for SAS/SAR imagery.

    Adapted from the autoFocus2 function by Douglas Macdonald (RITSAR package).
    Uses ML estimation method from Jakowatz & Wahl (1993) and shadow PGA
    techniques from Prater et al. (2023).

    Assumes SLC azimuth is vertical dimension and range increases left to right
    along the horizontal dimension.

    Args:
        img (numpy.ndarray): Input complex image
        win (str): Window type ('auto' or 'custom')
        win_params (list): Window parameters [width, decay_factor]
        shadow_pga (bool): Whether to use shadow-based PGA

    Returns:
        tuple: (focused_image, phase_error, rms_values)
    """
    # Derive parameters
    npulses = int(img.shape[0])
    nsamples = int(img.shape[1])

    # Initialize loop variables
    img_af = 1.0 * img
    max_iter = 30
    af_ph = 0
    rms = []

    # Compute phase error and apply correction
    for iii in range(max_iter):
        # Find brightest/darkest azimuth sample in each range bin
        if shadow_pga:
            index = np.argsort(np.abs(img_af), axis=0)[0]
        else:
            index = np.argsort(np.abs(img_af), axis=0)[-1]

        # Circularly shift image so max values line up
        f = np.zeros(img.shape) + 0j
        for i in range(nsamples):
            f[:, i] = np.roll(img_af[:, i], int(npulses/2-index[i]))

        # Window computation
        if win == 'auto':
            s = np.sum(f * np.conj(f), axis=-1)
            s = 10 * np.log10(s/s.max())

            # Window width selection based on iteration
            if iii == 0:
                width = npulses
            elif iii == 1:
                width = npulses // 2
            else:
                width = np.sum(s > -10)

            window = np.arange(npulses/2-width/2, npulses/2+width/2)
        else:
            width = int(win_params[0] * win_params[1]**iii)
            window = np.arange(npulses/2-width/2, npulses/2+width/2)
            if width < 5:
                break

        window = window.astype('int')

        # Window image
        g = np.zeros(img.shape) + 0j
        g[window] = f[window]

        # Fourier Transform
        G = sig.ift(g, ax=0)

        # ML method for phase gradient estimation
        phi_dot = np.angle(np.sum(np.conj(G[:-1, :]) * G[1:, :], axis=1))
        phi = np.concatenate([[0], np.cumsum(phi_dot)])
        phi = np.unwrap(phi)

        # Remove linear trend
        t = np.arange(0, nsamples)
        slope, intercept, _, _, _ = scipy.stats.linregress(t, phi)
        line = slope * t + intercept
        phi = phi - line

        if shadow_pga:
            phi = -phi

        rms.append(np.sqrt(np.mean(phi**2)))

        if win == 'auto' and rms[iii] < 0.01:
            break

        # Apply correction
        phi2 = np.tile(np.array([phi]).T, (1, nsamples))
        IMG_af = sig.ift(img_af, ax=0)
        IMG_af = IMG_af * np.exp(-1j * phi2)
        img_af = sig.ft(IMG_af, ax=0)

        # Store phase
        af_ph += phi

    print('number of iterations: {}'.format(iii + 1))
    return img_af, np.flip(af_ph), rms


def schlick(L, target_brightness=0.3, median_flag=True):
    """
    C. Schlick Rational Tone Mapping Operator.

    Args:
        L (numpy.ndarray): Input luminance values
        target_brightness (float): Target brightness in [0,1]
        median_flag (bool): Use median instead of RMS for brightness estimation

    Returns:
        numpy.ndarray: Tone mapped image
    """
    L = np.squeeze(L)
    if np.iscomplex(L).sum() > 0:
        L = np.abs(L)

    L = normalize(L.astype('float32'))

    # Determine brightness parameter
    if median_flag:
        m = np.median(L[np.where(L > 0)])
    else:
        m = np.sqrt(np.sum(L**2) / (2 * np.prod(L.shape)))

    if np.isnan(m):
        return np.zeros_like(L)

    # Calculate and clip brightness adjustment parameter
    b = (target_brightness - target_brightness * m) / (m - target_brightness * m)
    b = np.clip(b, 1, 99999999)

    # Apply tone mapping
    L = (b * L) / ((b - 1) * L + 1 + 1e-9)
    return L


def imwrite(mat, filename, normalize_data=True):
    """
    Write numpy array as image file.

    Args:
        mat (numpy.ndarray): Input image array
        filename (str): Output filename
        normalize_data (bool): Whether to normalize data to [0,255]
    """
    if mat.ndim == 3:
        mat = mat[:, :, 0:3]
        if normalize_data:
            mat = (normalize(mat) * 255).astype('uint8')
    else:  # grayscale
        if normalize_data:
            mat = (normalize(mat) * 255).astype('uint8')
        else:
            mat = (mat * 255).astype('uint8')

    img = Image.fromarray(mat)
    img.save(filename)


def normalize(arr):
    """
    Normalize array to range [0,1].

    Args:
        arr (numpy.ndarray): Input array

    Returns:
        numpy.ndarray: Normalized array
    """
    arr -= arr.min()
    arr /= (arr.max() + 1e-9)
    return arr


def get_fig_as_numpy(fig):
    """
    Convert matplotlib figure to numpy array with resizing.

    Args:
        fig (matplotlib.figure.Figure): Input figure

    Returns:
        numpy.ndarray: Resized image array (512x512)
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # Open the image with PIL and convert to NumPy array
    image = Image.open(buf)
    image_array = np.array(image)

    # Close the BytesIO object
    buf.close()

    # Resize the image to 512x512 using PIL
    resized_image = Image.fromarray(image_array).resize((512, 512), Resampling.LANCZOS)
    resized_image_array = np.array(resized_image)

    return resized_image_array