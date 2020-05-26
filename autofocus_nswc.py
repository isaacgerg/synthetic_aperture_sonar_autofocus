# Simple autofocus os synthtic apeture sonar imagery
import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import Image
import tqdm
import scipy as sp
import scipy.optimize
import pyfftw
import tensorflow as tf

# Parameters
fiename_dataset = r'sassed.h5'
output_dir = 'results'

# read in the data
h5 = h5py.File(fiename_dataset)
slc_data = h5['data'][...]  # num images x h x w (complex64)
labels = h5['segments'][...]  # same shape as slc_data

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# I got tired of checking if a directory exists so I put this pattern in its own file.
def makeDir(dirname):
    if os.path.exists(dirname):
        return
    else:
        os.makedirs(dirname)
    return

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

makeDir(output_dir)

def brute_force_qpe_autofocus(slc, strength = 15, steps = 500):
    h = slc.shape[0]
    strength_vec = np.linspace(-strength, strength, steps)
    pe_base = np.linspace(-1,1, h)[:,None]**2
    f = np.fft.fftshift(np.fft.fft(slc, axis=0), axes=0)
    pe = np.tile(pe_base, (1, slc.shape[1]))
    def score(raster):
        return raster.var() / (raster.mean()**2)
    best_score = -999999
    for kk in range(steps):
        this_slc = np.fft.ifft(np.fft.ifftshift(np.abs(f)*np.exp(1j*(np.angle(f) + pe*(strength_vec[kk]))), axes=0), axis=0)
        this_score = score(np.abs(this_slc))
        if this_score > best_score:
            best_slc = this_slc
            best_score = this_score
            print(strength_vec[kk])
    #
    return best_slc

def optimizer_qpe_autofocus(slc):
    h = slc.shape[0]
    pe_base = np.linspace(-1,1, h)[:,None]**2
    f = np.fft.fftshift(np.fft.fft(slc, axis=0), axes=0)
    pe = np.tile(pe_base, (1, slc.shape[1]))
    def score(raster):
        return -(raster.var() / (raster.mean()**2)) # minimize this

    def fun(strength, slc):
        h = slc.shape[0]
        pe_base = np.linspace(-1,1, h)[:,None]**2
        pe = np.tile(pe_base, (1, slc.shape[1]))        
        this_slc = np.fft.ifft(np.fft.ifftshift(np.abs(f)*np.exp(1j*(np.angle(f) + pe*(strength))), axes=0), axis=0)
        this_score = score(np.abs(this_slc))
        return this_score
        
    r = sp.optimize.minimize_scalar(fun, args=(f,))

    # Apply focus
    strength = r['x']
    h = slc.shape[0]
    pe_base = np.linspace(-1,1, h)[:,None]
    pe = np.tile(pe_base, (1, slc.shape[1]))        
    af_slc = np.fft.ifft(np.fft.ifftshift(np.abs(f)*np.exp(1j*(np.angle(f) + pe*(strength))), axes=0), axis=0)    

    return af_slc

def poly_qpe_autofocus(slc):
    h = slc.shape[0]
    f = np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft(slc, axis=0), axes=0)
    def score(raster):
        return -(raster.var() / (raster.mean()**2)) # minimize this

    def fun(p, slc):
        h = slc.shape[0]
        #pe_base = np.linspace(-1,1, h)[:,None]**2  # 1001 x 1
        pe_base = np.polyval(p, np.linspace(-1,1, h)).reshape((h, 1))
        pe_base = pe_base - pe_base.mean()
        pe = np.tile(pe_base, (1, slc.shape[1]))
        this_slc = pyfftw.interfaces.numpy_fft.ifft(np.fft.ifftshift(np.abs(f)*np.exp(1j*(np.angle(f) + pe)), axes=0), axis=0, threads=2)
        this_score = score(np.abs(this_slc))
        #print(this_score)
        return this_score
        
    poly_order = 5
    x0 = np.zeros(poly_order).astype('float32')        
        
    # SPSA
    #import noisyopt
    #r = noisyopt.minimizeSPSA(fun, x0, args=(f,), paired=False, disp=True, a = 1e-1, c=1e-5)
    # Compass
    #r = noisyopt.minimizeSPSA(fun, x0, args=(f,), paired=False, disp=True, a = 1e0, c=1e0)
    
    # Scipy out of the box optimizers
    #r = sp.optimize.minimize_scalar(fun, args=(f,))
    r = sp.optimize.minimize(fun, x0, args=(f), tol=1e-3)# options={'gtol': 1e-9, 'norm': np.inf, 'eps': 1e-8, 'maxiter': 10, 'disp': True, 'return_all': False})
    #r = sp.optimize.minimize(fun, x0, args=(f), method='Powell', options={'xtol': 0.0001, 'ftol': 1e-1, 'maxiter': None, 'maxfev': None, 'disp': True, 'direc': None, 'return_all': False})
    #r = sp.optimize.minimize(fun, x0, args=(f), method='Nelder-Mead', options={'xtol': 1e-4, 'ftol': 1e-4, 'maxiter': None, 'maxfev': 200, 'disp': True, 'return_all': False})
    
    # Apply focus
    vec = r['x']
    h = slc.shape[0]
    #pe_base = np.linspace(-1,1, h)[:,None]**2
    pe_base = np.polyval(vec, np.linspace(-1,1, h)).reshape((h, 1))
    pe = np.tile(pe_base, (1, slc.shape[1]))        
    af_slc = np.fft.ifft(np.fft.ifftshift(np.abs(f)*np.exp(1j*(np.angle(f) + pe)), axes=0), axis=0)

    return af_slc, pe_base
    
# TF autofocus method
polynomial_order = 10
input_complex = tf.keras.layers.Input(shape=(1001,1001,1), dtype=tf.complex64)
input_dummy = tf.keras.layers.Input(shape=(1,1))
# Assemble the phase error curve.
pe_curve = tf.linspace(-1.0, 1.0, 1001)
pe_curve = tf.reshape(pe_curve, (1001,1)) # add extra dimension
pe_curve = tf.tile(pe_curve, (1,1001)) # tile
pe_curve = pe_curve[None,:,:] # add extra dimension
pe_curve = tf.tile(pe_curve, (1,1,1)) # tile:  None, 256, 256

phaseError = tf.zeros_like(pe_curve)
d = tf.keras.layers.Dense(polynomial_order+1)(input_dummy)
for k in range(polynomial_order+1):
    if k == 0 or k == 1:  # Skip constant and linear terms
        continue
    phaseError = phaseError + d[0,:,k][:,None, None]*(pe_curve**k)

# Frequency domain version of input
freq_domain  = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(x[:,:,:,0]))(input_complex)
freq_domain_dc_centered  = tf.keras.layers.Lambda(lambda x: tf.signal.fftshift(x, axes=[1,2]))(freq_domain)

# Split into mag and phase    
magFreqDomain = tf.keras.layers.Lambda(lambda x: tf.abs(x))(freq_domain_dc_centered)
phaseFreqDomain = tf.keras.layers.Lambda(lambda x: tf.math.angle(x), name='phase_freq_domain')(freq_domain_dc_centered)

#phaseError = phaseError - tf.reduce_mean(phaseError, axis=[1,2], keepdims=True)
phaseError = tf.keras.layers.Lambda(lambda x:x, name='phase_correction')(phaseError)  # 1, 256, 256, 1
phaseCorrected = phaseFreqDomain - phaseError

phaseCorrected = tf.keras.layers.Lambda(lambda x:x, name='phase_correction_post')(phaseCorrected) 

# Helper function
def mag_phase_combine(mag, phase):      
    cos_phase = tf.math.cos(phase)
    sin_phase = tf.math.sin(phase)
    r = tf.complex(mag*cos_phase, mag*sin_phase)
    return  r
#    

reconstructedFreq_dc_centered = tf.keras.layers.Lambda(lambda x:mag_phase_combine(x[0], x[1]))([magFreqDomain, phaseCorrected])      

# IFFT and ifftshift
reconstructedFreq = tf.keras.layers.Lambda(lambda x: tf.signal.ifftshift(x, axes=[1,2]))(reconstructedFreq_dc_centered)
image_domain_reconstructed = tf.keras.layers.Lambda(lambda x: tf.signal.ifft2d(x), name='slc_af')(reconstructedFreq)
image_domain = tf.keras.layers.Lambda(lambda x: x[:,:,:,None], name='td_drc')(tf.abs(image_domain_reconstructed))

image_domain_power = tf.abs(image_domain) ** 2    # Input contrast
input_image_abs = tf.abs(input_complex)
mean, var = tf.nn.moments(input_image_abs, axes=[1,2,3])  
contrast_input = -(var / (mean**2))    

# Output contrast
mean, var = tf.nn.moments(image_domain, axes=[1,2,3])  
contrast_output = -(var / (mean**2))

# relative contrast improvement
contrast_improvement = (contrast_output - contrast_input) / contrast_input
contrast_improvement = - contrast_improvement

model = tf.keras.models.Model([input_complex, input_dummy], contrast_improvement)
af_output_model = tf.keras.models.Model(model.inputs, [model.get_layer('slc_af').output, model.get_layer('phase_correction').output])
def my_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-1), loss=my_loss)

my_tf_function = tf.reduce_mean(contrast_improvement)
@tf.function
def val_and_grad(x):
    with tf.GradientTape() as tape:
        tape.watch(x)  # records gradient functions on x and adds to the tape
        loss = my_tf_function(x)
    grad = tape.gradient(loss, x)
    return loss, grad

def tf_autofocus(slc):
    prev_loss = 99999
    tmp = []
    f_threshold = 1e-4
    for k in range(100):
        e = model.fit(x=[slc[None,:,:,None], np.zeros((1,1,1))], y=np.zeros((1,1)), epochs=1)
        curr_loss = e.history['loss'][0]
        err_delta = curr_loss - prev_loss 
        prev_loss = curr_loss
        #tmp.append(err_delta)
        #print(err_delta)
        if np.abs(err_delta) < f_threshold:
            break
    
    p = np.squeeze(af_output_model.predict(x=[slc[None,:,:,None], np.zeros((1,1,1))]))
    af_slc = p[0]
    pe_curve = np.real(p[1])[:,0]  # TODO why complex?
    return af_slc, pe_curve

def tf_autofocus_scipy(slc):
    model.fit(x=[slc[None,:,:,None], np.zeros((1,1,1))], y=np.zeros((1,1)), epochs=100)
    
    p = np.squeeze(af_output_model.predict(x=[slc[None,:,:,None], np.zeros((1,1,1))]))
    af_slc = p[0]
    pe_curve = np.real(p[1])[:,0]  # TODO why complex?
    return af_slc, pe_curve

def contrast(x):
    return x.var() / (x.mean()**2)


contrast_before = []
contrast_after = []
for k in tqdm.tqdm(range(slc_data.shape[0])):
    slc = slc_data[k,:,:].copy()
    contrast_before.append(contrast(np.abs(slc)))
    slc_data[k,:,:], pe_curve = tf_autofocus(slc) #poly_qpe_autofocus(slc) #optimizer_qpe_autofocus(slc)
    contrast_after.append(contrast(np.abs(slc_data[k,:,:])))
    plt.figure()
    plt.plot(pe_curve)
    plt.grid(True)
    plt.ylabel('Phase Error [radians]')
    plt.xlabel('Aperture Position')
    plt.savefig(os.path.join(output_dir, 'output_phase_correction_{0}.png'.format(k)))
    plt.close('all')
    imwrite(np.concatenate((schlick(slc), schlick(slc_data[k,:,:])), axis=1), os.path.join(output_dir, 'output_autofocus_before_and_after_{0}.png'.format(k)))
#
improvement = (np.array(contrast_after)-np.array(contrast_before))/np.array(contrast_before)
plt.figure()
plt.boxplot(improvement*100)
plt.grid(True)
plt.ylabel('Contrast Improvement [%]')
plt.savefig(os.path.join(output_dir, 'contrast_improvement.png'))
plt.close('all')

print('done')
    
