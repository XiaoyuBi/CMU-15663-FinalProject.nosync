import cv2
import numpy as np
from scipy.signal import fftconvolve
from utils.wrap_boundary import wrap_boundary_liu, opt_fft_size
from utils.fourier import psf2otf
from utils.util import pad_to_shape


def richardson_lucy_deblur(B, K, num_iter = 10, eps = 1e-5):
    """
    Image deblurring with known PSF

    Input:
        B: blurred image in gray scale (!!!)
        K: blurred kernel (should be known)
        num_iter:
    Output:
        I: latent image
    """

    # to utilize the conv2 function we must make sure the inputs are float
    
    # I = 0.5 * np.ones_like(B); # initial estimate
    I = B; # initial estimate
    K_flip = np.flip(K) # spatially reversed psf
    
    # iterate towards Max Likelihood estimate for the latent image
    for _ in range(num_iter):

        est_conv      = fftconvolve(I, K, 'same') + eps
        relative_blur = B / est_conv
        I             = I * fftconvolve(relative_blur, K_flip, 'same')
    
    # image clip
    I[I > 1.0] = 1.0
    I[I < 0.0] = 0.0

    return I


def richardson_lucy_deblur_blind(B, K_init, num_iter = 10, eps = 1e-5):
    """
    Image deblurring with unknown PSF

    Input:
        B: blurred image in gray scale (!!!)
        K: initial blurred kernel
        num_iter:
    Output:
        I: latent image
    """

    # to utilize the conv2 function we must make sure the inputs are float
    
    # I = 0.5 * np.ones_like(B); # initial estimate
    I = B; # initial estimate

    # pad kernel to make it same size as the image
    K = pad_to_shape(K_init, B.shape)
    
    # iterate towards Max Likelihood estimate for the latent image
    for _ in range(num_iter):

        K_flip        = np.flip(K) # spatially reversed psf
        est_conv      = fftconvolve(I, K, 'same') + eps
        relative_blur = B / est_conv
        I             = I * fftconvolve(relative_blur, K_flip, 'same')
        I_flip        = np.flip(I)
        K             = K * fftconvolve(relative_blur, I_flip, 'same')

    
    # image clip
    I[I > 1.0] = 1.0
    I[I < 0.0] = 0.0

    # psf normalization
    K = K / np.sum(K)

    return I, K


def L0_deblur(B, K, lamda, kappa = 2):
    """
    Blind image restoration with L0 prior
    Objective Function:
        I^* = argmin ||I * K - B||^2 + lambda * ||\\nabla I||_0

    The Code is created based on the method described in the following paper 
    [1] Jinshan Pan, et al. Deblurring Text Images via L0-Regularized Intensity 
    and Gradient Prior, CVPR, 2014. 
    [2] Li Xu, et al. Image smoothing via l0 gradient minimization.
    ACM Trans. Graph., 30(6):174, 2011.
    
    Input:
        B: blurred image
        K: blurred kernel
        lamda (lambda): regularization weight for L0 prior
        kappa: update ratio in the ADMM
    Output:
        I: latent image
    """

    betaMax = 1e5

    # boundary padding
    old_h, old_w, old_c = B.shape
    K_h, K_w = K.shape
    B = wrap_boundary_liu(B, (old_h+K_h-1, old_w+K_w-1))
    h, w, c = B.shape # new shape
    I = B

    # FFT
    nabla_x = np.array([[1, -1]])
    nabla_y = np.array([[1], [-1]])
    nabla_x_fft = psf2otf(nabla_x, (h, w))
    nabla_y_fft = psf2otf(nabla_y, (h, w))
    K_fft = psf2otf(K, (h, w))

    ##
    denominator1 = np.abs(K_fft) * np.abs(K_fft)
    denominator2 = np.abs(nabla_x_fft) * np.abs(nabla_x_fft) +\
                   np.abs(nabla_y_fft) * np.abs(nabla_y_fft)

    if c > 1:
        K_fft = np.stack([K_fft for _ in range(c)], axis = 2)
        denominator1 = np.stack([denominator1 for _ in range(c)], axis = 2)
        denominator2 = np.stack([denominator2 for _ in range(c)], axis = 2)
    
    numerator1 = np.conj(K_fft) * np.fft.fft2(I)

    # Iteration
    beta = kappa * lamda
    while beta < betaMax:
        denominator = denominator1 + beta * denominator2
        I_x, I_y = np.gradient(I, axis = (0, 1))

        if c == 1:
            mask = I_x * I_x + I_y * I_y < lamda / beta
        else:
            mask = np.sum(I_x * I_x + I_y * I_y, axis = 2) < lamda / beta
            mask = np.stack([mask for _ in range(c)], axis = 2)
        
        I_x[mask] = 0
        I_y[mask] = 0

        I_xx, _ = np.gradient(I_x, axis = (0, 1))
        _, I_yy = np.gradient(I_y, axis = (0, 1))
        numerator2 = np.fft.fft2(- I_xx - I_yy)

        I_fft = (numerator1 + beta * numerator2) / denominator
        I = np.real(np.fft.ifft2(I_fft))

        beta = beta * kappa

    return I

