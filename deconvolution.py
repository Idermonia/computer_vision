import numpy as np


def count_gauss(r, sigm):
    return np.exp((-(r**2))/(2*sigm**2))/(2*np.pi*sigm**2)


def gaussian_kernel(size, sigma):
    gauss = np.zeros((size, size), np.float64)
    x0 = size // 2
    y0 = size // 2
    for i in range (size):
        for j in range (size):
            r = np.sqrt((i - x0)**2 + (j - y0) ** 2)
            gauss[i, j] = count_gauss(r, sigma)
    gauss = gauss / (np.sum(gauss))
    return gauss

def fourier_transform(h, shape):
    hi, ln = np.shape(h)
    ans = np.zeros((shape[0], shape[1]), np.float64)
    ans[:hi, :ln] = h
    return np.fft.fft2(ans)


def inverse_kernel(H, threshold=1e-10):
    H = np.array(H, dtype=np.complex64)
    h_inv = np.zeros_like(H)
    hi, ln = np.shape(H)
    print(hi, ln)
    for i in range (hi):
        for j in range (ln):
            if np.abs(H[i, j]) <= threshold:
                h_inv[i, j] = 0
            else:
                h_inv[i, j] = 1/H[i, j]
    return h_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    blurred_img = fourier_transform(blurred_img, np.shape(blurred_img))
    h = fourier_transform(h, np.shape(blurred_img))
    h = inverse_kernel(h, threshold)
    f = blurred_img * h
    f = np.fft.ifft2(f)
    return np.abs(f)


def wiener_filtering(blurred_img, h, K=0.0002):
    blurred_img = fourier_transform(blurred_img, np.shape(blurred_img))
    h = fourier_transform(h, np.shape(blurred_img))
    f = (np.conjugate(h)/(np.abs(h * np.conjugate(h)) + K)) * blurred_img
    f = np.fft.ifft2(f)
    return np.abs(f)


def compute_psnr(img1, img2):
    maxi = 255
    mse = ((img1 - img2) ** 2).mean()
    return 20 * np.log10(maxi/np.sqrt(mse))
