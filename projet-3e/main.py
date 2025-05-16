import matplotlib.pyplot as plt
import numpy as np
import pywt
from nibabel import load
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.data import camera
from tqdm import trange, tqdm


# load the NIfTI file
T1 ='data/sub-CSI1_ses-16_run-01_T1w.nii'
T2 = 'data/sub-CSI1_ses-16_T2w.nii'
data_t1 = load(T1).get_fdata()
data_t2 = load(T2).get_fdata()

def f(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


# I. Navigating the volumes

# Figure 1: Visualizing slices from T1 and T2-weighted images
def figure1():
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    axes[0, 0].imshow(data_t1[88, ::-1, ::-1].T, cmap='gray')
    axes[0, 0].set_title('Sagittal slice 150, T1-weighted')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(data_t1[:, 70, ::-1].T, cmap='gray')
    axes[0, 1].set_title('Coronal slice 150, T1-weighted')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(data_t1[:, :, 150].T, cmap='gray')
    axes[0, 2].set_title('Axial slice 150, T1-weighted')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(data_t2[88, ::-1, ::-1].T, cmap='gray')
    axes[1, 0].set_title('Sagittal slice 150, T2-weighted')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(data_t2[:, 70, ::-1].T, cmap='gray')
    axes[1, 1].set_title('Coronal slice 150, T2-weighted')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(data_t2[:, :, 150].T, cmap='gray')
    axes[1, 2].set_title('Axial slice 150, T2-weighted')
    axes[1, 2].axis('off')

    fig.tight_layout()
    fig.savefig('figures/figure1.png', dpi=300)


# Create a guiding image for the rest of the assignment
img_t1 = data_t1[88, ::-1, ::-1].T
img_t2 = data_t2[88, ::-1, ::-1].T



# II. Auxiliary functions
def sparsify(img, s=30):
    cutoff = np.percentile(np.abs(img), s)
    sparse_img = np.copy(img)
    sparse_img[np.abs(sparse_img) < cutoff] = 0

    return sparse_img


mse = lambda x, x_hat: np.mean((x - x_hat) ** 2)
resquare = lambda arr: np.reshape(arr, (np.int32(np.sqrt(arr.shape[0])), np.int32(np.sqrt(arr.shape[0]))))
# norm2 = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
norm = lambda arr: arr / np.abs(arr).max()


def wavelet_visualization(img, wavelet, level, fp):
    """Uno approximation y seis details."""
    aprx, d1, d2 = pywt.wavedec2(img, wavelet, level=level)
    limit = np.shape(d2[0])[0]

    h = np.block([[norm(aprx), norm(np.abs(d1[1]))], [norm(np.abs(d1[0])), norm(np.abs(d1[2]))]])
    o = np.block([[h[:limit, :limit], norm(np.abs(d2[1]))], [norm(np.abs(d2[0])), norm(np.abs(d2[2]))]])

    fig, ax = plt.subplots(1, 1)
    ax.imshow(o, cmap='gray')
    ax.axis('off')

    fig.tight_layout()
    fig.savefig(f"figures/{fp}") if fp else fig.show()



# III. Fourier domain compression
img_k_domain = fftshift(fft2(img_t2))

# Figure 2: Display the Fourier transform of the reference image
img_k_domain_scaled = np.abs(img_k_domain)
img_k_domain_log_scaled = np.log(img_k_domain_scaled)

img_k_domain_scaled /= np.max(img_k_domain_scaled)
img_k_domain_log_scaled /= np.max(img_k_domain_log_scaled)

def figure2():
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))

    axes[0].imshow(img_k_domain_scaled, cmap='gray')
    axes[0].set_title("Frequency domain")
    axes[0].axis('off')

    axes[1].imshow(img_k_domain_log_scaled, cmap='gray')
    axes[1].set_title("Frequency domain, log scaled")
    axes[1].axis('off')

    fig.tight_layout()
    fig.savefig('figures/figure2.png', dpi=300)


# Figure 3: Histogram of Fourier domain
def figure3():
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    num_bins, rng = 100, (0, 1)

    ax.hist(img_k_domain_scaled.flatten(), bins=num_bins, range=rng, log=True)
    ax.set_xlabel('Magnitudes')
    ax.set_ylabel('Number of occurrences')
    ax.set_xticks([])

    fig.tight_layout()
    fig.savefig('figures/figure3.png', dpi=300)


# Figure 4: Comparison of fourier reconstructions
def figure4():
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    for row, col, s in [(0, 0, 0), (0, 1, 30), (0, 2, 50), (1, 0, 70), (1, 1, 90), (1, 2, 95)]:
        img_sparse = ifft2(ifftshift(sparsify(img_k_domain, s))).real
        mse_img = mse(norm(img_t2), norm(img_sparse))
        axes[row, col].imshow(img_sparse, cmap='gray')
        axes[row, col].set_title(f"{s}% removed ({mse_img:.04} MSE)")
        axes[row, col].axis('off')

    fig.tight_layout()
    fig.savefig('figures/figure4.png', dpi=300)



# IV. Wavelet domains

# Figures 5 - 7: The Haar, Daubuchies 4, and Coifet 3 wavelet transforms of the reference
def figure5_7():
    wavelet_visualization(img_t2, 'haar',  2, 'figure5.png')
    wavelet_visualization(img_t2, 'db4',   2, 'figure6.png')
    wavelet_visualization(img_t2, 'coif3', 2, 'figure7.png')


# New figure: Wavelet transform and - of a matrix with one element
def figure_basis_element():
    basis_element = np.zeros([256, 256])
    basis_element[128, 128] = 1

    wavelet_visualization(basis_element, 'haar',  2, 'basis_haar.png')
    wavelet_visualization(basis_element, 'db4',   2, 'basis_db4.png')
    wavelet_visualization(basis_element, 'coif3', 2, 'basis_coif3.png')

    fig, ax = plt.subplots(1, 1)
    ax.imshow(fftshift(fft2(basis_element)).real, cmap='gray')
    ax.axis('off')

    fig.savefig('figures/basis_fourier.png')



# Figure 8: Histograms of wavelet transforms
def wavelet_flatten(img, wavelet, level):
    """Flattens and concatenates wavelet transform in a way that it can be reversed."""
    dec = pywt.wavedecn(img, wavelet, level=level)
    coeffs, sizes = dec[0].flatten(), [dec[0].shape[0]]
    for i in range(1, level+1):
        flatten_level = np.hstack([dec[i][detail].flatten() for detail in ['ad', 'da', 'dd']])
        coeffs = np.hstack([coeffs, flatten_level])
        sizes.append(dec[i]['dd'].shape[0])

    return coeffs, sizes


def wavelet_reshape(coeffs, sizes, wavelet):
    """Reshapes flattened coefficients and reverses wavelet transform."""
    curr_index = sizes[0]**2
    dec = [coeffs[:curr_index].reshape((sizes[0], sizes[0]))]
    for size in sizes[1:]:
        dictionary = {}
        for detail in ['ad', 'da', 'dd']:
            dictionary[detail] = coeffs[curr_index:curr_index+size**2].reshape((size, size))
            curr_index += size ** 2
        dec.append(dictionary)

    return pywt.waverecn(dec, wavelet)


def compress_wavelets(img, s, wavelet, level):
    coeffs, sizes = wavelet_flatten(img, wavelet, level)
    coeffs = sparsify(coeffs, s)
    img_comp = wavelet_reshape(coeffs, sizes, wavelet)
    img_comp_mse = mse(norm(img_comp), norm(img))
    
    return img_comp, img_comp_mse


def figure8():
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))

    for i, wavelet in enumerate(['haar', 'db4', 'coif3']):
        for j in range(3):
            axes[i, j].hist(
                norm(wavelet_flatten(img_t2, wavelet, j+1)[0]),
                bins=num_bins,
                range=rng,
                log=True,
            )
            axes[i, j].set_title(f'level={j+1}')
            axes[i, j].set_xlabel('Magnitude') if i == 2 else None
            axes[i, j].set_ylabel(f'wavelet=\'{wavelet}\'\nFrequency') if j==0 else None
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([]) if j!=0 else None
            axes[i, j].set_ylim(0, 1e5)

    fig.tight_layout()
    fig.savefig('figures/figure8.png', dpi=300)


# Figure 9: Comparison of image against image with Wavelet coefficients removed
def figure9():
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))

    img_comp, img_comp_mse = compress_wavelets(img_t2, 10, 'haar', 2)

    axes[0].imshow(img_t2, cmap='gray')
    axes[0].set_title('Reference image')
    axes[0].axis('off')

    axes[1].imshow(img_comp, cmap='gray')
    axes[1].set_title(f'Compressed image (MSE = {img_comp_mse:.04})')
    axes[1].axis('off')

    fig.tight_layout()
    fig.savefig('figures/figure9.png')



# V. How sparse

# Figure 10: Comparisons of the efficacy of different wavelet domains for compression
def test_sparseness(img, wavelet, level, num_points):
    comp_mses = []
    for s in np.linspace(0, 100, num_points):
        _, img_comp_mse = compress_wavelets(img, s, wavelet, level)
        comp_mses.append(img_comp_mse)

    return np.array(comp_mses)

def figure10():
    fig, axes = plt.subplots(3, 3, figsize=(6, 5))

    for i, wavelet in enumerate(['haar', 'db4', 'coif3']):
        for j in range(3):
            axes[i, j].plot(
                np.linspace(0, 100, 50),
                test_sparseness(img_t2, wavelet, j+1, 50),
            )
            axes[i, j].set_title(f'level={j+1}') if i==0 else None
            axes[i, j].set_xlabel(f's (%)') if i==2 else None
            axes[i, j].set_ylabel(f'wavelet=\'{wavelet}\'\nMSE') if j==0 else None
            axes[i, j].set_xticks(range(0, 101, 20) if i==2 else [])
            axes[i, j].set_yticks([]) if j else None
            axes[i, j].set_ylim(0, 4e-2)

    fig.tight_layout()
    fig.savefig('figures/figure10.png')


# Figures 11 - 12: Visualisation of images with different wavelet compressions
def s_visualizer(img, s, fp):
    """There's no good name for the function, it just does these visualizations."""
    fig, axes = plt.subplots(3, 3, figsize=(5, 5))
    
    for i, wavelet in enumerate(['haar', 'db4', 'coif3']):
        for j in range(3):
            img_comp, img_comp_mse = compress_wavelets(img, s, wavelet, j+1)

            axes[i, j].imshow(np.abs(img_comp), cmap='gray')
            axes[i, j].set_title(f'level={j+1}') if i==0 else None
            axes[i, j].set_xlabel(f'mse={img_comp_mse:.4}')
            axes[i, j].set_ylabel(f'wavelet=\'{wavelet}\'') if j==0 else None
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([]) 
    
    fig.tight_layout()
    fig.savefig(fp)


def figure11_12():
    s_visualizer(img_t2, 60, "figures/figure11.png")
    s_visualizer(img_t2, 98, "figures/figure12.png")


def figure13():
    fig, axes = plt.subplots(2, 3, figsize=(6, 4.5))

    sagittal_t1 = data_t1[88,   ::-1,        ::-1].T
    coronal_t1  = data_t1[ :,     70, 241:65:-1].T
    axial_t1 =    data_t1[ :, 35:211,         150].T

    sagittal_t2 = data_t2[88,   ::-1,      ::-1].T
    coronal_t2  = data_t2[ :,     70, 241:65:-1].T
    axial_t2    = data_t2[ :, 35:211,       150].T

    sagittal_t1, sagittal_t1_mse = compress_wavelets(sagittal_t1, 90, 'coif3', 2)
    coronal_t1,  coronal_t1_mse  = compress_wavelets(coronal_t1,  90, 'coif3', 2)
    axial_t1,    axial_t1_mse    = compress_wavelets(axial_t1,    90, 'coif3', 2)

    sagittal_t2, sagittal_t2_mse = compress_wavelets(sagittal_t2, 90, 'coif3', 2)
    coronal_t2,  coronal_t2_mse  = compress_wavelets(coronal_t2,  90, 'coif3', 2)
    axial_t2,    axial_t2_mse    = compress_wavelets(axial_t2,    90, 'coif3', 2)

    axes[0, 0].imshow(sagittal_t1, cmap='gray')
    axes[0, 0].set_xlabel(f'mse={sagittal_t1_mse:.04}', fontsize=11)
    axes[0, 0].set_ylabel('T1', fontsize=12)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    axes[0, 0].set_title('Sagittal slice')
    
    axes[0, 1].imshow(coronal_t1, cmap='gray')
    axes[0, 1].set_xlabel(f'mse={coronal_t1_mse:.04}', fontsize=11)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    axes[0, 1].set_title('Coronal slice')

    axes[0, 2].imshow(axial_t1, cmap='gray')
    axes[0, 2].set_xlabel(f'mse={axial_t1_mse:.04}', fontsize=11)
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    axes[0, 2].set_title('Axial slice')

    axes[1, 0].imshow(sagittal_t2, cmap='gray')
    axes[1, 0].set_xlabel(f'mse={sagittal_t2_mse:.04}', fontsize=11)
    axes[1, 0].set_ylabel('T2', fontsize=12)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    axes[1, 1].imshow(coronal_t2, cmap='gray')
    axes[1, 1].set_xlabel(f'mse={coronal_t2_mse:.04}', fontsize=11)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    axes[1, 2].imshow(axial_t2, cmap='gray')
    axes[1, 2].set_xlabel(f'mse={axial_t2_mse:.04}', fontsize=11)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    fig.tight_layout()
    fig.savefig('figures/figure13.png')



# VI. Compressed sensing

def figure14():
    def f(p):
        y_orig = fftshift(fft2(img_t2)).flatten()
        A = np.random.choice([0, 1], size=(y_orig.shape[0]), p=[p, 1-p])
        y = A * y_orig
        
        return np.abs(ifft2(ifftshift(resquare(y))))

    fig, axes = plt.subplots(1, 4, figsize=(6, 1.8))

    axes[0].imshow(img_t2, cmap='gray')
    axes[0].set_title('p=0')
    axes[0].axis('off')

    axes[1].imshow(f(0.3), cmap='gray')
    axes[1].set_title('p=0.3')
    axes[1].axis('off')

    axes[2].imshow(f(0.5), cmap='gray')
    axes[2].set_title('p=0.5')
    axes[2].axis('off')

    axes[3].imshow(f(0.7), cmap='gray')
    axes[3].set_title('p=0.7')
    axes[3].axis('off')

    fig.tight_layout()
    fig.savefig('figures/figure14.png')


prox_op = lambda v, scale: np.sign(v) * np.maximum(0, np.abs(v) - scale)

def H_matrix(coeffs, sizes, wavelet, A):
    spatial_d = wavelet_reshape(coeffs, sizes, wavelet)
    fourier_d = fftshift(fft2(spatial_d)).flatten()
    out = A * fourier_d

    return out


def H_inv_matrix(img, wavelet, level, A):
    fourier_d = resquare(A * img)
    spatial_d = ifft2(ifftshift(fourier_d))
    wavelet_d = wavelet_flatten(spatial_d, wavelet, level)

    return wavelet_d


def ista_iter(z, y, lambda_, eta, sizes, wavelet, level, A):
    fourier_d = H_matrix(z, sizes, wavelet, A) - y

    wavelet_d, sizes = H_inv_matrix(fourier_d, wavelet, level, A)
    wavelet_d = z - eta * wavelet_d

    return prox_op(wavelet_d, lambda_*eta), sizes


def apply_ista(img, p, lambda_, wavelet, level, return_init=False, iterations=10000, A=None):
    y_orig = fftshift(fft2(img)).flatten()
    if A is None:
        A = np.random.choice([0, 1], size=(y_orig.shape[0]), p=[p, 1-p])
    y = A * y_orig
    img_init = ifft2(ifftshift(resquare(y)))
    z, sizes = wavelet_flatten(img_init, wavelet, level)

    for _ in trange(iterations):
        z, sizes = ista_iter(z, y, lambda_, 1, sizes, wavelet, level, A)

    recreated_img = np.abs(wavelet_reshape(z, sizes, wavelet))
    recreated_mse = mse(norm(img), norm(recreated_img))

    if return_init:
        return recreated_img, recreated_mse, np.abs(img_init)
    else:
        return recreated_img, recreated_mse


def figure15():
    fig, axes = plt.subplots(2, 4, figsize=(4.9,3))

    for i in range(4):
        img, img_mse, img_init = apply_ista(img_t2, 0.3, 0.15, 'coif3', 3, True)

        axes[0, i].imshow(img_init, cmap='gray')
        axes[0, i].set_title(f'Iteration {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_xlabel(f'mse={img_mse:.04}')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    fig.tight_layout()
    fig.savefig('figures/figure15.png')


def figure16():
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))

    y_orig = fftshift(fft2(img_t2)).flatten()
    A = np.random.choice([0, 1], size=(y_orig.shape[0]), p=[0.5, 0.5])

    for i, wavelet in enumerate(['haar', 'db4', 'coif3']):
        for j in range(3):
            img, img_mse = apply_ista(img_t2, 0.5, 0.15, wavelet, j+1, A=A)
            
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].set_ylabel(f'wavelet={wavelet}', fontsize=12) if j==0 else None
            axes[i, j].set_xlabel(f'mse={img_mse:0.4}', fontsize=11)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_title(f'level={j+1}', fontsize=11) if i==0 else None
    
    fig.tight_layout()
    fig.savefig('figures/figure16.png')


def figure17():
    fig, axes = plt.subplots(1, 4, figsize=(6, 2))

    for i, p in enumerate([0.1, 0.3, 0.5, 0.7]):
        img, img_mse = apply_ista(img_t2, p, 0.15, 'haar', 1)

        axes[i].imshow(img, cmap='gray')
        axes[i].set_xlabel(f'mse={img_mse:.04}', fontsize=11)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f'p={p}', fontsize=12)

    fig.tight_layout()
    fig.savefig('figures/figure17.png')


def figure18():
    fig, axes = plt.subplots(1, 4, figsize=(6, 2))

    y_orig = fftshift(fft2(img_t2)).flatten()
    A = np.random.choice([0, 1], size=(y_orig.shape[0]), p=[0.5, 0.5])

    for i, lambda_ in enumerate([0.01, 0.1, 1, 5]):
        img, img_mse = apply_ista(img_t2, 0.5, lambda_, 'haar', 1, iterations=10000, A=A)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_xlabel(f'mse={img_mse:.04}', fontsize=11)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f'lambda={lambda_}', fontsize=12)
    
    fig.tight_layout()
    fig.savefig('figures/figure18.png')


def ista_mse_progression(img, A, lambda_, wavelet, level, its=10000, freq=10):
    y_orig = fftshift(fft2(img)).flatten()
    y = A * y_orig
    img_init = ifft2(ifftshift(resquare(y)))
    z, sizes = wavelet_flatten(img_init, wavelet, level)

    losses = []
    for it in range(its):
        z, sizes = ista_iter(z, y, lambda_, 1, sizes, wavelet, level, A)
        if it % freq == 0:
            recreated_img = np.abs(wavelet_reshape(z, sizes, wavelet))
            recreated_mse = mse(norm(img), norm(recreated_img))
            losses.append(recreated_mse)

    return np.array(losses)

A_matricies = []
for _ in range(10):
    y_orig = fftshift(fft2(img_t2)).flatten()
    A = np.random.choice([0, 1], size=(y_orig.shape[0]), p=[0.5, 0.5])
    A_matricies.append(A)

# progs = []
# for A in A_matricies:
#     prog = ista_mse_progression(img_t2, A, 0.15, 'haar', 1, 10000)
#     progs.append(np.log(prog))

# prog_mean = np.vstack(progs).mean(0)
# plt.plot(np.exp(prog_mean))
# plt.show()


def figure19():
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    its, freq, num_of_As = 10001, 100, 20

    A_matricies = []
    for _ in range(num_of_As):
        y_orig = fftshift(fft2(img_t2)).flatten()
        A = np.random.choice([0, 1], size=(y_orig.shape[0]), p=[0.5, 0.5])
        A_matricies.append(A)

    for i, wavelet in enumerate(['haar', 'db4', 'coif3']):
        for j in range(3):
            progs = []
            for A in tqdm(A_matricies):
                prog = ista_mse_progression(img_t2, A, 0.15, wavelet, j+1, its, freq)
                progs.append(np.log(prog))

            prog_mean = np.exp(np.vstack(progs).mean(0))

            axes[i, j].plot(np.arange(0, its, freq), prog_mean)
            axes[i, j].set_xlabel(f'Iteration', fontsize=12) if i==2 else None
            axes[i, j].set_ylabel(f'wavelet={wavelet}\nMSE', fontsize=12) if j==0 else None
            axes[i, j].set_xticks([]) if i != 2 else None
            axes[i, j].set_yticks([]) if j != 0 else None
            axes[i, j].set_ylim(0, 3.5e-2)
            axes[i, j].set_title(f'level={j+1}', fontsize=11) if i==0 else None
    
    fig.tight_layout()
    fig.savefig('figures/figure19.png')
