import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter
from scipy.signal import medfilt


# I. Slicing

def generate_image(fp="thoraxCT"):
    """Générer une image 3-D à partir de fichiers JPG donnés."""
    images = []
    for i in range(1, 238):  # Il y a 237 fichiers dans le dossier
        image_fp = os.path.join(fp, f"axial{i:05}.jpg")
        image = Image.open(image_fp)
        image = np.array(image)
        images.append(image)

    image_3d = np.stack(images)
    image_3d = image_3d[:, :, :, 0] / 255  # Normaliser les valeurs de pixel entre 0 et 1
    image_3d = np.transpose(image_3d, (1, 2, 0))

    return image_3d


def display_slice(img, x=None, y=None, z=None, title=None):
    """Afficher une tranche d'une image 3-D."""
    if title is not None:
        plt.title(title)
    if x is not None:
        if title is None:
            plt.title(f"Slice x={x}")
        plt.xlabel("z")
        plt.ylabel("y")
        plt.imshow(img[x, :, :], cmap="gray")
    elif y is not None:
        if title is None:
            plt.title(f"Slice y={y}")
        plt.xlabel("z")
        plt.ylabel("x")
        plt.imshow(img[:, y, :], cmap="gray")
    elif z is not None:
        if title is None:
            plt.title(f"Slice z={z}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.imshow(img[:, :, z], cmap="gray")
    plt.show()


# II. Salt and pepper

def add_salt_and_pepper(img, p=0.1):
    """Ajoute du bruit sel et poivre à une image. P est un paramètre entre 0 et 1 qui détermine le niveau de bruit."""
    new_img = img.copy()
    noise = np.random.rand(*new_img.shape)
    new_img[noise < p / 2] = 0
    new_img[noise > 1 - p / 2] = 1

    return new_img


# medfilt(img, kernel_size=N)


def mean_filter(img, N=3):
    """Filtre moyen, N est la taille du filtre."""
    kernel = np.ones((N, N, N)) / (N ** 3)

    return convolve(img, kernel, mode='constant')


# III. Gauss' wrath

def add_gaussian_noise(img, std=0.1):
    """Ajoute du bruit gaussien à une image. std est l'écart-type du bruit."""
    new_img = img.astype(np.float64)
    noise = np.random.normal(0, std, new_img.shape)
    new_img += noise

    return new_img


# gaussian_filter(img, sigma=std)


def FWHM_to_std(FWHM):
    """Convertir la largeur à mi-hauteur en écart-type."""
    return FWHM / (2 * np.sqrt(2 * np.log(2)))


# gaussian_filter(img, sigma=[std_x, std_y, std_z])


# IV. Foramina

def display_canny(img, x=None, y=None, z=None, title=None, low=100, high=200):
    """Appliquer le filtre de Canny à l'image et afficher le résultat."""
    if x is not None:
        slice_img = img[x, :, :]
    elif y is not None:
        slice_img = img[:, y, :]
    elif z is not None:
        slice_img = img[:, :, z]

    slice_img *= 255  # Convertir les valeurs de pixel entre 0 et 255
    edges = cv2.Canny(slice_img.astype(np.uint8), low, high)
    plt.imshow(edges, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()
