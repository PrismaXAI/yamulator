"""Material library (Disney principled BRDF parameters) and light colour helpers."""
from __future__ import annotations

import numpy as np

from .renderer import MAT_DIELECTRIC, MAT_LAMBERT, MAT_TWO_SIDED, Material

# ---------------------------------------------------------------- YAM arm parts
# Bead-blasted, black-dyed anodized aluminium (MIL-A-8625 Type II Class 2).
# Measured data (Marshall et al. 2014, arXiv:1407.8265, Texas A&M): total hemispherical reflectance of
# black-dyed anodized 6061 is ~4-5% flat across 400-650 nm for every finish (raw/machined/polished/
# bead-blasted); the mirror-direction fraction of a bead-blasted sample is only ~0.2% at 10-44 deg.
# Physically the surface is a rough (bead-blasted) transparent anodic oxide layer (n ~ 1.6) whose pores
# are saturated with black dye, so almost all visible reflection is the *broad* Fresnel lobe of a rough
# dielectric, with nearly no diffuse body reflection: diffuse at small angles, a satin sheen that
# strengthens toward grazing (Photon Engineering app note on anodized scatter).  In Disney terms:
# metallic 0, F0 ~ 0.045 (specular 0.56 -> 0.08*0.56), roughness ~0.5, and a very dark base colour
# for the residual diffuse part; the total normal-incidence reflectance then lands at ~5%.
BLACK_ANODIZED = Material("black_anodized_al", base_color=(0.018, 0.018, 0.020), metallic=0.0, roughness=0.50, specular=0.56, spec_tint=0.0)
# Matte white plastic covers (injection-moulded ABS/PC, lightly textured): dielectric, moderately rough.
WHITE_PLASTIC = Material("white_matte_plastic", base_color=(0.86, 0.86, 0.84), metallic=0.0, roughness=0.55, specular=0.45)
# Small black plastic parts (wrist camera housings, finger pads).
BLACK_PLASTIC = Material("black_plastic", base_color=(0.03, 0.03, 0.03), metallic=0.0, roughness=0.45, specular=0.5)
# Silicone finger pads / tips.
GRIPPER_TIP = Material("gripper_tip_rubber", base_color=(0.08, 0.08, 0.08), metallic=0.0, roughness=0.75, specular=0.35)

# ---------------------------------------------------------------- test objects
# Glazed white ceramic: smooth dielectric glaze (n~1.5) over a white body -> low roughness,
# a little clearcoat for the extra glossy layer.
CERAMIC_PLATE = Material("ceramic_plate_glazed", base_color=(0.90, 0.90, 0.88), metallic=0.0, roughness=0.14, specular=0.55, clearcoat=0.35, clearcoat_gloss=0.9)
# Apple skin: waxy cuticle -> textured albedo, roughness map from the scan, mild clearcoat sheen.
APPLE = Material("apple", base_color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.8, specular=0.5, clearcoat=0.25, clearcoat_gloss=0.75)
APPLE_FALLBACK = Material("apple_red_ball", base_color=(0.55, 0.06, 0.04), metallic=0.0, roughness=0.3, specular=0.5, clearcoat=0.3, clearcoat_gloss=0.8)

# Clear thin plastic (PET/PP clamshell): smooth dielectric, n = 1.5, no absorption.
CLEAR_PLASTIC = Material("clear_plastic", base_color=(1.0, 1.0, 1.0), flags=MAT_DIELECTRIC, ior=1.5)

# ---------------------------------------------------------------- environment (Lambertian)
CELL_WHITE = Material("work_cell_white", base_color=(0.85, 0.85, 0.85), flags=MAT_LAMBERT | MAT_TWO_SIDED)
TABLE_WHITE = Material("table_top", base_color=(0.80, 0.80, 0.79), flags=MAT_LAMBERT | MAT_TWO_SIDED)
ROOM_WALL = Material("room_wall_gray", base_color=(0.6, 0.6, 0.6), flags=MAT_LAMBERT | MAT_TWO_SIDED)
ROOM_FLOOR = Material("room_floor_gray", base_color=(0.3, 0.3, 0.3), flags=MAT_LAMBERT | MAT_TWO_SIDED)
LIGHT_FIXTURE = Material("light_fixture_frame", base_color=(0.7, 0.7, 0.7), flags=MAT_LAMBERT | MAT_TWO_SIDED)


# ---------------------------------------------------------------- colour temperature -> linear sRGB
def _cie_xyz_bar(lam_nm: np.ndarray):
    """Wyman, Sloan & Shirley 2013 multi-lobe Gaussian fit of the CIE 1931 2deg observer."""
    def g(x, mu, s1, s2):
        t = (x - mu) / np.where(x < mu, s1, s2)
        return np.exp(-0.5 * t * t)
    x = 1.056 * g(lam_nm, 599.8, 37.9, 31.0) + 0.362 * g(lam_nm, 442.0, 16.0, 26.7) - 0.065 * g(lam_nm, 501.1, 20.4, 26.2)
    y = 0.821 * g(lam_nm, 568.8, 46.9, 40.5) + 0.286 * g(lam_nm, 530.9, 16.3, 31.1)
    z = 1.217 * g(lam_nm, 437.0, 11.8, 36.0) + 0.681 * g(lam_nm, 459.0, 26.0, 13.8)
    return x, y, z


def blackbody_rgb(temp_k: float, normalize: str = "luminance") -> np.ndarray:
    """Linear-sRGB colour of a Planckian radiator at temp_k (RGB approximation of the spectrum).

    normalize='luminance' scales so that Y (luminance) == 1, i.e. the returned triple times a
    photometric-ish radiance gives that luminance. Fluorescent 5500K tubes are not true
    blackbodies (line spectra) but their chromaticity sits near the Planckian locus, which is
    all an RGB renderer can represent.
    """
    lam = np.arange(380.0, 781.0, 1.0)
    l_m = lam * 1e-9
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    B = (2 * h * c ** 2) / (l_m ** 5) / (np.exp(h * c / (l_m * k * temp_k)) - 1.0)
    xb, yb, zb = _cie_xyz_bar(lam)
    X, Y, Z = (B * xb).sum(), (B * yb).sum(), (B * zb).sum()
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])  # XYZ (D65) -> linear sRGB
    rgb = M @ np.array([X, Y, Z])
    rgb = np.clip(rgb, 0.0, None)
    if normalize == "luminance":
        rgb = rgb / Y
    elif normalize == "max":
        rgb = rgb / rgb.max()
    return rgb


def lambertian_panel_radiance(lumens: float, area_m2: float) -> float:
    """Luminance-equivalent radiance of a Lambertian panel: L = Phi / (pi * A) (in cd/m^2 for lm input)."""
    return lumens / (np.pi * area_m2)
