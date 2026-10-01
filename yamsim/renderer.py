"""ctypes wrapper around the yamtrace OptiX path tracer (native/yamtrace.cpp)."""
from __future__ import annotations

import ctypes as C
import os
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
LIB_PATH = os.path.join(_ROOT, "build", "libyamtrace.so")
DEVICE_PROGRAM = os.path.join(_ROOT, "build", "device.ptx")

MAT_LAMBERT = 1
MAT_TWO_SIDED = 2
MAT_DIELECTRIC = 4


class YtMaterial(C.Structure):
    _fields_ = [
        ("base_r", C.c_float), ("base_g", C.c_float), ("base_b", C.c_float),
        ("metallic", C.c_float), ("roughness", C.c_float), ("specular", C.c_float), ("spec_tint", C.c_float),
        ("clearcoat", C.c_float), ("clearcoat_gloss", C.c_float), ("sheen", C.c_float), ("subsurface", C.c_float),
        ("emit_r", C.c_float), ("emit_g", C.c_float), ("emit_b", C.c_float),
        ("tex_albedo", C.c_int32), ("tex_rough", C.c_int32), ("tex_normal", C.c_int32),
        ("flags", C.c_int32), ("light_id", C.c_int32), ("uv_scale", C.c_float), ("ior", C.c_float),
    ]


class YtRectLight(C.Structure):
    _fields_ = [(n, C.c_float) for n in
                ["cx", "cy", "cz", "ux", "uy", "uz", "vx", "vy", "vz", "nx", "ny", "nz", "rad_r", "rad_g", "rad_b", "area"]] + [("ltype", C.c_int32), ("radius", C.c_float)]


@dataclass
class Material:
    """Disney principled BRDF parameters (isotropic subset)."""
    name: str
    base_color: tuple = (0.8, 0.8, 0.8)
    metallic: float = 0.0
    roughness: float = 0.5
    specular: float = 0.5
    spec_tint: float = 0.0
    clearcoat: float = 0.0
    clearcoat_gloss: float = 1.0
    sheen: float = 0.0
    subsurface: float = 0.0
    emission: tuple = (0.0, 0.0, 0.0)
    tex_albedo: int = -1
    tex_rough: int = -1
    tex_normal: int = -1
    flags: int = 0
    light_id: int = -1
    uv_scale: float = 1.0
    ior: float = 1.5

    def to_c(self) -> YtMaterial:
        m = YtMaterial()
        m.base_r, m.base_g, m.base_b = [float(x) for x in self.base_color]
        m.metallic, m.roughness, m.specular, m.spec_tint = self.metallic, self.roughness, self.specular, self.spec_tint
        m.clearcoat, m.clearcoat_gloss, m.sheen, m.subsurface = self.clearcoat, self.clearcoat_gloss, self.sheen, self.subsurface
        m.emit_r, m.emit_g, m.emit_b = [float(x) for x in self.emission]
        m.tex_albedo, m.tex_rough, m.tex_normal = self.tex_albedo, self.tex_rough, self.tex_normal
        m.flags, m.light_id, m.uv_scale, m.ior = self.flags, self.light_id, self.uv_scale, self.ior
        return m


@dataclass
class RectLight:
    corner: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    radiance: np.ndarray  # linear RGB radiance
    normal: np.ndarray | None = None  # emitting side; default = cross(u, v)
    disk_radius: float = 0.0  # > 0: disk light centred at `corner`, edge_u/edge_v = unit in-plane axes

    def to_c(self) -> YtRectLight:
        u, v = np.asarray(self.edge_u, float), np.asarray(self.edge_v, float)
        n = np.cross(u, v) if self.normal is None else np.asarray(self.normal, float)
        n = n / np.linalg.norm(n)
        L = YtRectLight()
        L.cx, L.cy, L.cz = self.corner
        L.nx, L.ny, L.nz = n
        L.rad_r, L.rad_g, L.rad_b = self.radiance
        if self.disk_radius > 0:
            L.ltype, L.radius = 1, float(self.disk_radius)
            u = u / np.linalg.norm(u) * self.disk_radius; v = v / np.linalg.norm(v) * self.disk_radius
            L.area = float(np.pi * self.disk_radius ** 2)
        else:
            L.ltype, L.radius = 0, 0.0
            L.area = float(np.linalg.norm(np.cross(u, v)))
        L.ux, L.uy, L.uz = u
        L.vx, L.vy, L.vz = v
        return L


class Renderer:
    def __init__(self, log_level: int = 2):
        self.lib = C.CDLL(LIB_PATH)
        lib = self.lib
        lib.yt_create.restype = C.c_void_p
        lib.yt_create.argtypes = [C.c_char_p, C.c_int]
        lib.yt_destroy.argtypes = [C.c_void_p]
        lib.yt_last_error.restype = C.c_char_p
        lib.yt_last_error.argtypes = [C.c_void_p]
        lib.yt_add_mesh.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint32]
        lib.yt_add_texture.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_uint32, C.c_int]
        lib.yt_add_instance.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_void_p]
        lib.yt_set_instance_transform.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
        lib.yt_set_instance_material.argtypes = [C.c_void_p, C.c_int, C.c_int]
        lib.yt_set_instance_mask.argtypes = [C.c_void_p, C.c_int, C.c_uint]
        lib.yt_set_materials.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32]
        lib.yt_set_lights.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32]
        lib.yt_set_camera.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_float, C.c_float]
        lib.yt_set_distortion.argtypes = [C.c_void_p, C.c_void_p, C.c_float, C.c_float]
        lib.yt_set_background.argtypes = [C.c_void_p, C.c_float, C.c_float, C.c_float]
        lib.yt_commit.argtypes = [C.c_void_p]
        lib.yt_render.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, C.c_int, C.c_float, C.c_int, C.c_void_p, C.c_void_p]
        lib.yt_render_begin.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32]
        lib.yt_render_accumulate.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, C.c_int, C.c_float]
        lib.yt_render_finish.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32, C.c_int, C.c_void_p, C.c_void_p]
        self.ctx = lib.yt_create(DEVICE_PROGRAM.encode(), log_level)
        if not self.ctx:
            raise RuntimeError("yt_create failed (see stderr)")
        self.materials: list[Material] = []
        self.lights: list[RectLight] = []
        self._keep = []  # keep numpy buffers alive until upload done (they are copied, but be safe)

    def __del__(self):
        try:
            if getattr(self, "ctx", None):
                self.lib.yt_destroy(self.ctx)
                self.ctx = None
        except Exception:
            pass

    def _check(self, rc: int, what: str):
        if rc < 0:
            raise RuntimeError(f"{what}: {self.lib.yt_last_error(self.ctx).decode()}")
        return rc

    def add_mesh(self, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray | None = None, uvs: np.ndarray | None = None) -> int:
        v = np.ascontiguousarray(vertices, dtype=np.float32)
        f = np.ascontiguousarray(faces, dtype=np.uint32)
        n = np.ascontiguousarray(normals, dtype=np.float32) if normals is not None else None
        t = np.ascontiguousarray(uvs, dtype=np.float32) if uvs is not None else None
        rc = self.lib.yt_add_mesh(self.ctx, v.ctypes.data, len(v), n.ctypes.data if n is not None else None,
                                  t.ctypes.data if t is not None else None, f.ctypes.data, len(f))
        return self._check(rc, "add_mesh")

    def add_texture(self, rgba: np.ndarray, srgb: bool = True) -> int:
        img = np.ascontiguousarray(rgba, dtype=np.uint8)
        assert img.ndim == 3 and img.shape[2] == 4
        h, w = img.shape[:2]
        return self._check(self.lib.yt_add_texture(self.ctx, img.ctypes.data, w, h, 1 if srgb else 0), "add_texture")

    def add_material(self, m: Material) -> int:
        self.materials.append(m)
        return len(self.materials) - 1

    def add_light(self, L: RectLight) -> int:
        self.lights.append(L)
        return len(self.lights) - 1

    @staticmethod
    def _xf12(T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=np.float32)
        return np.ascontiguousarray(T[:3, :4].reshape(12))

    def add_instance(self, mesh_id: int, material_id: int, T: np.ndarray) -> int:
        xf = self._xf12(T)
        inst = self._check(self.lib.yt_add_instance(self.ctx, mesh_id, material_id, xf.ctypes.data), "add_instance")
        if self.materials[material_id].flags & MAT_DIELECTRIC:
            self._check(self.lib.yt_set_instance_mask(self.ctx, inst, 2), "set_instance_mask")
        return inst

    def set_instance_transform(self, inst: int, T: np.ndarray):
        xf = self._xf12(T)
        self._check(self.lib.yt_set_instance_transform(self.ctx, inst, xf.ctypes.data), "set_instance_transform")

    def set_instance_material(self, inst: int, material_id: int):
        self._check(self.lib.yt_set_instance_material(self.ctx, inst, material_id), "set_instance_material")

    def upload_materials_and_lights(self):
        arr = (YtMaterial * max(1, len(self.materials)))(*[m.to_c() for m in self.materials])
        self._check(self.lib.yt_set_materials(self.ctx, arr, len(self.materials)), "set_materials")
        if self.lights:
            larr = (YtRectLight * len(self.lights))(*[L.to_c() for L in self.lights])
            self._check(self.lib.yt_set_lights(self.ctx, larr, len(self.lights)), "set_lights")
        else:
            self._check(self.lib.yt_set_lights(self.ctx, None, 0), "set_lights")

    def set_camera(self, eye, look_dir, up, tan_half_x: float, tan_half_y: float):
        e = np.ascontiguousarray(eye, dtype=np.float32); d = np.ascontiguousarray(look_dir, dtype=np.float32); u = np.ascontiguousarray(up, dtype=np.float32)
        self._check(self.lib.yt_set_camera(self.ctx, e.ctypes.data, d.ctypes.data, u.ctypes.data, float(tan_half_x), float(tan_half_y)), "set_camera")

    def set_background(self, r: float, g: float, b: float):
        """Uniform environment radiance seen by rays that miss everything (0 = black)."""
        self._check(self.lib.yt_set_background(self.ctx, float(r), float(g), float(b)), "set_background")

    def set_distortion(self, k=(0, 0, 0, 0, 0), principal_offset=(0.0, 0.0)):
        """Lens deviations on top of the pinhole: OpenCV coefficients [k1, k2, p1, p2, k3] (image y down) and the
        principal point offset as a fraction of the half image (+x right, +y down). Rays are undistorted per sample,
        so the image shows what an uncorrected sensor records (barrel for k1 < 0)."""
        k5 = np.zeros(5, dtype=np.float32); k5[:len(k)] = k
        self._check(self.lib.yt_set_distortion(self.ctx, k5.ctypes.data, float(principal_offset[0]), float(principal_offset[1])), "set_distortion")

    def commit(self):
        self._check(self.lib.yt_commit(self.ctx), "commit")

    def render(self, width: int, height: int, spp: int = 64, seed: int = 0, max_bounces: int = 6, clamp: float = 0.0, denoise: bool = False):
        """Returns (rgb linear float32 [h,w,3], aux float32 [h,w,4] = depth, instance_id, prim_id, hit)."""
        rgb = np.empty((height, width, 3), dtype=np.float32)
        aux = np.empty((height, width, 4), dtype=np.float32)
        self._check(self.lib.yt_render(self.ctx, width, height, spp, seed, max_bounces, float(clamp), 1 if denoise else 0, rgb.ctypes.data, aux.ctypes.data), "render")
        return rgb, aux

    # multi-pose accumulation (motion blur): begin -> [change transforms / camera, accumulate]* -> finish
    def render_begin(self, width: int, height: int):
        self._check(self.lib.yt_render_begin(self.ctx, width, height), "render_begin")

    def render_accumulate(self, width: int, height: int, spp: int, seed: int = 0, max_bounces: int = 6, clamp: float = 0.0):
        self._check(self.lib.yt_render_accumulate(self.ctx, width, height, spp, seed, max_bounces, float(clamp)), "render_accumulate")

    def render_finish(self, width: int, height: int, denoise: bool = False):
        rgb = np.empty((height, width, 3), dtype=np.float32)
        aux = np.empty((height, width, 4), dtype=np.float32)
        self._check(self.lib.yt_render_finish(self.ctx, width, height, 1 if denoise else 0, rgb.ctypes.data, aux.ctypes.data), "render_finish")
        return rgb, aux


def tonemap(rgb_linear: np.ndarray, exposure: float = 1.0, mode: str = "srgb") -> np.ndarray:
    """Linear radiance -> 8-bit sRGB. mode 'srgb' = clip + sRGB OETF (camera-like); 'aces' = filmic curve."""
    x = np.clip(rgb_linear * exposure, 0.0, None)
    if mode == "aces":
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        x = (x * (a * x + b)) / (x * (c * x + d) + e)
    x = np.clip(x, 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


def auto_exposure(rgb_linear: np.ndarray, target_mean: float = 0.18, percentile: float | None = None) -> float:
    lum = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]
    if percentile is not None:
        ref = np.percentile(lum, percentile)
        return float(0.9 / max(ref, 1e-8))
    return float(target_mean / max(lum.mean(), 1e-8))
