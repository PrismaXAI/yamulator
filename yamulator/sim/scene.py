"""Assembly of the bimanual YAM work cell: geometry, materials, lights, camera, and arm rigging.

World frame: z up, origin at the midpoint between the two arm mounting flanges on the work surface
(z = 0 is the arm base plane). The arms face +y toward the rear wall of the work cell; +x is to the
right when standing behind the arms looking at the rear wall (i.e. from the overhead camera's view).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

import numpy as np
import trimesh
from PIL import Image

from . import materials as M
from .kinematics import GripperCalibration, TRAIN_GRIPPER_CAL, YamArm, rpy_to_matrix
from .renderer import MAT_LAMBERT, MAT_TWO_SIDED, Material, RectLight, Renderer, tonemap

from yamulator.paths import ROOT

_ROOT = str(ROOT)
YAM_URDF = os.path.join(_ROOT, "assets", "yam", "yam.urdf")
WRIST_CAM_STL = os.path.join(_ROOT, "assets", "yam", "assets", "d405_wrist_linear_4310.stl")
APPLE_GLTF = os.path.join(_ROOT, "assets", "apple", "food_apple_01_1k.gltf")

# White plastic cover extents along each long link's mesh z axis (metres, mesh coordinates),
# found by slicing the fused STL: the cover is the waisted, fluted section between motor housings.
COVER_Z_RANGE = {"link2": (0.057, 0.247), "link3": (0.079, 0.246)}


def tf(xyz=(0, 0, 0), rpy=(0, 0, 0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(*rpy)
    T[:3, 3] = xyz
    return T


@dataclass
class OverheadCamera:
    """Pinhole camera on the arm centreline. Defaults per the station spec."""
    height: float = 0.6          # above the work surface
    tilt_deg: float = 45.0       # looking down from horizontal
    diag_fov_deg: float = 105.0
    aspect: float = 16 / 10
    x_offset: float = 0.0        # along the arm baseline
    y_offset: float = 0.0        # forward of the baseline
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    distortion: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)   # OpenCV [k1, k2, p1, p2, k3] of the raw (uncorrected) lens; zeros = ideal pinhole
    principal_offset: tuple = (0.0, 0.0)             # principal point offset, fraction of the half image (+x right, +y down)

    def pose(self):
        eye = np.array([self.x_offset, self.y_offset, self.height])
        t = np.radians(self.tilt_deg)
        look = np.array([0.0, np.cos(t), -np.sin(t)])
        Rz = rpy_to_matrix(0, 0, np.radians(self.yaw_deg))
        look = Rz @ look
        up = np.array([0.0, 0.0, 1.0])
        if self.roll_deg:
            K = look / np.linalg.norm(look)
            r = np.radians(self.roll_deg)
            up = up * np.cos(r) + np.cross(K, up) * np.sin(r) + K * np.dot(K, up) * (1 - np.cos(r))
        return eye, look, up

    def tan_half(self):
        d = np.tan(np.radians(self.diag_fov_deg) / 2)
        ty = d / np.sqrt(1 + self.aspect ** 2)
        return ty * self.aspect, ty

    def resolution(self, width: int):
        return width, int(round(width / self.aspect))


@dataclass
class LookAtCamera:
    """Free camera: eye position, target point, diagonal FOV, aspect."""
    eye: tuple = (0.3, -0.3, 0.4)
    target: tuple = (0.0, 0.3, 0.0)
    diag_fov_deg: float = 60.0
    aspect: float = 16 / 10
    distortion: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)   # OpenCV [k1, k2, p1, p2, k3] of the raw (uncorrected) lens; zeros = ideal pinhole
    principal_offset: tuple = (0.0, 0.0)             # principal point offset, fraction of the half image (+x right, +y down)

    def pose(self):
        eye = np.asarray(self.eye, float); look = np.asarray(self.target, float) - eye
        return eye, look / np.linalg.norm(look), np.array([0.0, 0.0, 1.0])

    def tan_half(self):
        d = np.tan(np.radians(self.diag_fov_deg) / 2)
        ty = d / np.sqrt(1 + self.aspect ** 2)
        return ty * self.aspect, ty

    def resolution(self, width: int):
        return width, int(round(width / self.aspect))


# Wrist camera mount pose from the i2rt station model (its D405 camera_joint; our module sits in the same mount)
# (camera_joint of yam_station_linear_4310_d405.urdf): optical frame, OpenCV convention (+z optical
# axis, +x image right, +y image down).
WRIST_CAM_T = tf(xyz=(-0.07043497874827391, 0.0, -0.07700592901799347), rpy=(2.70525, 0.0, 1.5707963267814136))


@dataclass
class WristCamera:
    """Wrist camera on one gripper: same optics as the overhead camera (105 deg diagonal FOV, 16:10).
    The unit is the left eye of a stereo module, so its optical centre sits ~2 cm to the right (+x of
    the optical frame, image right) of the mount centre given by the station model."""
    side: str = "left"
    diag_fov_deg: float = 105.0
    aspect: float = 16 / 10
    lateral_offset: float = 0.02   # metres along optical +x (image right)
    pitch_deg: float = 0.0         # mounting error: rotation about the optical x axis (+ = looks further down toward the fingers)
    yaw_deg: float = 0.0           # mounting error: rotation about the optical y axis
    offset: tuple = (0.0, 0.0, 0.0)  # mounting error: translation in the optical frame (x right, y down, z forward)
    needs_station: bool = True
    distortion: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)   # OpenCV [k1, k2, p1, p2, k3] of the raw (uncorrected) lens; zeros = ideal pinhole
    principal_offset: tuple = (0.0, 0.0)             # principal point offset, fraction of the half image (+x right, +y down)

    def pose(self, station: "YamStation"):
        T = station.link_pose(self.side, "gripper") @ WRIST_CAM_T
        R = T[:3, :3]
        Rerr = rpy_to_matrix(np.radians(self.pitch_deg), np.radians(self.yaw_deg), 0.0)  # in the optical frame
        R = R @ Rerr
        eye = T[:3, 3] + R @ np.array([self.lateral_offset + self.offset[0], self.offset[1], self.offset[2]])
        look = R[:, 2]; up = -R[:, 1]
        return eye, look, up

    def tan_half(self):
        d = np.tan(np.radians(self.diag_fov_deg) / 2)
        ty = d / np.sqrt(1 + self.aspect ** 2)
        return ty * self.aspect, ty

    def resolution(self, width: int):
        return width, int(round(width / self.aspect))


@dataclass
class StationConfig:
    arm_spacing: float = 0.61
    arm_base_height: float = 0.0      # arm mounting flanges above the work surface (data rig 'mode B' had elevated bases)
    arm_base_pitch_deg: float = 0.0   # whole-arm mount tilt about the lateral axis, + = reach direction tips UP (bases not flat)
    rear_wall_dist: float = 0.8       # from the arm baseline
    cell_wall_w: float = 1.0
    cell_wall_h: float = 0.5
    cell_x_center: float = 0.0        # lateral offset of the work cell (rear panel centre, side panels) from the arm midline
    cell_wall_len: float | None = None  # side panels run from the rear wall toward the arms this far (None = cell_wall_w)
    table_size: tuple = (1.2, 1.3)    # x, y extent of the work surface slab
    table_y_center: float = 0.22
    table_thickness: float = 0.03
    room_size: float = 3.0            # x and y
    room_height: float = 3.0          # floor to ceiling (assumed; not in the spec)
    table_height: float = 0.7         # work surface above the floor
    light_size: tuple = (0.5, 1.0)    # each panel (x, y)
    light_positions: tuple = ((-0.75, -0.75), (0.75, -0.75), (-0.75, 0.75), (0.75, 0.75))
    light_drop: float = 0.02          # panel face below the ceiling
    light_lumens: float = 5000.0      # per panel (flux is kept when the shape changes)
    light_shape: str = "rect"         # 'rect' (0.5 x 1 m panels) or 'disk'
    light_disk_radius: float = 0.15   # for 'disk'
    light_cct: float = 5500.0
    camera: OverheadCamera = field(default_factory=OverheadCamera)
    exposure: float = 1.0 / 300.0     # linear radiance (cd/m^2) -> display; fixed so poses are comparable
    plate_diameter: float = 0.20
    n_apples: int = 1               # render instances apple, apple1, ... (extra ones start parked below the table)
    apple_diameter: float = 0.075   # a real apple; the 3 cm of the original spec looked like a cherry from the overhead camera
    white_balance: bool = False     # camera-style AWB: divide by the light colour so grey surfaces render grey
    plate_xy: tuple = (-0.22, 0.33)
    containers: bool = True
    cell_wall_albedo: float = 0.85          # work-cell walls (data rig: black curtain ~0.15)
    table_surface: str = "white"           # 'white' laminate or 'wood' (light varnished wood grain texture)
    cell_walls: str = "all"                # 'all' (rear + sides), 'rear', or 'none' (no shielding panels)
    rear_texture: str | None = None         # image file (assets/backdrops/...) shown on the rear panel, or on the room's back wall when cell_walls='none'
    wood_texture: str = "oak_veneer_01"
    wood_tile_m: float = 0.6                # texture repeat length on the table, m
    container_size: tuple = (0.15, 0.20, 0.03)   # outer x, y, depth (thin clear plastic boxes)
    container_wall: float = 0.0005
    container_offset_x: float = 0.29             # container centres at plate_x +- this (data rig: flanking the plate, ~12 cm from its rim)
    gripper_cal: dict = field(default_factory=lambda: dict(TRAIN_GRIPPER_CAL))
    include_wrist_cameras: bool = True


def _smooth(mesh: trimesh.Trimesh, angle_deg: float = 32.0):
    """Split vertices at sharp edges so smooth normals do not bleed across CAD creases."""
    try:
        return trimesh.graph.smooth_shade(mesh, angle=np.radians(angle_deg))
    except Exception:
        m = mesh.copy()
        m.unmerge_vertices()
        return m


def make_quad(p0, p1, p2, p3):
    v = np.array([p0, p1, p2, p3], dtype=np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return v, f


def make_box(center, size):
    b = trimesh.creation.box(extents=size)
    b.apply_translation(center)
    return b


def make_plate(diameter: float, segments: int = 128) -> trimesh.Trimesh:
    """Shallow round ceramic plate: flat well, rising rim, foot ring. Origin at the resting plane."""
    R = diameter / 2
    t = 0.0045  # wall thickness
    # (r, z) profile, outer surface bottom -> rim -> inner surface back to centre
    prof = [
        (0.0, 0.003), (0.55 * R, 0.003), (0.60 * R, 0.0), (0.66 * R, 0.0),  # underside + foot ring
        (0.68 * R, 0.005), (0.85 * R, 0.010), (0.96 * R, 0.020), (R, 0.026),  # outer flare
        (R - 0.4 * t, 0.028), (R - t, 0.026),  # rim lip
        (0.95 * R - t, 0.020 - t * 0.2), (0.84 * R - t, 0.011), (0.66 * R - t, 0.0055 + t * 0.4),  # inner flare
        (0.55 * R, 0.003 + t), (0.0, 0.003 + t),
    ]
    prof = np.array(prof)
    m = trimesh.creation.revolve(prof, sections=segments)
    if m.volume < 0:
        m.invert()
    return m


def make_container(size=(0.15, 0.20, 0.03), wall=0.0005) -> trimesh.Trimesh:
    """Open-top thin-walled box (bottom slab + 4 walls, non-overlapping), origin at the outer bottom centre."""
    sx, sy, sz = size
    parts = [make_box([0, 0, wall / 2], [sx, sy, wall])]
    for x in (-(sx - wall) / 2, (sx - wall) / 2):
        parts.append(make_box([x, 0, wall + (sz - wall) / 2], [wall, sy, sz - wall]))
    for y in (-(sy - wall) / 2, (sy - wall) / 2):
        parts.append(make_box([0, y, wall + (sz - wall) / 2], [sx - 2 * wall, wall, sz - wall]))
    return trimesh.util.concatenate(parts)


def load_apple(diameter: float):
    """Poly Haven food_apple_01 (CC0): returns (mesh z-up, bottom at z=0, scaled to `diameter`), albedo RGBA, roughness RGBA."""
    sc = trimesh.load(APPLE_GLTF)
    mesh = next(iter(sc.geometry.values())).copy()
    ext = mesh.bounds[1] - mesh.bounds[0]
    width = max(ext[0], ext[2])  # glTF is y-up
    mesh.apply_scale(diameter / width)
    mesh.apply_transform(tf(rpy=(np.pi / 2, 0, 0)))  # y-up -> z-up
    mesh.apply_translation([-(mesh.bounds[0][0] + mesh.bounds[1][0]) / 2, -(mesh.bounds[0][1] + mesh.bounds[1][1]) / 2, -mesh.bounds[0][2]])
    uv = np.array(mesh.visual.uv, dtype=np.float32)
    uv[:, 1] = 1.0 - uv[:, 1]  # glTF uv origin is top-left; our textures are uploaded top row first with v down
    tex_dir = os.path.join(os.path.dirname(APPLE_GLTF), "textures")
    albedo = np.array(Image.open(os.path.join(tex_dir, "food_apple_01_diff_1k.jpg")).convert("RGBA"))
    rough = np.array(Image.open(os.path.join(tex_dir, "food_apple_01_rough_1k.jpg")).convert("RGBA"))
    return mesh, uv, albedo, rough


class YamStation:
    """The full virtual rig. Set joint states in actuator radians, then render."""

    def __init__(self, cfg: StationConfig | None = None, renderer: Renderer | None = None, log_level: int = 1):
        self.cfg = cfg or StationConfig()
        self.r = renderer or Renderer(log_level=log_level)
        self.mat_ids: dict[str, int] = {}
        self.instances: dict[str, int] = {}
        self.link_instances: dict[str, list[tuple[int, np.ndarray]]] = {"left": [], "right": []}  # (instance, link, local T)
        self.object_meshes: dict[str, int] = {}
        self._build_environment()
        self._build_lights()
        self._build_arms()
        self._build_objects()
        self.r.upload_materials_and_lights()
        self.set_camera(self.cfg.camera)
        self.r.commit()

    # ------------------------------------------------------------------ materials
    def mat(self, m: Material) -> int:
        if m.name not in self.mat_ids:
            self.mat_ids[m.name] = self.r.add_material(m)
        return self.mat_ids[m.name]

    def replace_material(self, name: str, m: Material):
        """Swap the parameters of a named material (appearance domain randomisation)."""
        mid = self.mat_ids[name]
        self.r.materials[mid] = replace(m, name=name)
        self.r.upload_materials_and_lights()

    # ------------------------------------------------------------------ environment
    def _add_mesh_instance(self, name: str, mesh: trimesh.Trimesh, mat_id: int, T=None, smooth=False, uv=None):
        if smooth:
            mesh = _smooth(mesh)
            mid = self.r.add_mesh(mesh.vertices, mesh.faces, mesh.vertex_normals, uv)
        else:
            mid = self.r.add_mesh(mesh.vertices, mesh.faces, None, uv)
        inst = self.r.add_instance(mid, mat_id, np.eye(4) if T is None else T)
        self.instances[name] = inst
        return inst

    def _build_environment(self):
        c = self.cfg
        I = np.eye(4)
        z0 = -c.table_height
        z1 = c.room_height - c.table_height
        s = c.room_size / 2
        floor = self.mat(M.ROOM_FLOOR); wall = self.mat(M.ROOM_WALL)
        tex_mat = None
        if c.rear_texture:
            img = np.array(Image.open(os.path.join(_ROOT, "assets", "backdrops", c.rear_texture)).convert("RGBA"))
            tex_mat = self.mat(Material("rear_texture", base_color=(1.0, 1.0, 1.0), flags=MAT_LAMBERT | MAT_TWO_SIDED, tex_albedo=self.r.add_texture(img, srgb=True)))
        back_mid = tex_mat if (c.rear_texture and c.cell_walls == "none") else wall
        for name, quad, mid in [
            ("floor", make_quad([-s, -s, z0], [s, -s, z0], [s, s, z0], [-s, s, z0]), floor),
            ("ceiling", make_quad([-s, -s, z1], [-s, s, z1], [s, s, z1], [s, -s, z1]), wall),
            ("wall_back", make_quad([-s, s, z0], [s, s, z0], [s, s, z1], [-s, s, z1]), back_mid),
            ("wall_front", make_quad([-s, -s, z0], [-s, -s, z1], [s, -s, z1], [s, -s, z0]), wall),
            ("wall_left", make_quad([-s, -s, z0], [-s, s, z0], [-s, s, z1], [-s, -s, z1]), wall),
            ("wall_right", make_quad([s, -s, z0], [s, -s, z1], [s, s, z1], [s, s, z0]), wall),
        ]:
            v, f = quad
            uv = None
            if name == "wall_back" and mid == tex_mat:
                # photo on the 3 x 3 m room wall: image aspect 1.6 -> show it as a 3 m x 1.875 m band centred 1.2 m up
                zc = 1.2 - c.table_height; hh = (c.room_size / 1.6) / 2
                uv = np.array([[(x + s) / c.room_size, 1.0 - (z - (zc - hh)) / (2 * hh)] for x, y, z in v], np.float32)
            m = self.r.add_mesh(v, f, None, uv)
            self.instances[name] = self.r.add_instance(m, mid, I)
        # work surface slab (top face at z=0)
        table = make_box([0, c.table_y_center, -c.table_thickness / 2], [c.table_size[0], c.table_size[1], c.table_thickness])
        if c.table_surface in ("wood", "wood_matte"):
            tex_dir = os.path.join(_ROOT, "assets", "wood")
            albedo = np.array(Image.open(os.path.join(tex_dir, f"{c.wood_texture}_diff_2k.jpg")).convert("RGBA"))
            rough = np.array(Image.open(os.path.join(tex_dir, f"{c.wood_texture}_rough_2k.jpg")).convert("RGBA"))
            ta = self.r.add_texture(albedo, srgb=True); tr = self.r.add_texture(rough, srgb=False)
            # varnished wood: textured albedo, low-ish roughness scaled by the map, a clear coat for the varnish
            if c.table_surface == "wood":   # varnished: glossy clear coat over the grain
                wood = Material("table_wood", base_color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.45, specular=0.5, clearcoat=0.6, clearcoat_gloss=0.85,
                                tex_albedo=ta, tex_rough=tr)
            else:                            # matte: same grain, Lambertian-like (ablation of the specular highlights)
                wood = Material("table_wood_matte", base_color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.95, specular=0.15, tex_albedo=ta)
            uv = np.stack([table.vertices[:, 0] / c.wood_tile_m, table.vertices[:, 1] / c.wood_tile_m], axis=1).astype(np.float32)
            mid = self.r.add_mesh(table.vertices, table.faces, None, uv)
            self.instances["table"] = self.r.add_instance(mid, self.mat(wood), np.eye(4))
        else:
            self._add_mesh_instance("table", table, self.mat(M.TABLE_WHITE))
        # work cell: rear wall + two side walls abutting its edges, thin white panels
        cw = self.mat(replace(M.CELL_WHITE, name=f"work_cell_{c.cell_wall_albedo:.2f}", base_color=(c.cell_wall_albedo,) * 3))
        w, h, y, xc = c.cell_wall_w, c.cell_wall_h, c.rear_wall_dist, c.cell_x_center
        ln = c.cell_wall_len if c.cell_wall_len is not None else w
        if c.cell_walls in ("all", "rear"):
            v, f = make_quad([xc - w / 2, y, 0], [xc + w / 2, y, 0], [xc + w / 2, y, h], [xc - w / 2, y, h])
            if tex_mat is not None:   # photo on the rear panel: crop the 1.6:1 image to the 2:1 panel (central band)
                uv = np.array([[0, 0.81], [1, 0.81], [1, 0.19], [0, 0.19]], np.float32)
                self.instances["cell_rear"] = self.r.add_instance(self.r.add_mesh(v, f, None, uv), tex_mat, I)
            else:
                self.instances["cell_rear"] = self.r.add_instance(self.r.add_mesh(v, f), cw, I)
        if c.cell_walls == "all":
            for side, x in (("left", xc - w / 2), ("right", xc + w / 2)):
                q = make_quad([x, y - ln, 0], [x, y, 0], [x, y, h], [x, y - ln, h])
                self.instances[f"cell_{side}"] = self.r.add_instance(self.r.add_mesh(*q), cw, I)

    def _build_lights(self):
        c = self.cfg
        rgb = M.blackbody_rgb(c.light_cct)
        area = np.pi * c.light_disk_radius ** 2 if c.light_shape == "disk" else c.light_size[0] * c.light_size[1]
        L = M.lambertian_panel_radiance(c.light_lumens, area)   # same lumens per fixture regardless of shape
        z = c.room_height - c.table_height - c.light_drop
        fixture = self.mat(M.LIGHT_FIXTURE)
        for i, (lx, ly) in enumerate(c.light_positions):
            if c.light_shape == "disk":
                R = c.light_disk_radius; centre = np.array([lx, ly, z])
                lid = self.r.add_light(RectLight(centre, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), rgb * L, normal=[0, 0, -1], disk_radius=R))
                em = self.r.add_material(Material(f"light_{i}", (0, 0, 0), emission=tuple(rgb * L), light_id=lid))
                self.mat_ids[f"light_{i}"] = em
                nseg = 48; ang = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
                vtx = np.vstack([centre, centre + np.stack([R * np.cos(ang), R * np.sin(ang), np.zeros(nseg)], 1)]).astype(np.float32)
                fac = np.array([[0, 1 + (k + 1) % nseg, 1 + k] for k in range(nseg)], np.uint32)  # normal down (-z)
                self.instances[f"light_{i}"] = self.r.add_instance(self.r.add_mesh(vtx, fac), em, np.eye(4))
                frame = make_box([lx, ly, z + 0.02], [2 * R + 0.04, 2 * R + 0.04, 0.04])
                self._add_mesh_instance(f"light_frame_{i}", frame, fixture)
                continue
            corner = np.array([lx - c.light_size[0] / 2, ly - c.light_size[1] / 2, z])
            u = np.array([c.light_size[0], 0, 0]); v = np.array([0, c.light_size[1], 0])
            lid = self.r.add_light(RectLight(corner, u, v, rgb * L, normal=[0, 0, -1]))
            em = self.r.add_material(Material(f"light_{i}", (0, 0, 0), emission=tuple(rgb * L), light_id=lid))
            self.mat_ids[f"light_{i}"] = em
            vtx = np.array([corner, corner + u, corner + u + v, corner + v], np.float32)
            fac = np.array([[0, 2, 1], [0, 3, 2]], np.uint32)  # wound so the geometric normal points down (-z)
            self.instances[f"light_{i}"] = self.r.add_instance(self.r.add_mesh(vtx, fac), em, np.eye(4))
            # a slim housing frame around the panel
            frame = make_box([lx, ly, z + 0.02], [c.light_size[0] + 0.04, c.light_size[1] + 0.04, 0.04])
            self._add_mesh_instance(f"light_frame_{i}", frame, fixture)

    # ------------------------------------------------------------------ arms
    def _build_arms(self):
        c = self.cfg
        self.arms: dict[str, YamArm] = {}
        alu, plastic, tip, cam = self.mat(M.BLACK_ANODIZED), self.mat(M.WHITE_PLASTIC), self.mat(M.GRIPPER_TIP), self.mat(M.BLACK_PLASTIC)
        link_mesh_cache: dict[str, list[tuple[int, str, int]]] = {}  # link -> [(mesh_id, part, material)]
        for side, x in (("left", -c.arm_spacing / 2), ("right", c.arm_spacing / 2)):
            base = tf(xyz=(x, 0, c.arm_base_height), rpy=(0, 0, np.pi / 2))  # arm +x (reach direction) -> world +y
            if c.arm_base_pitch_deg:
                base = tf(rpy=(np.radians(c.arm_base_pitch_deg), 0, 0)) @ base   # rotate about world x: +y (reach) tips toward +z
            arm = YamArm(YAM_URDF, c.gripper_cal[side], base_tf=base)
            self.arms[side] = arm
            for lname, link in arm.urdf.links.items():
                for mesh_path, vis_T in link.visuals:
                    if lname not in link_mesh_cache:
                        raw = trimesh.load(mesh_path, force="mesh")
                        parts = []
                        if lname in COVER_Z_RANGE:
                            zc = raw.triangles_center[:, 2]
                            lo, hi = COVER_Z_RANGE[lname]
                            sel = (zc > lo) & (zc < hi)
                            cover = raw.submesh([np.where(sel)[0]], append=True)
                            metal = raw.submesh([np.where(~sel)[0]], append=True)
                            parts = [(cover, "cover", plastic), (metal, "metal", alu)]
                        elif lname.startswith("tip"):
                            parts = [(raw, "finger", tip)]
                        else:
                            parts = [(raw, "body", alu)]
                        entries = []
                        for pm, pname, pmat in parts:
                            pm = pm.copy(); pm.apply_transform(vis_T)  # bake the visual origin into the link frame
                            pm = _smooth(pm)
                            entries.append((self.r.add_mesh(pm.vertices, pm.faces, pm.vertex_normals), pname, pmat))
                        link_mesh_cache[lname] = entries
                    for mesh_id, pname, pmat in link_mesh_cache[lname]:
                        inst = self.r.add_instance(mesh_id, pmat, np.eye(4))
                        self.instances[f"{side}_{lname}_{pname}"] = inst
                        self.link_instances[side].append((inst, lname, np.eye(4)))
            if c.include_wrist_cameras:
                if "wrist_cam" not in link_mesh_cache:
                    wm = _smooth(trimesh.load(WRIST_CAM_STL, force="mesh"))
                    link_mesh_cache["wrist_cam"] = [(self.r.add_mesh(wm.vertices, wm.faces, wm.vertex_normals), "cam", cam)]
                cam_T = tf(xyz=(-0.07043497874827391, 0.0, -0.07700592901799347), rpy=(2.70525, 0.0, 1.5707963267814136))  # station URDF camera_joint
                mesh_id, _, pmat = link_mesh_cache["wrist_cam"][0]
                inst = self.r.add_instance(mesh_id, pmat, np.eye(4))
                self.instances[f"{side}_wrist_cam"] = inst
                self.link_instances[side].append((inst, "gripper", cam_T))
        self.arm_state = {"left": np.array([0, 0, 0, 0, 0, 0, c.gripper_cal["left"].open_rad]),
                          "right": np.array([0, 0, 0, 0, 0, 0, c.gripper_cal["right"].open_rad])}
        self._pose_arms()

    def _pose_arms(self):
        for side, arm in self.arms.items():
            poses = arm.link_poses(self.arm_state[side])
            for inst, lname, local in self.link_instances[side]:
                self.r.set_instance_transform(inst, poses[lname] @ local)

    def set_arm_state(self, left: np.ndarray | None = None, right: np.ndarray | None = None):
        """7-vectors [j1..j6 rad, gripper motor rad]; gripper radians map through the calibration."""
        if left is not None:
            self.arm_state["left"] = np.asarray(left, float)
        if right is not None:
            self.arm_state["right"] = np.asarray(right, float)
        self._pose_arms()

    def set_gripper_open_state(self, side: str, g: float):
        self.arm_state[side][6] = self.cfg.gripper_cal[side].to_rad(g)
        self._pose_arms()

    def link_pose(self, side: str, link: str) -> np.ndarray:
        return self.arms[side].link_poses(self.arm_state[side])[link]

    # ------------------------------------------------------------------ objects
    def _build_objects(self):
        c = self.cfg
        plate = make_plate(c.plate_diameter)
        self.plate_top_z = 0.003 + 0.0045  # inner well surface height above the table
        self._add_mesh_instance("plate", plate, self.mat(M.CERAMIC_PLATE), smooth=True)
        try:
            mesh, uv, albedo, rough = load_apple(c.apple_diameter)
            ta = self.r.add_texture(albedo, srgb=True)
            tr = self.r.add_texture(rough, srgb=False)
            apple_mat = replace(M.APPLE, tex_albedo=ta, tex_rough=tr)
            m = _smooth(mesh, 60.0) if mesh.visual.uv is None else mesh
            mid = self.r.add_mesh(m.vertices, m.faces, m.vertex_normals, uv)
            self.instances["apple"] = self.r.add_instance(mid, self.mat(apple_mat), np.eye(4))
            for k in range(1, max(1, int(c.n_apples))):
                self.instances[f"apple{k}"] = self.r.add_instance(mid, self.mat(apple_mat), np.eye(4))
        except Exception as e:  # fall back to a red ball
            print("apple model unavailable, using red sphere:", e)
            ball = trimesh.creation.icosphere(subdivisions=4, radius=c.apple_diameter / 2)
            ball.apply_translation([0, 0, c.apple_diameter / 2])
            self._add_mesh_instance("apple", ball, self.mat(M.APPLE_FALLBACK), smooth=True)
            for k in range(1, max(1, int(c.n_apples))):
                self._add_mesh_instance(f"apple{k}", ball, self.mat(M.APPLE_FALLBACK), smooth=True)
        self.object_poses = {}
        self.container_names = []
        if c.containers:
            cm = make_container(c.container_size, c.container_wall)
            for name, dx in (("container_left", -c.container_offset_x), ("container_right", c.container_offset_x)):
                self._add_mesh_instance(name, cm, self.mat(M.CLEAR_PLASTIC))
                self.container_names.append(name)
        self.set_object_pose("plate", tf(xyz=(c.plate_xy[0], c.plate_xy[1], 0.0)))
        for name in self.container_names:
            dx = -c.container_offset_x if name.endswith("left") else c.container_offset_x
            self.set_object_pose(name, tf(xyz=(c.plate_xy[0] + dx, c.plate_xy[1], 0.0003)))  # tiny lift: avoid coplanar faces with the table
        self.set_object_pose("apple", tf(xyz=(c.plate_xy[0], c.plate_xy[1], self.plate_top_z), rpy=(0, 0, 0.7)))
        for k in range(1, max(1, int(c.n_apples))):
            self.set_object_pose(f"apple{k}", tf(xyz=(0.9 + 0.1 * k, -0.6, -0.7)))

    def set_object_pose(self, name: str, T: np.ndarray):
        self.object_poses[name] = np.asarray(T, float)
        self.r.set_instance_transform(self.instances[name], T)

    # ------------------------------------------------------------------ camera / render
    def set_camera(self, cam):
        self.cfg.camera = cam
        eye, look, up = cam.pose(self) if getattr(cam, "needs_station", False) else cam.pose()
        tx, ty = cam.tan_half()
        self.r.set_camera(eye, look, up, tx, ty)
        self.r.set_distortion(getattr(cam, "distortion", (0, 0, 0, 0, 0)), getattr(cam, "principal_offset", (0.0, 0.0)))

    def render(self, width: int = 1280, spp: int = 256, seed: int = 0, max_bounces: int = 6, clamp: float = 0.0, exposure: float | None = None, denoise: bool = False):
        """Returns (rgb8 [h,w,3], linear rgb [h,w,3], aux [h,w,4]: depth m, instance id, prim id, hit)."""
        w, h = self.cfg.camera.resolution(width)
        if getattr(self.cfg.camera, "needs_station", False):
            self.set_camera(self.cfg.camera)
        self.r.commit()
        lin, aux = self.r.render(w, h, spp=spp, seed=seed, max_bounces=max_bounces, clamp=clamp, denoise=denoise)
        ex = self.cfg.exposure if exposure is None else exposure
        disp = lin
        if self.cfg.white_balance:
            disp = lin / M.blackbody_rgb(self.cfg.light_cct)[None, None, :]
        return tonemap(disp, exposure=ex), lin, aux

    def render_blur(self, poses: list, width: int = 1280, spp: int = 256, seed: int = 0, max_bounces: int = 6, clamp: float = 0.0,
                    exposure: float | None = None, denoise: bool = False):
        """Motion blur by multi-pose accumulation: `poses` = list of (arm_state: {side: 7-vector}, object_poses: {name: 4x4})
        sampled across the exposure window; the samples are split evenly over them (camera follows the arm state for a
        wrist camera). The scene is left in the LAST pose. Same return as render()."""
        w, h = self.cfg.camera.resolution(width)
        self.r.render_begin(w, h)
        K = len(poses); per = [spp // K + (1 if i < spp % K else 0) for i in range(K)]
        for i, (arm_state, obj) in enumerate(poses):
            if per[i] == 0:
                continue
            self.set_arm_state(arm_state.get("left"), arm_state.get("right"))
            for name, T in obj.items():
                self.set_object_pose(name, T)
            self.set_camera(self.cfg.camera)
            self.r.commit()
            self.r.render_accumulate(w, h, per[i], seed=seed + 7919 * i, max_bounces=max_bounces, clamp=clamp)
        lin, aux = self.r.render_finish(w, h, denoise=denoise)
        ex = self.cfg.exposure if exposure is None else exposure
        disp = lin
        if self.cfg.white_balance:
            disp = lin / M.blackbody_rgb(self.cfg.light_cct)[None, None, :]
        return tonemap(disp, exposure=ex), lin, aux

    def instance_mask(self, aux: np.ndarray, name: str) -> np.ndarray:
        return aux[..., 1] == self.instances[name]
