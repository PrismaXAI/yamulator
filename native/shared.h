// Shared host/device definitions for the yamtrace OptiX path tracer.
#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define YT_MAT_LAMBERT   1   // flags: pure Lambertian (walls, floor)
#define YT_MAT_TWO_SIDED 2   // flags: shade both sides (thin quads)
#define YT_MAT_DIELECTRIC 4  // flags: smooth glass/clear plastic (refract + Fresnel), ior in `ior`

// Plain 4-byte-field layout so Python ctypes can mirror it exactly.
struct YtMaterial {
    float base_r, base_g, base_b;
    float metallic, roughness, specular, spec_tint;
    float clearcoat, clearcoat_gloss, sheen, subsurface;
    float emit_r, emit_g, emit_b;
    int32_t tex_albedo, tex_rough, tex_normal;   // texture slot or -1
    int32_t flags;
    int32_t light_id;                            // >=0 if this material is an area light
    float uv_scale;
    float ior;                                   // index of refraction for YT_MAT_DIELECTRIC
};

struct YtRectLight {
    float cx, cy, cz;      // corner
    float ux, uy, uz;      // edge vector u (full length)
    float vx, vy, vz;      // edge vector v (full length)
    float nx, ny, nz;      // emitting normal (one-sided)
    float rad_r, rad_g, rad_b; // radiance (W/sr/m^2, arbitrary scale)
    float area;
    int32_t ltype;             // 0 = rectangle (corner + edges), 1 = disk (centre, u/v = unit axes * radius)
    float radius;
};

struct YtHitGroupData {
    float3* vertices;
    float3* normals;   // may be null -> geometric normals
    float2* uvs;       // may be null
    uint3*  indices;
    int32_t material_id;
    int32_t instance_id;
    int32_t pad0, pad1;
};

struct YtCamera {
    float3 eye, U, V, W;   // U,V already scaled by tan(half fov)
    float thx, thy;        // tan(half fov) per axis (|U|, |V|)
    float k1, k2, k3, p1, p2;   // OpenCV Brown-Conrady distortion (applied to the pinhole image; rays are undistorted per sample)
    float cx_off, cy_off;  // principal point offset as a fraction of the half image (+x right, +y down), 0 = centred
    int32_t distort;       // 1 = any distortion coefficient non-zero
};

struct YtLaunchParams {
    uint32_t width, height;
    uint32_t spp;            // samples per pixel in this launch
    uint32_t sample_offset;  // total samples already accumulated
    uint32_t seed;
    int32_t  max_bounces;
    float    clamp;          // radiance clamp for indirect bounces (0 = off)
    float4*  accum;          // running sum of radiance
    float4*  aux;            // (depth, instance_id, prim_id, hit) from first sample
    float4*  albedo;         // denoiser guide: first-hit albedo (accumulated)
    float4*  normal;         // denoiser guide: first-hit shading normal in camera space (accumulated)
    YtCamera cam;
    YtMaterial* materials;
    YtRectLight* lights;
    int32_t n_lights;
    cudaTextureObject_t* textures;
    OptixTraversableHandle handle;
};

struct YtRaygenData {};
struct YtMissData { float3 background; };
