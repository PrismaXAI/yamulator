// C API of the yamtrace OptiX path tracer (loaded from Python via ctypes).
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct YtContext YtContext;
YtContext* yt_create(const char* device_program_path, int log_level);
void  yt_destroy(YtContext* c);
const char* yt_last_error(YtContext* c);
// meshes: vertices float[3n], normals float[3n] or NULL, uvs float[2n] or NULL, indices uint32[3m]
int   yt_add_mesh(YtContext* c, const float* vertices, uint32_t n_vertices, const float* normals, const float* uvs, const uint32_t* indices, uint32_t n_triangles);
// textures: RGBA8, row-major, top row first; srgb=1 decodes sRGB -> linear on fetch
int   yt_add_texture(YtContext* c, const uint8_t* rgba, uint32_t width, uint32_t height, int srgb);
// instances: xform = 3x4 row-major affine (object -> world)
int   yt_add_instance(YtContext* c, int mesh_id, int material_id, const float* xform12);
int   yt_set_instance_transform(YtContext* c, int instance_id, const float* xform12);
int   yt_set_instance_material(YtContext* c, int instance_id, int material_id);
// visibility mask bits: 1 = opaque (occludes shadow rays), 2 = dielectric (radiance rays only)
int   yt_set_instance_mask(YtContext* c, int instance_id, unsigned mask);
// materials/lights: arrays of YtMaterial / YtRectLight (see shared.h layout)
int   yt_set_materials(YtContext* c, const void* materials, uint32_t n);
int   yt_set_lights(YtContext* c, const void* lights, uint32_t n);
// camera: pinhole; tan_half_x/y = tan of half field of view horizontally/vertically
int   yt_set_camera(YtContext* c, const float* eye, const float* look_dir, const float* up, float tan_half_x, float tan_half_y);
int   yt_set_background(YtContext* c, float r, float g, float b);
// build/refresh acceleration structures + SBT after adding instances or changing transforms
int   yt_commit(YtContext* c);
// render: out_rgb float[w*h*3] linear radiance mean, out_aux float[w*h*4] (depth, instance_id, prim_id, hit)
// denoise: 0 = raw, 1 = OptiX AI denoiser (HDR, albedo+normal guided)
int   yt_render(YtContext* c, uint32_t width, uint32_t height, uint32_t spp, uint32_t seed, int max_bounces, float clamp, int denoise, float* out_rgb, float* out_aux);
// multi-pose accumulation (motion blur): begin -> [set transforms/camera; accumulate]* -> finish
int   yt_render_begin(YtContext* c, uint32_t width, uint32_t height);
int   yt_render_accumulate(YtContext* c, uint32_t width, uint32_t height, uint32_t spp, uint32_t seed, int max_bounces, float clamp);
int   yt_render_finish(YtContext* c, uint32_t width, uint32_t height, int denoise, float* out_rgb, float* out_aux);
#ifdef __cplusplus
}
#endif
