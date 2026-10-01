#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include "shared.h"
#include "yamtrace.h"

template <typename T> struct SbtRecord { __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE]; T data; };
typedef SbtRecord<YtRaygenData> RaygenRecord;
typedef SbtRecord<YtMissData> MissRecord;
typedef SbtRecord<YtHitGroupData> HitRecord;

struct Mesh {
    CUdeviceptr d_vertices = 0, d_normals = 0, d_uvs = 0, d_indices = 0, d_gas = 0;
    OptixTraversableHandle gas = 0;
    uint32_t n_vertices = 0, n_triangles = 0;
};
struct Instance { int mesh_id; int material_id; float xform[12]; unsigned mask; };

struct YtContext {
    std::string err;
    int log_level = 2;
    OptixDeviceContext ctx = nullptr;
    OptixModule module = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixProgramGroup pg_raygen = nullptr, pg_miss_rad = nullptr, pg_miss_shadow = nullptr, pg_hit_rad = nullptr, pg_hit_shadow = nullptr;
    std::vector<Mesh> meshes;
    std::vector<Instance> instances;
    std::vector<cudaArray_t> tex_arrays;
    std::vector<cudaTextureObject_t> textures;
    CUdeviceptr d_textures = 0, d_materials = 0, d_lights = 0, d_ias = 0, d_ias_tmp = 0, d_instances = 0;
    size_t ias_size = 0, ias_tmp_size = 0;
    uint32_t n_materials = 0, n_lights = 0;
    OptixTraversableHandle ias = 0;
    OptixShaderBindingTable sbt = {};
    CUdeviceptr d_raygen = 0, d_miss = 0, d_hit = 0;
    size_t hit_records = 0;
    YtLaunchParams params = {};
    CUdeviceptr d_params = 0, d_accum = 0, d_aux = 0, d_albedo = 0, d_normal = 0;
    size_t accum_pixels = 0;
    uint32_t accum_samples = 0;   // samples accumulated since yt_render_begin (multi-pose motion blur)
    OptixDenoiser denoiser = nullptr; uint32_t dn_w = 0, dn_h = 0;
    CUdeviceptr d_dn_state = 0, d_dn_scratch = 0, d_dn_in = 0, d_dn_out = 0, d_dn_alb = 0, d_dn_nrm = 0, d_dn_intensity = 0;
    size_t dn_state_size = 0, dn_scratch_size = 0;
    float3 background = make_float3(0.f, 0.f, 0.f);
    bool dirty = true;
    CUstream stream = 0;
};

static void log_cb(unsigned int level, const char* tag, const char* msg, void* data) {
    YtContext* c = (YtContext*)data;
    if ((int)level <= c->log_level) fprintf(stderr, "[optix][%u][%s] %s\n", level, tag, msg);
}
#define OPTIX_CHECK(c, call) do { OptixResult r_ = (call); if (r_ != OPTIX_SUCCESS) { (c)->err = std::string(#call) + " failed: " + optixGetErrorName(r_); fprintf(stderr, "yamtrace: %s\n", (c)->err.c_str()); return -1; } } while (0)
#define CUDA_CHECK(c, call) do { cudaError_t r_ = (call); if (r_ != cudaSuccess) { (c)->err = std::string(#call) + " failed: " + cudaGetErrorString(r_); fprintf(stderr, "yamtrace: %s\n", (c)->err.c_str()); return -1; } } while (0)

static int build_pipeline(YtContext* c, const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { c->err = std::string("cannot open device program ") + path; return -1; }
    std::stringstream sstr; sstr << f.rdbuf(); std::string code = sstr.str();
    OptixModuleCompileOptions mco = {};
    mco.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions pco = {};
    pco.usesMotionBlur = 0;
    pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pco.numPayloadValues = 2;
    pco.numAttributeValues = 2;
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pco.pipelineLaunchParamsVariableName = "params";
    pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    char log[4096]; size_t logsz = sizeof(log);
    OptixResult r = optixModuleCreate(c->ctx, &mco, &pco, code.data(), code.size(), log, &logsz, &c->module);
    if (logsz > 1 && c->log_level >= 3) fprintf(stderr, "[module] %s\n", log);
    if (r != OPTIX_SUCCESS) { c->err = std::string("optixModuleCreate failed: ") + optixGetErrorName(r) + " " + log; fprintf(stderr, "yamtrace: %s\n", c->err.c_str()); return -1; }
    OptixProgramGroupOptions pgo = {};
    OptixProgramGroupDesc d[5] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module = c->module; d[0].raygen.entryFunctionName = "__raygen__rg";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module = c->module; d[1].miss.entryFunctionName = "__miss__radiance";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d[2].miss.module = c->module; d[2].miss.entryFunctionName = "__miss__shadow";
    d[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[3].hitgroup.moduleCH = c->module; d[3].hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    d[4].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; // shadow: no programs
    OptixProgramGroup pgs[5];
    logsz = sizeof(log);
    OPTIX_CHECK(c, optixProgramGroupCreate(c->ctx, d, 5, &pgo, log, &logsz, pgs));
    c->pg_raygen = pgs[0]; c->pg_miss_rad = pgs[1]; c->pg_miss_shadow = pgs[2]; c->pg_hit_rad = pgs[3]; c->pg_hit_shadow = pgs[4];
    OptixPipelineLinkOptions plo = {}; plo.maxTraceDepth = 2;
    logsz = sizeof(log);
    OPTIX_CHECK(c, optixPipelineCreate(c->ctx, &pco, &plo, pgs, 5, log, &logsz, &c->pipeline));
    OptixStackSizes ss = {};
    for (int i = 0; i < 5; ++i) OPTIX_CHECK(c, optixUtilAccumulateStackSizes(pgs[i], &ss, c->pipeline));
    uint32_t dc_trav, dc_state, cont;
    OPTIX_CHECK(c, optixUtilComputeStackSizes(&ss, 2, 0, 0, &dc_trav, &dc_state, &cont));
    OPTIX_CHECK(c, optixPipelineSetStackSize(c->pipeline, dc_trav, dc_state, cont, 2));
    // raygen + miss records
    RaygenRecord rg; OPTIX_CHECK(c, optixSbtRecordPackHeader(c->pg_raygen, &rg));
    CUDA_CHECK(c, cudaMalloc((void**)&c->d_raygen, sizeof(rg))); CUDA_CHECK(c, cudaMemcpy((void*)c->d_raygen, &rg, sizeof(rg), cudaMemcpyHostToDevice));
    MissRecord ms[2]; OPTIX_CHECK(c, optixSbtRecordPackHeader(c->pg_miss_rad, &ms[0])); OPTIX_CHECK(c, optixSbtRecordPackHeader(c->pg_miss_shadow, &ms[1]));
    ms[0].data.background = c->background; ms[1].data.background = c->background;
    CUDA_CHECK(c, cudaMalloc((void**)&c->d_miss, sizeof(ms))); CUDA_CHECK(c, cudaMemcpy((void*)c->d_miss, ms, sizeof(ms), cudaMemcpyHostToDevice));
    c->sbt.raygenRecord = c->d_raygen;
    c->sbt.missRecordBase = c->d_miss; c->sbt.missRecordStrideInBytes = sizeof(MissRecord); c->sbt.missRecordCount = 2;
    CUDA_CHECK(c, cudaMalloc((void**)&c->d_params, sizeof(YtLaunchParams)));
    return 0;
}

extern "C" YtContext* yt_create(const char* device_program_path, int log_level) {
    YtContext* c = new YtContext(); c->log_level = log_level;
    cudaFree(0);
    if (optixInit() != OPTIX_SUCCESS) { fprintf(stderr, "yamtrace: optixInit failed\n"); delete c; return nullptr; }
    OptixDeviceContextOptions o = {}; o.logCallbackFunction = log_cb; o.logCallbackData = c; o.logCallbackLevel = 4;
    if (optixDeviceContextCreate(0, &o, &c->ctx) != OPTIX_SUCCESS) { fprintf(stderr, "yamtrace: context create failed\n"); delete c; return nullptr; }
    if (build_pipeline(c, device_program_path) != 0) { delete c; return nullptr; }
    return c;
}
extern "C" void yt_destroy(YtContext* c) {
    if (!c) return;
    for (auto& m : c->meshes) { cudaFree((void*)m.d_vertices); cudaFree((void*)m.d_normals); cudaFree((void*)m.d_uvs); cudaFree((void*)m.d_indices); cudaFree((void*)m.d_gas); }
    for (auto t : c->textures) cudaDestroyTextureObject(t);
    for (auto a : c->tex_arrays) cudaFreeArray(a);
    cudaFree((void*)c->d_textures); cudaFree((void*)c->d_materials); cudaFree((void*)c->d_lights); cudaFree((void*)c->d_ias); cudaFree((void*)c->d_ias_tmp);
    cudaFree((void*)c->d_albedo); cudaFree((void*)c->d_normal);
    if (c->denoiser) optixDenoiserDestroy(c->denoiser);
    cudaFree((void*)c->d_dn_state); cudaFree((void*)c->d_dn_scratch); cudaFree((void*)c->d_dn_in); cudaFree((void*)c->d_dn_out); cudaFree((void*)c->d_dn_alb); cudaFree((void*)c->d_dn_nrm); cudaFree((void*)c->d_dn_intensity);
    cudaFree((void*)c->d_instances); cudaFree((void*)c->d_raygen); cudaFree((void*)c->d_miss); cudaFree((void*)c->d_hit); cudaFree((void*)c->d_params); cudaFree((void*)c->d_accum); cudaFree((void*)c->d_aux);
    if (c->pipeline) optixPipelineDestroy(c->pipeline);
    if (c->module) optixModuleDestroy(c->module);
    if (c->ctx) optixDeviceContextDestroy(c->ctx);
    delete c;
}
extern "C" const char* yt_last_error(YtContext* c) { return c ? c->err.c_str() : "null context"; }

extern "C" int yt_add_mesh(YtContext* c, const float* vertices, uint32_t nv, const float* normals, const float* uvs, const uint32_t* indices, uint32_t nt) {
    Mesh m; m.n_vertices = nv; m.n_triangles = nt;
    CUDA_CHECK(c, cudaMalloc((void**)&m.d_vertices, nv * sizeof(float3))); CUDA_CHECK(c, cudaMemcpy((void*)m.d_vertices, vertices, nv * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(c, cudaMalloc((void**)&m.d_indices, nt * sizeof(uint3))); CUDA_CHECK(c, cudaMemcpy((void*)m.d_indices, indices, nt * sizeof(uint3), cudaMemcpyHostToDevice));
    if (normals) { CUDA_CHECK(c, cudaMalloc((void**)&m.d_normals, nv * sizeof(float3))); CUDA_CHECK(c, cudaMemcpy((void*)m.d_normals, normals, nv * sizeof(float3), cudaMemcpyHostToDevice)); }
    if (uvs) { CUDA_CHECK(c, cudaMalloc((void**)&m.d_uvs, nv * sizeof(float2))); CUDA_CHECK(c, cudaMemcpy((void*)m.d_uvs, uvs, nv * sizeof(float2), cudaMemcpyHostToDevice)); }
    OptixAccelBuildOptions ao = {}; ao.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixBuildInput bi = {}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    uint32_t flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; bi.triangleArray.vertexStrideInBytes = sizeof(float3);
    bi.triangleArray.numVertices = nv; bi.triangleArray.vertexBuffers = &m.d_vertices;
    bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3; bi.triangleArray.indexStrideInBytes = sizeof(uint3);
    bi.triangleArray.numIndexTriplets = nt; bi.triangleArray.indexBuffer = m.d_indices;
    bi.triangleArray.flags = &flags; bi.triangleArray.numSbtRecords = 1;
    OptixAccelBufferSizes sz; OPTIX_CHECK(c, optixAccelComputeMemoryUsage(c->ctx, &ao, &bi, 1, &sz));
    CUdeviceptr d_tmp, d_out; CUDA_CHECK(c, cudaMalloc((void**)&d_tmp, sz.tempSizeInBytes)); CUDA_CHECK(c, cudaMalloc((void**)&d_out, sz.outputSizeInBytes));
    CUdeviceptr d_csize; CUDA_CHECK(c, cudaMalloc((void**)&d_csize, sizeof(size_t)));
    OptixAccelEmitDesc em = {}; em.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE; em.result = d_csize;
    OPTIX_CHECK(c, optixAccelBuild(c->ctx, 0, &ao, &bi, 1, d_tmp, sz.tempSizeInBytes, d_out, sz.outputSizeInBytes, &m.gas, &em, 1));
    size_t csize; CUDA_CHECK(c, cudaMemcpy(&csize, (void*)d_csize, sizeof(size_t), cudaMemcpyDeviceToHost));
    if (csize < sz.outputSizeInBytes) {
        CUDA_CHECK(c, cudaMalloc((void**)&m.d_gas, csize));
        OPTIX_CHECK(c, optixAccelCompact(c->ctx, 0, m.gas, m.d_gas, csize, &m.gas));
        cudaFree((void*)d_out);
    } else m.d_gas = d_out;
    cudaFree((void*)d_tmp); cudaFree((void*)d_csize);
    c->meshes.push_back(m); c->dirty = true;
    return (int)c->meshes.size() - 1;
}

extern "C" int yt_add_texture(YtContext* c, const uint8_t* rgba, uint32_t w, uint32_t h, int srgb) {
    cudaChannelFormatDesc cd = cudaCreateChannelDesc<uchar4>();
    cudaArray_t arr; CUDA_CHECK(c, cudaMallocArray(&arr, &cd, w, h));
    CUDA_CHECK(c, cudaMemcpy2DToArray(arr, 0, 0, rgba, w * 4, w * 4, h, cudaMemcpyHostToDevice));
    cudaResourceDesc rd = {}; rd.resType = cudaResourceTypeArray; rd.res.array.array = arr;
    cudaTextureDesc td = {}; td.addressMode[0] = cudaAddressModeWrap; td.addressMode[1] = cudaAddressModeWrap;
    td.filterMode = cudaFilterModeLinear; td.readMode = cudaReadModeNormalizedFloat; td.normalizedCoords = 1; td.sRGB = srgb ? 1 : 0;
    cudaTextureObject_t t; CUDA_CHECK(c, cudaCreateTextureObject(&t, &rd, &td, nullptr));
    c->tex_arrays.push_back(arr); c->textures.push_back(t);
    if (c->d_textures) cudaFree((void*)c->d_textures);
    CUDA_CHECK(c, cudaMalloc((void**)&c->d_textures, c->textures.size() * sizeof(cudaTextureObject_t)));
    CUDA_CHECK(c, cudaMemcpy((void*)c->d_textures, c->textures.data(), c->textures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));
    return (int)c->textures.size() - 1;
}

extern "C" int yt_add_instance(YtContext* c, int mesh_id, int material_id, const float* xf) {
    if (mesh_id < 0 || mesh_id >= (int)c->meshes.size()) { c->err = "bad mesh id"; return -1; }
    Instance in; in.mesh_id = mesh_id; in.material_id = material_id; memcpy(in.xform, xf, 12 * sizeof(float)); in.mask = 1;
    c->instances.push_back(in); c->dirty = true;
    return (int)c->instances.size() - 1;
}
extern "C" int yt_set_instance_transform(YtContext* c, int id, const float* xf) {
    if (id < 0 || id >= (int)c->instances.size()) { c->err = "bad instance id"; return -1; }
    memcpy(c->instances[id].xform, xf, 12 * sizeof(float)); c->dirty = true; return 0;
}
extern "C" int yt_set_instance_mask(YtContext* c, int id, unsigned mask) {
    if (id < 0 || id >= (int)c->instances.size()) { c->err = "bad instance id"; return -1; }
    c->instances[id].mask = mask; c->dirty = true; return 0;
}
extern "C" int yt_set_instance_material(YtContext* c, int id, int material_id) {
    if (id < 0 || id >= (int)c->instances.size()) { c->err = "bad instance id"; return -1; }
    c->instances[id].material_id = material_id; c->dirty = true; return 0;
}
extern "C" int yt_set_materials(YtContext* c, const void* mats, uint32_t n) {
    if (c->d_materials) cudaFree((void*)c->d_materials);
    CUDA_CHECK(c, cudaMalloc((void**)&c->d_materials, n * sizeof(YtMaterial)));
    CUDA_CHECK(c, cudaMemcpy((void*)c->d_materials, mats, n * sizeof(YtMaterial), cudaMemcpyHostToDevice));
    c->n_materials = n; return 0;
}
extern "C" int yt_set_lights(YtContext* c, const void* lights, uint32_t n) {
    if (c->d_lights) { cudaFree((void*)c->d_lights); c->d_lights = 0; }
    if (n > 0) { CUDA_CHECK(c, cudaMalloc((void**)&c->d_lights, n * sizeof(YtRectLight))); CUDA_CHECK(c, cudaMemcpy((void*)c->d_lights, lights, n * sizeof(YtRectLight), cudaMemcpyHostToDevice)); }
    c->n_lights = n; return 0;
}
static float3 f3(const float* p) { return make_float3(p[0], p[1], p[2]); }
static float3 f3norm(float3 v) { float l = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); return make_float3(v.x / l, v.y / l, v.z / l); }
static float3 f3cross(float3 a, float3 b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
extern "C" int yt_set_camera(YtContext* c, const float* eye, const float* look, const float* up, float thx, float thy) {
    float3 W = f3norm(f3(look));
    float3 U = f3norm(f3cross(W, f3(up)));
    float3 V = f3cross(U, W);
    c->params.cam.eye = f3(eye); c->params.cam.W = W;
    c->params.cam.U = make_float3(U.x * thx, U.y * thx, U.z * thx);
    c->params.cam.V = make_float3(V.x * thy, V.y * thy, V.z * thy);
    c->params.cam.thx = thx; c->params.cam.thy = thy;
    return 0;
}
// Lens intrinsics deviations: OpenCV distortion coefficients [k1,k2,p1,p2,k3] (OpenCV order, image y down) and a
// principal point offset in fractions of the half image. The pinhole focal length (tan half fov) stays as set.
extern "C" int yt_set_distortion(YtContext* c, const float* k5, float cx_off, float cy_off) {
    YtCamera& cam = c->params.cam;
    cam.k1 = k5[0]; cam.k2 = k5[1]; cam.p1 = k5[2]; cam.p2 = k5[3]; cam.k3 = k5[4];
    cam.cx_off = cx_off; cam.cy_off = cy_off;
    cam.distort = (cam.k1 != 0.0f || cam.k2 != 0.0f || cam.k3 != 0.0f || cam.p1 != 0.0f || cam.p2 != 0.0f) ? 1 : 0;
    return 0;
}
extern "C" int yt_set_background(YtContext* c, float r, float g, float b) { c->background = make_float3(r, g, b); return 0; }

extern "C" int yt_commit(YtContext* c) {
    if (!c->dirty) return 0;
    const size_t n = c->instances.size();
    if (n == 0) { c->err = "no instances"; return -1; }
    // SBT hit records: 2 per instance (radiance, shadow)
    std::vector<HitRecord> recs(2 * n);
    for (size_t i = 0; i < n; ++i) {
        const Instance& in = c->instances[i]; const Mesh& m = c->meshes[in.mesh_id];
        YtHitGroupData hd = {};
        hd.vertices = (float3*)m.d_vertices; hd.normals = (float3*)m.d_normals; hd.uvs = (float2*)m.d_uvs; hd.indices = (uint3*)m.d_indices;
        hd.material_id = in.material_id; hd.instance_id = (int)i;
        if (optixSbtRecordPackHeader(c->pg_hit_rad, &recs[2 * i]) != OPTIX_SUCCESS) return -1;
        if (optixSbtRecordPackHeader(c->pg_hit_shadow, &recs[2 * i + 1]) != OPTIX_SUCCESS) return -1;
        recs[2 * i].data = hd; recs[2 * i + 1].data = hd;
    }
    if (c->hit_records != 2 * n) { if (c->d_hit) cudaFree((void*)c->d_hit); CUDA_CHECK(c, cudaMalloc((void**)&c->d_hit, recs.size() * sizeof(HitRecord))); c->hit_records = 2 * n; }
    CUDA_CHECK(c, cudaMemcpy((void*)c->d_hit, recs.data(), recs.size() * sizeof(HitRecord), cudaMemcpyHostToDevice));
    c->sbt.hitgroupRecordBase = c->d_hit; c->sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord); c->sbt.hitgroupRecordCount = (unsigned)recs.size();
    // IAS
    std::vector<OptixInstance> oi(n);
    for (size_t i = 0; i < n; ++i) {
        OptixInstance& o = oi[i]; memset(&o, 0, sizeof(o));
        memcpy(o.transform, c->instances[i].xform, 12 * sizeof(float));
        o.instanceId = (unsigned)i; o.visibilityMask = c->instances[i].mask; o.sbtOffset = (unsigned)(2 * i); o.flags = OPTIX_INSTANCE_FLAG_NONE;
        o.traversableHandle = c->meshes[c->instances[i].mesh_id].gas;
    }
    if (c->d_instances) cudaFree((void*)c->d_instances);
    CUDA_CHECK(c, cudaMalloc((void**)&c->d_instances, n * sizeof(OptixInstance)));
    CUDA_CHECK(c, cudaMemcpy((void*)c->d_instances, oi.data(), n * sizeof(OptixInstance), cudaMemcpyHostToDevice));
    OptixBuildInput bi = {}; bi.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES; bi.instanceArray.instances = c->d_instances; bi.instanceArray.numInstances = (unsigned)n;
    OptixAccelBuildOptions ao = {}; ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sz; OPTIX_CHECK(c, optixAccelComputeMemoryUsage(c->ctx, &ao, &bi, 1, &sz));
    if (sz.tempSizeInBytes > c->ias_tmp_size) { if (c->d_ias_tmp) cudaFree((void*)c->d_ias_tmp); CUDA_CHECK(c, cudaMalloc((void**)&c->d_ias_tmp, sz.tempSizeInBytes)); c->ias_tmp_size = sz.tempSizeInBytes; }
    if (sz.outputSizeInBytes > c->ias_size) { if (c->d_ias) cudaFree((void*)c->d_ias); CUDA_CHECK(c, cudaMalloc((void**)&c->d_ias, sz.outputSizeInBytes)); c->ias_size = sz.outputSizeInBytes; }
    OPTIX_CHECK(c, optixAccelBuild(c->ctx, 0, &ao, &bi, 1, c->d_ias_tmp, c->ias_tmp_size, c->d_ias, c->ias_size, &c->ias, nullptr, 0));
    CUDA_CHECK(c, cudaDeviceSynchronize());
    c->dirty = false;
    return 0;
}

static int setup_denoiser(YtContext* c, uint32_t w, uint32_t h) {
    if (c->denoiser && c->dn_w == w && c->dn_h == h) return 0;
    if (c->denoiser) { optixDenoiserDestroy(c->denoiser); c->denoiser = nullptr; }
    OptixDenoiserOptions o = {}; o.guideAlbedo = 1; o.guideNormal = 1; o.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
    OPTIX_CHECK(c, optixDenoiserCreate(c->ctx, OPTIX_DENOISER_MODEL_KIND_HDR, &o, &c->denoiser));
    OptixDenoiserSizes sz; OPTIX_CHECK(c, optixDenoiserComputeMemoryResources(c->denoiser, w, h, &sz));
    c->dn_state_size = sz.stateSizeInBytes; c->dn_scratch_size = sz.withoutOverlapScratchSizeInBytes;
    cudaFree((void*)c->d_dn_state); cudaFree((void*)c->d_dn_scratch); cudaFree((void*)c->d_dn_in); cudaFree((void*)c->d_dn_out); cudaFree((void*)c->d_dn_alb); cudaFree((void*)c->d_dn_nrm); cudaFree((void*)c->d_dn_intensity);
    CUDA_CHECK(c, cudaMalloc((void**)&c->d_dn_state, c->dn_state_size)); CUDA_CHECK(c, cudaMalloc((void**)&c->d_dn_scratch, c->dn_scratch_size));
    size_t np = (size_t)w * h;
    CUDA_CHECK(c, cudaMalloc((void**)&c->d_dn_in, np * sizeof(float4))); CUDA_CHECK(c, cudaMalloc((void**)&c->d_dn_out, np * sizeof(float4)));
    CUDA_CHECK(c, cudaMalloc((void**)&c->d_dn_alb, np * sizeof(float4))); CUDA_CHECK(c, cudaMalloc((void**)&c->d_dn_nrm, np * sizeof(float4)));
    CUDA_CHECK(c, cudaMalloc((void**)&c->d_dn_intensity, sizeof(float)));
    OPTIX_CHECK(c, optixDenoiserSetup(c->denoiser, 0, w, h, c->d_dn_state, c->dn_state_size, c->d_dn_scratch, c->dn_scratch_size));
    c->dn_w = w; c->dn_h = h;
    return 0;
}

static OptixImage2D img2d(CUdeviceptr p, uint32_t w, uint32_t h) {
    OptixImage2D i = {}; i.data = p; i.width = w; i.height = h; i.rowStrideInBytes = w * sizeof(float4); i.pixelStrideInBytes = sizeof(float4); i.format = OPTIX_PIXEL_FORMAT_FLOAT4; return i;
}

// Motion blur / multi-pose accumulation: yt_render_begin clears the accumulation buffers, yt_render_accumulate adds
// samples for the CURRENT scene state (call it several times with different instance transforms / camera), and
// yt_render_finish normalises, denoises and reads back. yt_render() = the three in a row (single pose).
extern "C" int yt_render_begin(YtContext* c, uint32_t w, uint32_t h) {
    if (!c->d_materials) { c->err = "no materials"; return -1; }
    size_t np = (size_t)w * h;
    if (np != c->accum_pixels) {
        if (c->d_accum) cudaFree((void*)c->d_accum); if (c->d_aux) cudaFree((void*)c->d_aux); if (c->d_albedo) cudaFree((void*)c->d_albedo); if (c->d_normal) cudaFree((void*)c->d_normal);
        CUDA_CHECK(c, cudaMalloc((void**)&c->d_accum, np * sizeof(float4))); CUDA_CHECK(c, cudaMalloc((void**)&c->d_aux, np * sizeof(float4)));
        CUDA_CHECK(c, cudaMalloc((void**)&c->d_albedo, np * sizeof(float4))); CUDA_CHECK(c, cudaMalloc((void**)&c->d_normal, np * sizeof(float4)));
        c->accum_pixels = np;
    }
    CUDA_CHECK(c, cudaMemset((void*)c->d_accum, 0, np * sizeof(float4)));
    CUDA_CHECK(c, cudaMemset((void*)c->d_aux, 0, np * sizeof(float4)));
    CUDA_CHECK(c, cudaMemset((void*)c->d_albedo, 0, np * sizeof(float4))); CUDA_CHECK(c, cudaMemset((void*)c->d_normal, 0, np * sizeof(float4)));
    c->accum_samples = 0;
    return 0;
}

extern "C" int yt_render_accumulate(YtContext* c, uint32_t w, uint32_t h, uint32_t spp, uint32_t seed, int max_bounces, float clampv) {
    if (c->dirty && yt_commit(c) != 0) return -1;
    if ((size_t)w * h != c->accum_pixels) { c->err = "yt_render_begin size mismatch"; return -1; }
    YtLaunchParams& p = c->params;
    p.width = w; p.height = h; p.seed = seed; p.max_bounces = max_bounces; p.clamp = clampv;
    p.accum = (float4*)c->d_accum; p.aux = (float4*)c->d_aux; p.albedo = (float4*)c->d_albedo; p.normal = (float4*)c->d_normal;
    p.materials = (YtMaterial*)c->d_materials; p.lights = (YtRectLight*)c->d_lights; p.n_lights = (int)c->n_lights;
    p.textures = (cudaTextureObject_t*)c->d_textures; p.handle = c->ias;
    const uint32_t chunk = 4;
    for (uint32_t done = 0; done < spp; done += chunk) {
        p.spp = std::min(chunk, spp - done); p.sample_offset = c->accum_samples + done;   // distinct RNG streams per pose
        CUDA_CHECK(c, cudaMemcpy((void*)c->d_params, &p, sizeof(p), cudaMemcpyHostToDevice));
        OPTIX_CHECK(c, optixLaunch(c->pipeline, 0, c->d_params, sizeof(YtLaunchParams), &c->sbt, w, h, 1));
    }
    c->accum_samples += spp;
    CUDA_CHECK(c, cudaDeviceSynchronize());
    return 0;
}

extern "C" int yt_render_finish(YtContext* c, uint32_t w, uint32_t h, int denoise, float* out_rgb, float* out_aux) {
    size_t np = (size_t)w * h;
    if (np != c->accum_pixels) { c->err = "yt_render_begin size mismatch"; return -1; }
    std::vector<float4> acc(np);
    CUDA_CHECK(c, cudaMemcpy(acc.data(), (void*)c->d_accum, np * sizeof(float4), cudaMemcpyDeviceToHost));
    if (!denoise) {
        for (size_t i = 0; i < np; ++i) { float inv = acc[i].w > 0 ? 1.0f / acc[i].w : 0.0f; out_rgb[3 * i] = acc[i].x * inv; out_rgb[3 * i + 1] = acc[i].y * inv; out_rgb[3 * i + 2] = acc[i].z * inv; }
    } else {
        if (setup_denoiser(c, w, h) != 0) return -1;
        std::vector<float4> alb(np), nrm(np);
        CUDA_CHECK(c, cudaMemcpy(alb.data(), (void*)c->d_albedo, np * sizeof(float4), cudaMemcpyDeviceToHost));
        CUDA_CHECK(c, cudaMemcpy(nrm.data(), (void*)c->d_normal, np * sizeof(float4), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < np; ++i) {
            float inv = acc[i].w > 0 ? 1.0f / acc[i].w : 0.0f;
            acc[i] = make_float4(acc[i].x * inv, acc[i].y * inv, acc[i].z * inv, 1.0f);
            alb[i] = make_float4(alb[i].x * inv, alb[i].y * inv, alb[i].z * inv, 1.0f);
            float l = sqrtf(nrm[i].x * nrm[i].x + nrm[i].y * nrm[i].y + nrm[i].z * nrm[i].z); float ninv = l > 0 ? 1.0f / l : 0.0f;
            nrm[i] = make_float4(nrm[i].x * ninv, nrm[i].y * ninv, nrm[i].z * ninv, 0.0f);
        }
        CUDA_CHECK(c, cudaMemcpy((void*)c->d_dn_in, acc.data(), np * sizeof(float4), cudaMemcpyHostToDevice));
        CUDA_CHECK(c, cudaMemcpy((void*)c->d_dn_alb, alb.data(), np * sizeof(float4), cudaMemcpyHostToDevice));
        CUDA_CHECK(c, cudaMemcpy((void*)c->d_dn_nrm, nrm.data(), np * sizeof(float4), cudaMemcpyHostToDevice));
        OptixDenoiserParams dp = {}; dp.hdrIntensity = c->d_dn_intensity; dp.blendFactor = 0.0f;
        OptixDenoiserGuideLayer guide = {}; guide.albedo = img2d(c->d_dn_alb, w, h); guide.normal = img2d(c->d_dn_nrm, w, h);
        OptixDenoiserLayer layer = {}; layer.input = img2d(c->d_dn_in, w, h); layer.output = img2d(c->d_dn_out, w, h);
        OPTIX_CHECK(c, optixDenoiserComputeIntensity(c->denoiser, 0, &layer.input, c->d_dn_intensity, c->d_dn_scratch, c->dn_scratch_size));
        OPTIX_CHECK(c, optixDenoiserInvoke(c->denoiser, 0, &dp, c->d_dn_state, c->dn_state_size, &guide, &layer, 1, 0, 0, c->d_dn_scratch, c->dn_scratch_size));
        CUDA_CHECK(c, cudaDeviceSynchronize());
        CUDA_CHECK(c, cudaMemcpy(acc.data(), (void*)c->d_dn_out, np * sizeof(float4), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < np; ++i) { out_rgb[3 * i] = acc[i].x; out_rgb[3 * i + 1] = acc[i].y; out_rgb[3 * i + 2] = acc[i].z; }
    }
    if (out_aux) CUDA_CHECK(c, cudaMemcpy(out_aux, (void*)c->d_aux, np * sizeof(float4), cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int yt_render(YtContext* c, uint32_t w, uint32_t h, uint32_t spp, uint32_t seed, int max_bounces, float clampv, int denoise, float* out_rgb, float* out_aux) {
    if (yt_render_begin(c, w, h) != 0) return -1;
    if (yt_render_accumulate(c, w, h, spp, seed, max_bounces, clampv) != 0) return -1;
    return yt_render_finish(c, w, h, denoise, out_rgb, out_aux);
}
