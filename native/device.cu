#include <optix.h>
#include <optix_device.h>
#include "shared.h"
#include "vec_math.h"

extern "C" { __constant__ YtLaunchParams params; }

// ------------------------------------------------------------------ RNG (PCG32-lite)
struct Rng { uint32_t s; };
__device__ __forceinline__ uint32_t pcg_hash(uint32_t x) { x = x * 747796405u + 2891336453u; uint32_t w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u; return (w >> 22u) ^ w; }
__device__ __forceinline__ float rnd(Rng& r) { r.s = pcg_hash(r.s); return (r.s >> 8) * (1.0f / 16777216.0f); }

// ------------------------------------------------------------------ payload plumbing
struct Prd {
    float3 p; float3 ng; float3 ns; float2 uv; float t;
    int material_id; int instance_id; int prim_id; int hit;
};
__device__ __forceinline__ void* unpack_ptr(uint32_t a, uint32_t b) { return (void*)(((uint64_t)a << 32) | (uint64_t)b); }
__device__ __forceinline__ void pack_ptr(void* p, uint32_t& a, uint32_t& b) { uint64_t v = (uint64_t)p; a = (uint32_t)(v >> 32); b = (uint32_t)(v & 0xffffffffu); }

__device__ __forceinline__ bool trace_radiance(float3 o, float3 d, float tmin, float tmax, Prd& prd) {
    uint32_t a, b; pack_ptr(&prd, a, b); prd.hit = 0;
    optixTrace(params.handle, o, d, tmin, tmax, 0.0f, OptixVisibilityMask(3), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 2, 0, a, b);
    return prd.hit != 0;
}
__device__ __forceinline__ bool occluded(float3 o, float3 d, float tmin, float tmax) {
    uint32_t vis = 0;
    optixTrace(params.handle, o, d, tmin, tmax, 0.0f, OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               1, 2, 1, vis);
    return vis == 0;
}

// ------------------------------------------------------------------ frames / sampling helpers
__device__ __forceinline__ void onb(const float3& n, float3& t, float3& b) {
    float s = copysignf(1.0f, n.z); float a = -1.0f / (s + n.z); float c = n.x * n.y * a;
    t = make_float3(1.0f + s * n.x * n.x * a, s * c, -s * n.x);
    b = make_float3(c, s + n.y * n.y * a, -n.y);
}
__device__ __forceinline__ float3 to_world(const float3& v, const float3& t, const float3& b, const float3& n) { return v.x * t + v.y * b + v.z * n; }
__device__ __forceinline__ float3 to_local(const float3& v, const float3& t, const float3& b, const float3& n) { return make_float3(dot(v, t), dot(v, b), dot(v, n)); }
__device__ __forceinline__ float3 cosine_hemisphere(float u1, float u2) {
    float r = sqrtf(u1), phi = 2.0f * M_PIf * u2; float x = r * cosf(phi), y = r * sinf(phi);
    return make_float3(x, y, sqrtf(fmaxf(0.0f, 1.0f - u1)));
}
__device__ __forceinline__ float sqr(float x) { return x * x; }
__device__ __forceinline__ float luminance(const float3& c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }
__device__ __forceinline__ float schlick_w(float u) { float m = clamp(1.0f - u, 0.0f, 1.0f); float m2 = m * m; return m2 * m2 * m; }

// ------------------------------------------------------------------ Disney principled BRDF (isotropic)
struct Surf {
    float3 base; float metallic, roughness, specular, spec_tint, clearcoat, cc_gloss, sheen; int lambert;
};
__device__ __forceinline__ float D_GGX(float NdotH, float a) { float a2 = a * a; float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH; return a2 / (M_PIf * t * t); }
__device__ __forceinline__ float G1_GGX(float NdotV, float a) { float a2 = a * a; return 2.0f * NdotV / (NdotV + sqrtf(a2 + (1.0f - a2) * NdotV * NdotV)); }
__device__ __forceinline__ float D_GTR1(float NdotH, float a) { if (a >= 1.0f) return 1.0f / M_PIf; float a2 = a * a; float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH; return (a2 - 1.0f) / (M_PIf * logf(a2) * t); }

// lobe weights for sampling
__device__ __forceinline__ void lobe_weights(const Surf& s, float& wd, float& ws, float& wc) {
    if (s.lambert) { wd = 1.0f; ws = 0.0f; wc = 0.0f; return; }
    float3 ctint = luminance(s.base) > 0.0f ? s.base / luminance(s.base) : make_float3(1.0f);
    float3 F0 = lerp(0.08f * s.specular * lerp(make_float3(1.0f), ctint, s.spec_tint), s.base, s.metallic);
    wd = (1.0f - s.metallic) * luminance(s.base);
    ws = luminance(F0) + 0.15f * (1.0f - s.metallic);   // keep a floor so rough dielectrics still sample the spec lobe
    wc = 0.25f * s.clearcoat;
    float sum = wd + ws + wc; wd /= sum; ws /= sum; wc /= sum;
}

// evaluate f(wo,wi)*|cos| and pdf in the local frame (z = shading normal)
__device__ __forceinline__ float3 bsdf_eval(const Surf& s, const float3& wo, const float3& wi, float& pdf) {
    pdf = 0.0f;
    float NdotV = wo.z, NdotL = wi.z;
    if (NdotL <= 0.0f || NdotV <= 0.0f) return make_float3(0.0f);
    if (s.lambert) { pdf = NdotL / M_PIf; return s.base * (NdotL / M_PIf); }
    float3 h = normalize(wo + wi);
    float NdotH = h.z, LdotH = dot(wi, h);
    float3 ctint = luminance(s.base) > 0.0f ? s.base / luminance(s.base) : make_float3(1.0f);
    float3 F0 = lerp(0.08f * s.specular * lerp(make_float3(1.0f), ctint, s.spec_tint), s.base, s.metallic);
    // diffuse (Burley) + sheen
    float FL = schlick_w(NdotL), FV = schlick_w(NdotV), FH = schlick_w(LdotH);
    float Fd90 = 0.5f + 2.0f * LdotH * LdotH * s.roughness;
    float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);
    float3 sheen = s.sheen * FH * ctint;
    float3 fdiff = ((1.0f - s.metallic) / M_PIf) * (s.base * Fd + sheen);
    // specular GGX
    float a = fmaxf(0.001f, s.roughness * s.roughness);
    float D = D_GGX(NdotH, a);
    float G = G1_GGX(NdotV, a) * G1_GGX(NdotL, a);
    float3 F = lerp(F0, make_float3(1.0f), FH);
    float3 fspec = F * (D * G / (4.0f * NdotV * NdotL));
    // clearcoat
    float ac = lerp(0.1f, 0.001f, s.cc_gloss);
    float Dc = D_GTR1(NdotH, ac);
    float Gc = G1_GGX(NdotV, 0.25f) * G1_GGX(NdotL, 0.25f);
    float Fc = lerp(0.04f, 1.0f, FH);
    float fcc = 0.25f * s.clearcoat * Dc * Gc * Fc / (4.0f * NdotV * NdotL);
    float wd, ws, wc; lobe_weights(s, wd, ws, wc);
    float pdf_d = NdotL / M_PIf;
    float pdf_s = G1_GGX(NdotV, a) * D / (4.0f * NdotV);          // VNDF pdf
    float pdf_c = Dc * NdotH / (4.0f * LdotH);
    pdf = wd * pdf_d + ws * pdf_s + wc * pdf_c;
    return (fdiff + fspec + make_float3(fcc)) * NdotL;
}

// Heitz 2018 VNDF sampling of GGX
__device__ __forceinline__ float3 sample_vndf(const float3& wo, float a, float u1, float u2) {
    float3 Vh = normalize(make_float3(a * wo.x, a * wo.y, wo.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0f ? make_float3(-Vh.y, Vh.x, 0.0f) / sqrtf(lensq) : make_float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);
    float r = sqrtf(u1), phi = 2.0f * M_PIf * u2;
    float t1 = r * cosf(phi), t2 = r * sinf(phi);
    float sf = 0.5f * (1.0f + Vh.z); t2 = (1.0f - sf) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + sf * t2;
    float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
    return normalize(make_float3(a * Nh.x, a * Nh.y, fmaxf(0.0f, Nh.z)));
}

__device__ __forceinline__ bool bsdf_sample(const Surf& s, const float3& wo, Rng& rng, float3& wi, float3& f_cos, float& pdf) {
    float wd, ws, wc; lobe_weights(s, wd, ws, wc);
    float u = rnd(rng), u1 = rnd(rng), u2 = rnd(rng);
    if (u < wd) {
        wi = cosine_hemisphere(u1, u2);
    } else if (u < wd + ws) {
        float a = fmaxf(0.001f, s.roughness * s.roughness);
        float3 h = sample_vndf(wo, a, u1, u2);
        wi = reflect(-wo, h);
    } else {
        float ac = lerp(0.1f, 0.001f, s.cc_gloss); float a2 = ac * ac;
        float cos2 = (1.0f - powf(a2, 1.0f - u1)) / (1.0f - a2);
        float ct = sqrtf(clamp(cos2, 0.0f, 1.0f)), st = sqrtf(1.0f - ct * ct), phi = 2.0f * M_PIf * u2;
        float3 h = make_float3(st * cosf(phi), st * sinf(phi), ct);
        wi = reflect(-wo, h);
    }
    if (wi.z <= 0.0f) return false;
    f_cos = bsdf_eval(s, wo, wi, pdf);
    return pdf > 0.0f;
}

__device__ __forceinline__ float mis_power(float a, float b) { a *= a; b *= b; return a / fmaxf(1e-20f, a + b); }

// ------------------------------------------------------------------ light helpers
__device__ __forceinline__ float light_pdf_solid_angle(const YtRectLight& L, const float3& p_surf, const float3& p_light, const float3& n_light, float3& wi, float& dist) {
    float3 d = p_light - p_surf; float d2 = dot(d, d); dist = sqrtf(d2); wi = d / dist;
    float cosl = -dot(n_light, wi);
    if (cosl <= 0.0f) return 0.0f;
    return d2 / (L.area * cosl);
}

// ------------------------------------------------------------------ raygen
extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint32_t px = idx.x, py = idx.y, pix = py * params.width + px;
    float3 sum = make_float3(0.0f), alb_sum = make_float3(0.0f), nrm_sum = make_float3(0.0f);
    for (uint32_t s = 0; s < params.spp; ++s) {
        uint32_t si = params.sample_offset + s;
        Rng rng; rng.s = pcg_hash(pix * 9781u + si * 6271u + params.seed * 15485863u + 1u);
        float jx = rnd(rng), jy = rnd(rng);
        float dx = ((px + jx) / (float)params.width) * 2.0f - 1.0f;
        float dy = 1.0f - ((py + jy) / (float)params.height) * 2.0f;
        float3 o = params.cam.eye;
        // pixel -> normalised image plane (metric, y up), then undo the lens distortion (OpenCV Brown-Conrady, iterated
        // like cv::undistortPoints) so the ray leaves through the *undistorted* direction: the rendered image shows the
        // scene as the raw sensor would, i.e. straight lines bow outward for barrel (k1 < 0).
        float xd = (dx - params.cam.cx_off) * params.cam.thx, yd = (dy + params.cam.cy_off) * params.cam.thy;
        float xu = xd, yu = yd; bool outside_circle = false;
        if (params.cam.distort) {
            // radial part: solve r_u * (1 + k1 r_u^2 + k2 r_u^4 + k3 r_u^6) = r_d by Newton (monotonic lenses converge in a
            // few steps; the plain fixed-point iteration of cv::undistortPoints oscillates for strong wide-angle barrel)
            float ycv = -yd;   // OpenCV convention: y down
            float rd = sqrtf(xd * xd + ycv * ycv), ru = rd;
            for (int it = 0; it < 10 && rd > 1e-6f; ++it) {
                float r2 = ru * ru;
                float g = ru * (1.0f + r2 * (params.cam.k1 + r2 * (params.cam.k2 + r2 * params.cam.k3))) - rd;
                float dg = 1.0f + r2 * (3.0f * params.cam.k1 + r2 * (5.0f * params.cam.k2 + r2 * 7.0f * params.cam.k3));
                if (dg <= 1e-3f || ru > 6.0f) { outside_circle = true; break; }   // non-monotonic here: beyond the lens's image circle
                ru -= g / dg;
            }
            float sc = rd > 1e-6f ? ru / rd : 1.0f;
            xu = xd * sc; float yc = ycv * sc;
            if (params.cam.p1 != 0.0f || params.cam.p2 != 0.0f) {   // tangential terms: a few fixed-point refinements from the radial solution
                for (int it = 0; it < 6; ++it) {
                    float r2 = xu * xu + yc * yc;
                    float rad = 1.0f + r2 * (params.cam.k1 + r2 * (params.cam.k2 + r2 * params.cam.k3));
                    float dxt = 2.0f * params.cam.p1 * xu * yc + params.cam.p2 * (r2 + 2.0f * xu * xu);
                    float dyt = params.cam.p1 * (r2 + 2.0f * yc * yc) + 2.0f * params.cam.p2 * xu * yc;
                    if (rad < 0.05f) { outside_circle = true; break; }
                    xu = (xd - dxt) / rad; yc = (ycv - dyt) / rad;
                }
            }
            yu = -yc;
        }
        if (outside_circle) { xu = 0.0f; yu = 0.0f; }   // ray through the centre with zero weight -> black (vignetted) pixel
        float3 d = normalize((xu / params.cam.thx) * params.cam.U + (yu / params.cam.thy) * params.cam.V + params.cam.W);
        float3 throughput = make_float3(outside_circle ? 0.0f : 1.0f), L = make_float3(0.0f);
        float last_pdf = 0.0f; bool specular_bounce = true; int n_diffuse = 0;
        float3 g_albedo = make_float3(0.0f), g_normal = make_float3(0.0f);
        for (int bounce = 0; bounce <= params.max_bounces; ++bounce) {
            Prd prd;
            bool hit = trace_radiance(o, d, 1e-4f, 1e16f, prd);
            if (bounce == 0 && si == 0) {
                params.aux[pix] = make_float4(hit ? prd.t * dot(d, normalize(params.cam.W)) : 0.0f,
                                              hit ? (float)prd.instance_id : -1.0f, hit ? (float)prd.prim_id : -1.0f, hit ? 1.0f : 0.0f);
            }
            if (!hit) break;
            const YtMaterial& m = params.materials[prd.material_id];
            if (bounce == 0) {
                if (m.light_id >= 0) g_albedo = make_float3(1.0f);
                else {
                    float3 bc = make_float3(m.base_r, m.base_g, m.base_b);
                    if (m.tex_albedo >= 0) { float2 uvt = prd.uv * m.uv_scale; float4 t = tex2D<float4>(params.textures[m.tex_albedo], uvt.x, uvt.y); bc = bc * make_float3(t.x, t.y, t.z); }
                    float gloss = (m.flags & YT_MAT_LAMBERT) ? 0.0f : (1.0f - m.roughness) * 0.5f;
                    g_albedo = lerp(bc, make_float3(0.9f), gloss);
                    if (m.flags & YT_MAT_DIELECTRIC) g_albedo = make_float3(0.85f);
                }
                float3 nn = dot(prd.ns, -d) < 0.0f ? -prd.ns : prd.ns;
                float3 cu = normalize(params.cam.U), cv = normalize(params.cam.V), cw = normalize(params.cam.W);
                g_normal = make_float3(dot(nn, cu), dot(nn, cv), -dot(nn, cw));
            }
            float3 wo = -d;
            float3 ng = prd.ng, ns = prd.ns;
            bool backface = dot(ng, wo) < 0.0f;
            // emission
            if (m.light_id >= 0) {
                if (!backface) {
                    float3 Le = make_float3(m.emit_r, m.emit_g, m.emit_b);
                    float w = 1.0f;
                    if (!specular_bounce) {
                        const YtRectLight& Lt = params.lights[m.light_id];
                        float3 nl = make_float3(Lt.nx, Lt.ny, Lt.nz);
                        float3 wi; float dist;
                        float pl = light_pdf_solid_angle(Lt, o, prd.p, nl, wi, dist) / (float)params.n_lights;
                        w = mis_power(last_pdf, pl);
                    }
                    L += throughput * Le * w;
                }
                break;  // lights don't reflect
            }
            if (m.flags & YT_MAT_DIELECTRIC) {
                // smooth dielectric: Fresnel-weighted choice between reflection and refraction, no NEE
                float3 n = ns; float eta = m.ior;
                if (dot(n, wo) < 0.0f) { n = -n; }             // shading normal toward the incoming side
                bool entering = !backface;
                float eta_i = entering ? 1.0f : m.ior, eta_t = entering ? m.ior : 1.0f;
                float cos_i = fminf(1.0f, dot(n, wo));
                float sin2_t = (eta_i / eta_t) * (eta_i / eta_t) * fmaxf(0.0f, 1.0f - cos_i * cos_i);
                float Fr = 1.0f;
                float cos_t = 0.0f;
                if (sin2_t < 1.0f) {
                    cos_t = sqrtf(1.0f - sin2_t);
                    float rs = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t);
                    float rp = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t);
                    Fr = 0.5f * (rs * rs + rp * rp);
                }
                float3 tint = make_float3(m.base_r, m.base_g, m.base_b);
                if (rnd(rng) < Fr) {
                    d = reflect(-wo, n);
                    o = prd.p + (backface ? -ng : ng) * 1e-4f;
                } else {
                    float r = eta_i / eta_t;
                    d = normalize(r * (-wo) + (r * cos_i - cos_t) * n);
                    o = prd.p + (backface ? ng : -ng) * 1e-4f;
                    throughput *= tint;                          // absorption tint per interface (1 = clear)
                }
                specular_bounce = true; last_pdf = 0.0f;
                continue;
            }
            if (backface) {
                if (m.flags & YT_MAT_TWO_SIDED) { ng = -ng; ns = -ns; }
                else { // solid object seen from inside: continue through
                    o = prd.p + d * 1e-4f; continue; }
            }
            if (dot(ns, wo) <= 0.0f) ns = ng;
            // material params (+textures)
            Surf sf; sf.base = make_float3(m.base_r, m.base_g, m.base_b);
            sf.metallic = m.metallic; sf.roughness = m.roughness; sf.specular = m.specular; sf.spec_tint = m.spec_tint;
            sf.clearcoat = m.clearcoat; sf.cc_gloss = m.clearcoat_gloss; sf.sheen = m.sheen; sf.lambert = (m.flags & YT_MAT_LAMBERT) ? 1 : 0;
            float2 uv = prd.uv * m.uv_scale;
            if (m.tex_albedo >= 0) { float4 t = tex2D<float4>(params.textures[m.tex_albedo], uv.x, uv.y); sf.base = sf.base * make_float3(t.x, t.y, t.z); }
            if (m.tex_rough >= 0) { float4 t = tex2D<float4>(params.textures[m.tex_rough], uv.x, uv.y); sf.roughness = sf.roughness * t.y; }
            float3 T, B; onb(ns, T, B);
            float3 wo_l = to_local(wo, T, B, ns);
            float3 p = prd.p + ng * 1e-4f;
            // next event estimation
            if (params.n_lights > 0) {
                int li = min((int)(rnd(rng) * params.n_lights), params.n_lights - 1);
                const YtRectLight& Lt = params.lights[li];
                float u1 = rnd(rng), u2 = rnd(rng);
                float3 pl;
                if (Lt.ltype == 1) { float rr = sqrtf(u1), ph = 2.0f * M_PIf * u2; pl = make_float3(Lt.cx, Lt.cy, Lt.cz) + (rr * cosf(ph)) * make_float3(Lt.ux, Lt.uy, Lt.uz) + (rr * sinf(ph)) * make_float3(Lt.vx, Lt.vy, Lt.vz); }
                else pl = make_float3(Lt.cx, Lt.cy, Lt.cz) + u1 * make_float3(Lt.ux, Lt.uy, Lt.uz) + u2 * make_float3(Lt.vx, Lt.vy, Lt.vz);
                float3 nl = make_float3(Lt.nx, Lt.ny, Lt.nz);
                float3 wi; float dist;
                float pdf_l = light_pdf_solid_angle(Lt, p, pl, nl, wi, dist) / (float)params.n_lights;
                if (pdf_l > 0.0f && dot(wi, ng) > 0.0f) {
                    float3 wi_l = to_local(wi, T, B, ns);
                    float pdf_b; float3 f = bsdf_eval(sf, wo_l, wi_l, pdf_b);
                    if (luminance(f) > 0.0f && !occluded(p, wi, 1e-4f, dist - 2e-4f)) {
                        float w = mis_power(pdf_l, pdf_b);
                        float3 c = throughput * f * make_float3(Lt.rad_r, Lt.rad_g, Lt.rad_b) * (w / pdf_l);
                        if (params.clamp > 0.0f && n_diffuse > 0) { float lum = luminance(c); if (lum > params.clamp) c *= params.clamp / lum; }
                        L += c;
                    }
                }
            }
            // continue path
            float3 wi_l, f; float pdf;
            if (!bsdf_sample(sf, wo_l, rng, wi_l, f, pdf)) break;
            float3 wi = to_world(wi_l, T, B, ns);
            if (dot(wi, ng) <= 0.0f) break;
            throughput *= f / pdf;
            last_pdf = pdf; specular_bounce = false; n_diffuse++;
            o = p; d = wi;
            if (bounce >= 3) { float q = fminf(0.95f, fmaxf(throughput.x, fmaxf(throughput.y, throughput.z))); if (rnd(rng) > q) break; throughput /= q; }
            if (params.clamp > 0.0f && n_diffuse > 1) { float lum = luminance(throughput); if (lum > params.clamp * 4.0f) throughput *= params.clamp * 4.0f / lum; }
        }
        sum += L; alb_sum += g_albedo; nrm_sum += g_normal;
    }
    float4 a = params.accum[pix];
    params.accum[pix] = make_float4(a.x + sum.x, a.y + sum.y, a.z + sum.z, a.w + (float)params.spp);
    float4 ga = params.albedo[pix]; params.albedo[pix] = make_float4(ga.x + alb_sum.x, ga.y + alb_sum.y, ga.z + alb_sum.z, 0.0f);
    float4 gn = params.normal[pix]; params.normal[pix] = make_float4(gn.x + nrm_sum.x, gn.y + nrm_sum.y, gn.z + nrm_sum.z, 0.0f);
}

extern "C" __global__ void __miss__radiance() {
    Prd* prd = (Prd*)unpack_ptr(optixGetPayload_0(), optixGetPayload_1());
    prd->hit = 0;
}
extern "C" __global__ void __miss__shadow() { optixSetPayload_0(1u); }

extern "C" __global__ void __closesthit__radiance() {
    const YtHitGroupData* hg = (const YtHitGroupData*)optixGetSbtDataPointer();
    Prd* prd = (Prd*)unpack_ptr(optixGetPayload_0(), optixGetPayload_1());
    const uint32_t pid = optixGetPrimitiveIndex();
    const uint3 tri = hg->indices[pid];
    const float2 bc = optixGetTriangleBarycentrics();
    const float w0 = 1.0f - bc.x - bc.y;
    const float3 v0 = hg->vertices[tri.x], v1 = hg->vertices[tri.y], v2 = hg->vertices[tri.z];
    float3 ng = normalize(cross(v1 - v0, v2 - v0));
    float3 ns = ng;
    if (hg->normals) ns = normalize(w0 * hg->normals[tri.x] + bc.x * hg->normals[tri.y] + bc.y * hg->normals[tri.z]);
    float2 uv = make_float2(0.0f);
    if (hg->uvs) uv = w0 * hg->uvs[tri.x] + bc.x * hg->uvs[tri.y] + bc.y * hg->uvs[tri.z];
    // to world
    ng = normalize(optixTransformNormalFromObjectToWorldSpace(ng));
    ns = normalize(optixTransformNormalFromObjectToWorldSpace(ns));
    const float3 ro = optixGetWorldRayOrigin(), rd = optixGetWorldRayDirection();
    const float t = optixGetRayTmax();
    prd->p = ro + t * rd; prd->t = t; prd->ng = ng; prd->ns = ns; prd->uv = uv;
    prd->material_id = hg->material_id; prd->instance_id = hg->instance_id; prd->prim_id = (int)pid; prd->hit = 1;
}
