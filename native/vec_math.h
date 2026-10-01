// Minimal float2/float3/float4 helpers for CUDA device code.
#pragma once
#include <cuda_runtime.h>
#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif
__host__ __device__ __forceinline__ float3 make_float3(float s) { return make_float3(s, s, s); }
__host__ __device__ __forceinline__ float2 make_float2(float s) { return make_float2(s, s); }
__host__ __device__ __forceinline__ float3 operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ __forceinline__ float3 operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ __forceinline__ float3 operator-(const float3& a) { return make_float3(-a.x, -a.y, -a.z); }
__host__ __device__ __forceinline__ float3 operator*(const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__host__ __device__ __forceinline__ float3 operator*(const float3& a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__host__ __device__ __forceinline__ float3 operator*(float s, const float3& a) { return make_float3(a.x * s, a.y * s, a.z * s); }
__host__ __device__ __forceinline__ float3 operator/(const float3& a, float s) { float inv = 1.0f / s; return a * inv; }
__host__ __device__ __forceinline__ float3 operator/(const float3& a, const float3& b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
__host__ __device__ __forceinline__ void operator+=(float3& a, const float3& b) { a.x += b.x; a.y += b.y; a.z += b.z; }
__host__ __device__ __forceinline__ void operator*=(float3& a, const float3& b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; }
__host__ __device__ __forceinline__ void operator*=(float3& a, float s) { a.x *= s; a.y *= s; a.z *= s; }
__host__ __device__ __forceinline__ void operator/=(float3& a, float s) { float inv = 1.0f / s; a.x *= inv; a.y *= inv; a.z *= inv; }
__host__ __device__ __forceinline__ float2 operator+(const float2& a, const float2& b) { return make_float2(a.x + b.x, a.y + b.y); }
__host__ __device__ __forceinline__ float2 operator*(const float2& a, float s) { return make_float2(a.x * s, a.y * s); }
__host__ __device__ __forceinline__ float2 operator*(float s, const float2& a) { return make_float2(a.x * s, a.y * s); }
__host__ __device__ __forceinline__ float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ __forceinline__ float3 cross(const float3& a, const float3& b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__host__ __device__ __forceinline__ float length(const float3& a) { return sqrtf(dot(a, a)); }
__host__ __device__ __forceinline__ float3 normalize(const float3& a) { return a / length(a); }
__host__ __device__ __forceinline__ float clamp(float x, float a, float b) { return fminf(b, fmaxf(a, x)); }
__host__ __device__ __forceinline__ float lerp(float a, float b, float t) { return a + t * (b - a); }
__host__ __device__ __forceinline__ float3 lerp(const float3& a, const float3& b, float t) { return a + t * (b - a); }
__host__ __device__ __forceinline__ float3 reflect(const float3& i, const float3& n) { return i - 2.0f * dot(n, i) * n; }
