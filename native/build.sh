#!/bin/bash
# Build the yamtrace OptiX path tracer: device program (PTX) + host shared library -> ../build/
#
#   ./build.sh                      # PTX target picked from the GPU's compute capability (capped at 9.0)
#   PTX_ARCH=compute_80 ./build.sh  # explicit target (must not exceed the GPU: sm_80 cards need compute_80)
#   NVCC=/usr/local/cuda/bin/nvcc ./build.sh
#
# The device code is compiled to PTX, not OptiX-IR: drivers older than the CUDA toolkit reject OptiX-IR
# ("minor Version newer than tool"), while PTX is JIT-compiled by any driver. -skip-ptx-semantics-check keeps
# nvcc from running ptxas on the PTX (it would fail on the OptiX externs).
# OptiX headers come from github.com/NVIDIA/optix-dev (headers only, no SDK install); they are fetched into
# ../third_party/optix-dev on the first build and are NOT part of this repository (NVIDIA licence).
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
OPTIX_TAG=${OPTIX_TAG:-v9.1.0}
OPTIX_DIR=../third_party/optix-dev
if [ ! -f "$OPTIX_DIR/include/optix.h" ]; then
  echo "fetching OptiX headers ($OPTIX_TAG) into third_party/optix-dev"
  mkdir -p ../third_party
  git clone --depth 1 --branch "$OPTIX_TAG" https://github.com/NVIDIA/optix-dev "$OPTIX_DIR"
fi
if [ -z "$PTX_ARCH" ]; then
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -n | head -1 | tr -d ' .')
  [ -n "$cap" ] || { echo "cannot read the GPU compute capability from nvidia-smi: set PTX_ARCH=compute_XY"; exit 1; }
  [ "$cap" -gt 90 ] && cap=90
  PTX_ARCH=compute_$cap
fi
echo "nvcc: $($NVCC --version | grep -o 'release [0-9.]*'), PTX target: $PTX_ARCH"
mkdir -p ../build
$NVCC -O3 -std=c++17 -arch=$PTX_ARCH -ptx -skip-ptx-semantics-check -I "$OPTIX_DIR/include" -I . -o ../build/device.ptx device.cu
$NVCC -O2 -std=c++17 -Xcompiler -fPIC -shared -I "$OPTIX_DIR/include" -I . -o ../build/libyamtrace.so yamtrace.cpp -lcuda -ldl
echo "built ../build/device.ptx and ../build/libyamtrace.so"
