# TRELLIS (open-source image-to-3D) install recipe

Tested 2026-07-24 on the GPU server (A800, driver 580.159.03, CUDA toolkit 12.1 preinstalled,
96 cores). Purpose: open-source alternative to MeshyAI for the mesh stage
(`scripts/run_mesh.py` comparison). Result of the comparison: see `docs/PROGRESS.md` 2026-07-24.

## Server recipe (works, ~30 min)

```bash
# 1. CUDA 12.6 toolchain + dev headers (server toolkit was 12.1, torch cu126 needs 12.6;
#    cuda-nvcc-12-6 alone is NOT enough — cusparse.h etc. come from the dev packages)
sudo apt-get install -y cuda-nvcc-12-6 cuda-libraries-dev-12-6 cuda-cudart-dev-12-6

# 2. venv + deps
export PATH="$HOME/.local/bin:/usr/local/cuda-12.6/bin:$PATH"
uv venv /tmp/vm/trellis-venv --python 3.11
source /tmp/vm/trellis-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST="8.0"   # A800 = sm_80
uv pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja \
    trimesh xatlas pyvista pymeshfix igraph transformers safetensors omegaconf huggingface-hub \
    spconv-cu126 open3d rembg onnxruntime lpips pandas tensorboard setuptools wheel
uv pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# 3. nvdiffrast: MUST use --no-build-isolation (it imports torch at build time)
git clone --depth 1 https://github.com/NVlabs/nvdiffrast /tmp/nvdiffrast
uv pip install --no-build-isolation /tmp/nvdiffrast

# 4. kaolin: prebuilt wheel from NVIDIA's index (torch/cuda tag must match)
uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu126.html

# 5. TRELLIS: not a package — clone WITH submodules (flexicubes!), run from repo dir
git clone --depth 1 --recurse-submodules https://github.com/microsoft/TRELLIS /tmp/vm/TRELLIS
cd /tmp/vm/TRELLIS
python -c "from trellis.pipelines import TrellisImageTo3DPipeline; print('OK')"

# 6. run (weights ~4GB from HF — use the mirror, proxies unset)
HF_ENDPOINT=https://hf-mirror.com env -u all_proxy -u ALL_PROXY -u http_proxy -u https_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY python example.py --image /path/to/input.png --output-dir out/
```

Gotchas encountered (in order): uv not on PATH in non-interactive ssh; torch wheel
cu126 vs toolkit 12.1 mismatch (silent "Cannot compile CUDA extension" — the real error is
hidden, run `python setup.py build_ext --inplace` manually to see it); missing cusparse.h
(nvcc package has no dev headers); TRELLIS `flexicubes` is a git submodule; server github
connectivity is flaky — clone locally and rsync instead.

## Local machine recipe (RTX 5080, driver 580.159.03, NO nvcc installed)

Same as above, except:
- Install the toolkit first: `sudo apt-get install -y cuda-nvcc-12-6 cuda-libraries-dev-12-6
  cuda-cudart-dev-12-6` (the nvidia apt repo is already configured on this box; 12.6 matches
  torch cu126 wheels). No system CUDA is present by default.
- `TORCH_CUDA_ARCH_LIST="12.0"` for the RTX 5080 (sm_120, Blackwell). If sm_120 is too new
  for a given extension, fall back to `TORCH_CUDA_ARCH_LIST="9.0+PTX"` (PTX JIT via driver 580).
- venv location: keep it in the project: `uv venv .venv-trellis --python 3.11`.
- Alternative without any local toolkit: build wheels on the GPU server (recipe above) and
  copy them over (`uv pip install --no-build-isolation` produces importable builds under the
  server venv's site-packages; copy `nvdiffrast*`, `kaolin*` and build artifacts). Works
  because both are py3.11 + torch 2.7.0; still needs `spconv-cu126` + the pip deps locally.
