# Full-Molmo2 environment

This checkout uses a fresh, project-specific Conda prefix at
`.venv-smol5090`. It is not a copied virtual environment and does not contain
the broken absolute paths from the former `/raid5/...` prefix.

## Recreate the environment

From the repository root, run:

```bash
benchmarks/song_real_libero/scripts/setup_molmo2_environment.sh
conda activate "$PWD/.venv-smol5090"
```

The setup script refuses to overwrite an existing prefix. Pass a different
absolute path as its first argument when a side-by-side rebuild is desired.

The dependency records have separate roles:

- `environment-molmo2-conda.lock.txt` is the SHA-256-pinned Linux x86_64
  Conda lock for Python 3.11 and the CUDA 12.8 build toolchain.
- `requirements-molmo2-lock.txt` is the complete pinned pip runtime lock.
- `requirements-molmo2-direct.txt` documents the packages selected directly
  for the current Full-Molmo2, LitePT, test, and LIBERO code paths.
- `environment-molmo2.yml` is a readable minimal environment specification;
  the setup script uses the stricter explicit lock.

`flash-attn==2.8.3` and `torch-scatter==2.1.2` are intentionally excluded from
the pip lock and built from source by the script. Their public CUDA wheels need
`GLIBC_2.32`, which is newer than the host glibc. The local `pointops` package
is also compiled from this checkout. These three builds target both `sm_80`
(the A800 cards detected on this host) and `sm_120` (RTX 5090).

## LIBERO and GPU selection

Use the checked machine-local LIBERO configuration without triggering the
package's interactive first-import prompt:

```bash
export LIBERO_CONFIG_PATH="$PWD/benchmarks/song_real_libero/data/libero_setting/libero_config"
```

Training launchers reserve physical GPU 0 and expose only cards 1-7. Keep the
same rule for ad-hoc checks, for example:

```bash
CUDA_VISIBLE_DEVICES=1 python -c 'import torch; print(torch.cuda.get_device_name(0))'
```

PyTorch can use all eight GPUs, but this host currently reports an NVML
driver/library mismatch to `nvidia-smi`. Scripts that use `nvidia-smi` as a
memory gate will need the host NVIDIA driver/NVML installation repaired even
though CUDA kernels themselves run successfully.

## Validation recorded during rebuild

- `pip check`: no broken requirements.
- CUDA runtime: PyTorch `2.8.0+cu128`, CUDA `12.8`.
- Physical GPU 1: `NVIDIA A800 80GB PCIe`, compute capability `8.0`.
- CUDA smoke tests: `flash-attn`, `torch-scatter`, and local `pointops` passed.
- Focused repository tests: 33 passed, 2 skipped.
