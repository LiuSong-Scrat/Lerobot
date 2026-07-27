# SCAI SERVER

## Libero Benchmark
## 1.准备WEP-VLA Lerobot格式Benchmark数据集
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --demo-root /opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/libero_demos \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
  --suite libero_10 \
  --all-tasks \
  --episodes 50 \
  --num-workers 10 \
  --num-points 10000 \
  --point-cloud-storage zarr \
  --fps 20 \
  --replay-mode states \
  --state-observation-offset 1 \
  --restore-demo-model \
  --require-source-fps-match \
  --save-rgb-images \
  --image-camera agentview \
  --no-download-demos \
  --save-video \
  --vis-count 2 \
  --overwrite \
  --output-root /opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v041_dataset_fixed/libero_4suite_lerobot_dataset \
  --repo-id song_libero_4suite_pointcloud


# SXL FRANKA
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --demo-root /opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/libero_demos \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
  --suite libero_10 \
  --all-tasks \
  --episodes 50 \
  --num-workers 10 \
  --num-points 10000 \
  --point-cloud-storage zarr \
  --fps 20 \
  --replay-mode states \
  --state-observation-offset 1 \
  --restore-demo-model \
  --require-source-fps-match \
  --save-rgb-images \
  --image-camera agentview \
  --no-download-demos \
  --save-video \
  --vis-count 2 \
  --overwrite \
  --output-root /opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v041_dataset_fixed/libero_4suite_lerobot_dataset \
  --repo-id song_libero_4suite_pointcloud
