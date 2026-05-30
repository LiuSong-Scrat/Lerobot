## Paper

https://arxiv.org/abs/2506.01844

## Citation

```bibtex
@article{shukor2025smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  author={Shukor, Mustafa and Aubakirova, Dana and Capuano, Francesco and Kooijmans, Pepijn and Palma, Steven and Zouitine, Adil and Aractingi, Michel and Pascal, Caroline and Russi, Martino and Marafioti, Andres and Alibert, Simon and Cord, Matthieu and Wolf, Thomas and Cadene, Remi},
  journal={arXiv preprint arXiv:2506.01844},
  year={2025}
}
```

新增 cache_song_pointseg_samples.py (line 1)
离线生成 current point cloud + priors + pseudo labels/weights/scores
输出 manifest.json + shard_*/字段.npy
使用 mmap .npy shard，训练随机读取不会反复解压大文件
默认 float16 存储，减少空间占用
可保存少量 pseudo label PLY 预览

新增 SongPointSegCachedDataset (line 259)
读取离线 cache
返回训练需要的 observation.point_cloud 和 pointseg.* pseudo 字段

更新 train_song_pointseg.py (line 56)
新增 --sample-cache-dir
使用 cache 时跳过在线 generate_pseudo_labels / future point cloud / cdist
模型直接使用缓存的 priors

使用方式：

PYTHONPATH=src conda run -n reap python src/lerobot/scripts/cache_song_pointseg_samples.py \
  --output-dir /path/to/song_pointseg_cache \
  --current-points 8192 \
  --future-points 16384 \
  --batch-size 8 \
  --num-workers 4 \
  --storage-dtype float16 \
  --overwrite
  
然后训练：
PYTHONPATH=src conda run -n reap python src/lerobot/scripts/train_song_pointseg.py \
  --sample-cache-dir /path/to/song_pointseg_cache \
  --output-dir /path/to/train_output