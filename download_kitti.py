from huggingface_hub import snapshot_download

repo_id = "yujie2696/kitti_odometry_00"

print("Downloading KITTI Odometry dataset...")

snapshot_download(repo_id, local_dir="./data/kitti_odometry_00", repo_type="dataset", local_dir_use_symlinks=False)

print("Download complete.")