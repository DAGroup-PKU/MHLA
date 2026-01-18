# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import os.path as osp
import traceback
from functools import lru_cache
from glob import glob
from zipfile import ZipFile

import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from diffusion.data.builder import DATASETS
from diffusion.data.datasets.utils import *
from diffusion.data.transforms import ResizeCrop, ToTensorVideo, get_closest_ratio
from diffusion.utils.logger import get_root_logger

keys_mapping = {
    "non_shift": "non_shift",
}


@DATASETS.register_module()
class SingleZipDataset(Dataset):
    def __init__(
        self,
        zip_file_path,
        transform=None,
        load_vae_feat=False,
        config=None,
        num_frames: int = None,
        data_filter_config: dict = None,
        data_key: str = None,
        cache_dir: str = "output/data_cache",
        **kwargs,
    ):
        """
        Dataset for processing a single ZIP file containing video data for VAE feature extraction.
        Args:
            zip_file_path (str): Path to the ZIP file to process
            transform: Transform to apply to the video data
            load_vae_feat (bool): Whether to load VAE features
            config: Configuration object
            num_frames (int): Number of frames to extract
            **kwargs: Additional arguments
        """
        self.logger = (
            get_root_logger() if config is None else get_root_logger(osp.join(config.work_dir, "train_log.log"))
        )

        self.zip_file_path = osp.abspath(osp.expanduser(zip_file_path))
        if not osp.exists(self.zip_file_path):
            raise ValueError(f"ZIP file does not exist: {self.zip_file_path}")

        self.transform = transform if not load_vae_feat else None
        self.load_vae_feat = load_vae_feat
        self.aspect_ratio = eval(kwargs.pop("aspect_ratio_type"))  # base aspect ratio
        self.num_frames = num_frames

        # Extract dataset name from zip file path
        zip_dir = osp.dirname(self.zip_file_path)
        self.dataset_name = osp.basename(zip_dir)
        self.dataset_name = keys_mapping.get(self.dataset_name, self.dataset_name)

        # Initialize dataset items
        self.dataset = []
        self.failed_data = {}

        self.ratio_index = {}
        self.ratio_nums = {}
        for k, v in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []
            self.ratio_nums[float(k)] = 0

        self.if_filtering = data_filter_config is not None

        if self.if_filtering:
            resample_fps = data_filter_config.get("_resample_fps", None)
            external_data_filter = data_filter_config.get("external_data_filter", {})

            if isinstance(resample_fps, dict):
                self.resample_fps = resample_fps.get("min", -1)
                external_data_filter.update({"_resample_fps": data_filter_config["_resample_fps"]})
            else:
                self.resample_fps = -1
            zip_count = len(glob(f"{zip_dir}/*.zip"))
            dir_cache_name = self.generate_cache_filename(data_key, zip_count, external_data_filter)
            dir_save_path = osp.join(cache_dir, dir_cache_name)
            if not os.path.exists(dir_save_path):
                raise ValueError(f"Cache file not found for {dir_save_path}, will generate cache file")
            else:
                self.logger.info(f"Loaded cached dataset from {dir_save_path}")
            self.valid_dataset = json.load(open(dir_save_path))
            for item in self.valid_dataset:
                if os.path.basename(item["zip_file"]) == os.path.basename(self.zip_file_path):
                    self.dataset.append(item)

            self.logger.info(f"Loaded {len(self.dataset)} videos from cache for {self.zip_file_path}")
            self.logger.info(f"Dataset name: {self.dataset_name} for cache, data_key: {data_key}")
            self.logger.info(f"Total videos: {len(self.dataset)}")

        else:
            # Load data from the single ZIP file
            self._load_zip_data()
            self.logger.info(f"SingleZipDataset loaded: {self.zip_file_path}")
            self.logger.info(f"Dataset name: {self.dataset_name}")
            self.logger.info(f"Total videos: {len(self.dataset)}")

    def _load_zip_data(self):
        """Load data from the single ZIP file."""
        try:
            with ZipFile(self.zip_file_path, "r") as z:
                for i in z.infolist():
                    if i.filename.endswith(".json"):
                        continue
                    key, ext = osp.splitext(i.filename)

                    # Only process video files
                    if ext not in [".mp4", ".npy"]:
                        continue

                    # Determine JSON file name based on dataset type
                    if "mjv" in self.zip_file_path:
                        json_name = f"{key.split('/')[0]}/meta.json"
                    else:
                        json_name = f"{key}.json"

                    # Create cache key (dataset_name/relative_path)
                    cache_key = f"{self.dataset_name}/{key}"
                    info_data = {}

                    self.dataset.append(
                        {
                            "info": info_data,
                            "cache_key": cache_key,
                            "key": key,
                            "zip_file": self.zip_file_path,
                            "ext": ext,
                            "json_name": json_name,
                            "dataset_name": self.dataset_name,
                        }
                    )
        except Exception as e:
            self.logger.error(f"Failed to load ZIP file {self.zip_file_path}: {str(e)}")
            raise

    @staticmethod
    @lru_cache(16)
    def open_zip_file(path: str):
        return ZipFile(path, "r")

    def _sample_fps(self, frame_data, source_fps, target_fps=16, frame_num=None):
        sample_ratio = source_fps / target_fps
        if frame_num is not None:
            indices = np.arange(0, frame_num, sample_ratio).astype(int)
            return len(indices) + 1
        else:
            indices = np.arange(0, len(frame_data), sample_ratio).astype(int)

            if indices[0] > 0:
                indices = np.concatenate([[0], indices])
            else:
                last_valid_idx = min(indices[-1] + int(sample_ratio), len(frame_data) - 1)
                indices = np.concatenate([indices, [last_valid_idx]])

            return frame_data[indices]

    def getdata(self, idx):
        data = self.dataset[idx]
        self.key = data["key"]

        info = data["info"]
        cache_key = data["cache_key"]

        ext = data["ext"]
        z = SingleZipDataset.open_zip_file(data["zip_file"])
        # Load JSON metadata
        with z.open(data["json_name"], "r") as f:
            info.update(json.load(f))

        # Data info
        data_info = {"cache_key": cache_key, "zip_file": data["zip_file"], "key": data["key"]}
        if "wan" in data["zip_file"] and ("height" not in info or "width" not in info):
            ori_h = info["height"] = float(info.get("height", 480))
            ori_w = info["width"] = float(info.get("width", 832))
        else:
            ori_h = info["height"] = float(info["height"])
            ori_w = info["width"] = float(info["width"])

        closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
        closest_size = tuple(map(lambda x: int(x), closest_size))
        self.closest_ratio = closest_ratio

        data_info["img_hw"] = torch.tensor([ori_h, ori_w], dtype=torch.float32)
        data_info["aspect_ratio"] = closest_ratio

        # Load media data
        with z.open(data["key"] + ext, "r") as f:
            if ext in [".jpg", ".png", ".jpeg", ".webp"]:
                frame_data = iio.imread(f)
            elif ext == ".mp4":
                frame_data = iio.imread(f, plugin="pyav")
            elif ext == ".npy":
                frame_data = np.load(f)

        # FPS sampling
        frame_data_resample = None

        fps = 16

        unimatch_ratio = 16 / fps

        if data.get("resample_fps", -1) > 0 and data.get("fps", None) is not None:
            frame_data_resample = self._sample_fps(
                frame_data,
                source_fps=int(data["fps"]),
                target_fps=int(data["resample_fps"]),
            )
            unimatch_ratio = 16 / int(data["resample_fps"])
            # print(f"zipfile: {data['zip_file']}: {self.key}, frame_data before resample: {len(frame_data)}, frame_data after resample: {len(frame_data_resample)}, source_fps: {data['fps']}, target_fps: {data['resample_fps']}")
        # print(f"zipfile: {data['zip_file']}: {self.key}, unimatch_ratio: {unimatch_ratio}")
        # TODO: need better logi here.
        if self.num_frames is not None:
            if frame_data_resample is not None and len(frame_data_resample) >= self.num_frames:
                frame_data = frame_data_resample
            elif (
                frame_data_resample is not None
                and len(frame_data_resample) < self.num_frames
                and len(frame_data) < self.num_frames
            ):
                raise ValueError(
                    f"idx: {idx}, zipfile: {data['zip_file']}: {self.key}, frame data length is less than num_frames: {len(frame_data_resample)} < {self.num_frames} and {len(frame_data)} < {self.num_frames}"
                )

        frame_data = frame_data[: self.num_frames]

        # Transform
        if self.load_vae_feat:
            # Already processed, just convert to tensor
            vframes = torch.from_numpy(frame_data)
        else:
            # Apply transform: resize and normalize
            self.transform = T.Compose(
                [
                    ToTensorVideo(),  # TCHW
                    ResizeCrop(closest_size),
                    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ]
            )
            vframes = torch.from_numpy(frame_data).clone().permute(0, 3, 1, 2)
            vframes = self.transform(vframes)

        # Add to ratio index
        if idx not in self.ratio_index[closest_ratio]:
            self.ratio_index[closest_ratio].append(idx)

        # Return only necessary data for VAE feature extraction
        return (
            vframes,  # video tensor
            "",  # empty caption (not needed for VAE extraction)
            0,  # no attention mask needed
            data_info,  # data info with cache_key
            idx,  # index
            "",  # caption type (not used)
            {"height": ori_h, "width": ori_w},  # original dimensions
            0.0,  # dummy value
        )

    def __getitem__(self, idx):
        for _ in range(100):
            try:
                return self.getdata(idx)
            except Exception as e:
                traceback_str = traceback.format_exc()
                self.logger.error(
                    f"SingleZipDataset.getdata({idx}) Error: {str(e)}, data: {self.dataset[idx]['key']}"
                    f"\n{traceback_str}"
                )
                # For single zip dataset, we can't easily fallback to another index
                # Just return the next index
                idx = (idx + 1) % len(self.dataset)

        raise RuntimeError(f"Too many bad data in ZIP file: {self.zip_file_path}")

    def __len__(self):
        return len(self.dataset)

    def get_data_info(self, idx):
        """Get data info without loading the actual video data."""
        try:
            data = self.dataset[idx]
            info = data["info"]
            key = data["key"]
            ext = data["ext"]
            dataset_name = data["dataset_name"]

            z = SingleZipDataset.open_zip_file(data["zip_file"])
            with z.open(data["json_name"], "r") as f:
                info.update(json.load(f))

            if "wan" in data["zip_file"]:
                ori_h = info["height"] = float(info.get("height", 480))
                ori_w = info["width"] = float(info.get("width", 832))
            else:
                ori_h = info["height"] = float(info.get("height"))
                ori_w = info["width"] = float(info.get("width"))
            closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)

            return {
                "height": info["height"],
                "width": info["width"],
                "key": key,
                "index": idx,
                "zip_file": data["zip_file"],
                "ext": data["ext"],
                "closest_ratio": closest_ratio,
                "dataset_name": dataset_name,
            }
        except Exception as e:
            traceback_str = traceback.format_exc()
            self.logger.error(
                f"SingleZipDataset.get_data_info() Error: {str(e)}, data: {data['zip_file']}" f"\n{traceback_str}"
            )
            return None

    def generate_cache_filename(self, dataset_name, dataset_count, external_data_filter):
        filter_parts = []
        for filter_name, filter_info in external_data_filter.items():
            clean_name = filter_name.lstrip("_")
            min_val = float(filter_info["min"])
            max_val = float(filter_info["max"])
            filter_parts.append(f"{clean_name}_{min_val}-{max_val}")

        if filter_parts:
            filter_str = "_".join(filter_parts)
            filename = f"{dataset_name}-{dataset_count}_{filter_str}_f{max(self.num_frames, 81)}_cached_dataset.json"
        else:
            filename = f"{dataset_name}-{dataset_count}_cached_dataset.json"

        return filename


