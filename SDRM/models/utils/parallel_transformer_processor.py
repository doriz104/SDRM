"""
Local Parallel Transformer Processor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import sys
import socket
from concurrent.futures import ThreadPoolExecutor
import json
import time
import os
import platform
import subprocess
import numpy as np
import cv2
import zlib  # 添加压缩支持
import psutil
import uuid
import traceback
import logging
from torch.utils.data import DataLoader
import pickle
import struct
import threading
from torchvision import transforms

# Add the project root directory to the system path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from ..layers.multi_head_attention import MultiHeadAttention
from models.utils.svd_processor import SVDProcessor
from models.distributed.controller import DistributedAttentionController
from models.distributed.config import DistributedConfig
from models.utils.svd_attention_processor import SVDAttentionProcessor

# Get Module Logger
logger = logging.getLogger(__name__)

BUFFER_SIZE = 1024 * 1024  # 1MB

# First define the LoadBalancer class
class LoadBalancer:
    def __init__(self, processor):
        self.processor = processor
        self.worker_stats = {}
        self.last_update = {}

    def update_worker_stats(self, worker_ip, processing_time, success):
        """Update work node statistics"""
        now = time.time()
        if worker_ip not in self.worker_stats:
            self.worker_stats[worker_ip] = {
                'avg_time': processing_time,
                'success_rate': 1.0 if success else 0.0,
                'total_tasks': 1
            }
        else:
            stats = self.worker_stats[worker_ip]
            stats['total_tasks'] += 1
            stats['avg_time'] = (stats['avg_time'] * (stats['total_tasks'] - 1) +
                               processing_time) / stats['total_tasks']
            if success:
                stats['success_rate'] = (stats['success_rate'] * (stats['total_tasks'] - 1) +
                                       1.0) / stats['total_tasks']
            else:
                stats['success_rate'] = (stats['success_rate'] * (stats['total_tasks'] - 1)) / stats['total_tasks']

        self.last_update[worker_ip] = now

    def get_best_worker(self):
        """Getting the optimal work node"""
        active_workers = self.processor._get_active_workers()
        if not active_workers:
            return None

        best_worker = None
        best_score = float('-inf')

        for worker_ip in active_workers:
            if worker_ip not in self.worker_stats:
                # New work nodes
                return worker_ip

            stats = self.worker_stats[worker_ip]
            # Calculating the composite score
            score = (stats['success_rate'] / (stats['avg_time'] + 1e-6)) * (
                1.0 / (time.time() - self.last_update.get(worker_ip, 0) + 1))

            if score > best_score:
                best_score = score
                best_worker = worker_ip

        return best_worker

class ParallelTransformerProcessor(nn.Module):
    def __init__(self, config=None, block_size=(24,24)):
        super(ParallelTransformerProcessor, self).__init__()

        # Basic Attribute Initialization
        self.config = config
        self.block_size = block_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.local_processing_mode = False
        self.connections = {}  # Initializing the Connection Dictionary

        # Initializing the Logger
        self.logger = logging.getLogger(__name__)

        # Initialization Failure Count
        self.failure_counts = {ip: 0 for ip in config.raspberry_ips} if config else {}
        self.max_failures = 3

        # Initialize SVD-related properties
        self.svd_cache = {}
        self.cache_size = 1000
        self.svd_batch_size = 16
        self.enable_profiling = False
        self.cuda_events = {'svd': {}}

        # Initializing communication statistics
        self.communication_stats = {
            'total_data_size': 0,
            'total_transfer_time': 0,
            'total_blocks': 0,
            'total_raw_size': 0,
            'total_compressed_size': 0
        }

        # Initializing a thread lock
        self.stats_lock = threading.Lock()
        self.connection_lock = threading.Lock()

        # Initializing the Load Balancer
        self.load_balancer = LoadBalancer(self)

        # Add components needed for categorization
        self.embed_dim = 768
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 197, self.embed_dim))
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, config.num_classes if config else 10)

        # Creating a Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        ).to(self.device)

        # Creating a feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Conv2d(256, 768, kernel_size=1),  # 1x1 卷积调整通道数
            nn.LayerNorm([768, 1, 1]),  # 调整归一化层的维度
            nn.GELU()
        ).to(self.device)

        # Creating a classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, config.num_classes if config else 10)
        ).to(self.device)

        # Creating an SVD Processor
        self.svd_processor = SVDProcessor(energy_threshold=0.95)

        # Add batch size configuration
        self.attention_batch_size = 16  # 注意力批处理大小

        # Creating a Thread Pool
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Enable CUDA event timer
        if self.enable_profiling:
            self.cuda_events['svd']['start'] = torch.cuda.Event(enable_timing=True)
            self.cuda_events['svd']['end'] = torch.cuda.Event(enable_timing=True)

        # Add configuration for local processing
        self.local_fallback = True  # Allow local handling of fallbacks

        # Adding Multiple Attention Layers
        self.attention = nn.MultiheadAttention(
            embed_dim=3,  # Corresponds to three RGB channels
            num_heads=1,
            batch_first=True,
            device=self.device
        )

    def forward(self, x):
        """
        forward propagation function (FPF)
        Args:
            x: Input Tensor [B, C, H, W]
        Returns:
            Characteristics after treatment [B, C, H, W]
        """
        return self.process_image(x)

    def process_image(self, image, training=False):
        """
        process image
        Args:
            image: Input Image Tensor [B, C, H, W]
            training: Is it in training mode
        Returns:
            Processed image features
        """
        try:
            # Make sure the image is on the right device
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image, device=self.device)
            elif image.device != self.device:
                image = image.to(self.device)

            # Get image size
            B, C, H, W = image.shape
            self.logger.debug(f"process image shape: {image.shape}")

            # 1. Split image into blocks
            blocks = self._split_image_into_blocks(image)

            # 2. distributed processing block (computing)
            if not self.local_processing_mode and hasattr(self, 'config'):
                active_workers = self._get_active_workers()
                if active_workers:
                    processed_blocks = self._process_blocks_distributed(blocks, active_workers)
                else:
                    self.logger.info("No working nodes available, use local processing")
                    processed_blocks = self._process_blocks_locally(blocks)
            else:
                processed_blocks = self._process_blocks_locally(blocks)

            # 3. Reconstructed image
            reconstructed = self._reconstruct_image(processed_blocks, B, C, H, W)

            return reconstructed

        except Exception as e:
            self.logger.error(f"Image Processing Errors: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _preprocess_image(self, image):
        """Image Preprocessing"""
        try:
            # 1. standardization
            if image.max() > 1.0:
                image = image / 255.0

            # 2. data enhancement
            if self.training:
                if not hasattr(self, 'augmentation'):
                    self.augmentation = nn.Sequential(
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.ColorJitter(brightness=0.1, contrast=0.1)
                    )
                image = self.augmentation(image)

            # 3. standardization
            if not hasattr(self, 'normalize'):
                self.normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            image = self.normalize(image)

            return image

        except Exception as e:
            self.logger.error(f"Image preprocessing error: {str(e)}")
            return image

    def feature_extraction(self, images):
        """
        feature extraction
        Args:
            images: Input image batch
        Returns:
            Extracted Characteristics
        """
        try:
            # If it is a new feature extraction phase, reset the statistics
            if not hasattr(self, 'feature_extraction_started'):
                self.feature_extraction_started = True
                self._reset_communication_stats()
                self.logger.info("=== Starting the feature extraction phase ===")

            B = images.size(0)
            features = []
            
            # Processes all image batches
            for i in range(B):
                # Processing individual images
                image = images[i:i+1]  # Hold 4D shape [1, C, H, W].
                feature = self.process_image(image, training=False)
                if feature is not None:
                    features.append(feature)

            # If there are processing results, stack them together
            if features:
                result = torch.cat(features, dim=0)
                
                # Check if it's the start of training
                if hasattr(self, 'training_started'):
                    # Output cumulative communication cost reports and reset flags
                    if hasattr(self, 'feature_extraction_started'):
                        self._log_communication_stats()
                        delattr(self, 'feature_extraction_started')
                        delattr(self, 'training_started')  # Clearing training markers
                        self.logger.info("=== End of feature extraction phase ===")
                
                return result
                
            return None

        except Exception as e:
            self.logger.error(f"Feature extraction error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _process_image_locally(self, image, training=False):
        """
        Feature extraction error
        Args:
            image: Input Image Tensor [B, C, H, W]
            training: Is it in training mode
        Returns:
            Processed image features
        """
        try:
            B, C, H, W = image.shape

            # 1. SVD processing
            processed_channels = []
            channel_batch_size = 3
            for b in range(B):
                channels = [image[b, c] for c in range(C)]
                for i in range(0, len(channels), channel_batch_size):
                    batch = channels[i:i + channel_batch_size]
                    U, S, V = self._batch_svd_process(batch)

                    # Check for successful SVD processing
                    if not U or not S or not V:
                        self.logger.warning(f"SVD processing failed, using raw data")
                        processed_channels.extend(batch)
                        continue

                    for j, (u, s, v) in enumerate(zip(U, S, V)):
                        if i + j < len(channels):
                            try:
                                reconstructed = torch.matmul(
                                    torch.matmul(u, torch.diag(s)),
                                    v.t()
                                )
                                processed_channels.append(reconstructed)
                            except Exception as e:
                                self.logger.error(f"Error rebuilding channel {i+j}.: {str(e)}")
                                processed_channels.append(channels[i+j])

            # Reconstructing the SVD-processed image
            try:
                svd_processed = torch.stack(processed_channels, dim=1).view(B, C, H, W)
            except Exception as e:
                self.logger.error(f"Error while reconstructing SVD-processed image: {str(e)}")
                svd_processed = image

            # 2. feature extraction
            features = self.feature_extraction(svd_processed)

            # 3. Adaptation of processing to training patterns
            if training:
                # Additional processing may need to be added in training mode
                features = self._apply_training_augmentations(features)

            return features

        except Exception as e:
            self.logger.error(f"Local image processing error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _apply_training_augmentations(self, features):
        """
        Data enhancement during application training
        Args:
            features: input features [B, C, H, W]
        Returns:
            Enhanced features
        """
        if not hasattr(self, 'training_augmentations'):
            self.training_augmentations = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.BatchNorm2d(features.size(1))
            )

        return self.training_augmentations(features)

    def _send_image_to_worker(self, worker_ip, image, training=False):
        """
        Send image to work node for processing
        Args:
            worker_ip: Work node IP
            image: input image
            training: Is it in training mode
        Returns:
            Characteristics after treatment
        """
        try:
            # Getting or creating a connection
            conn = self._get_worker_connection(worker_ip)
            if not conn:
                raise ConnectionError(f"Unable to connect to a worker node {worker_ip}")

            # Prepare data
            data = {
                'action': 'process_image',
                'image': image.cpu().numpy(),
                'training': training,
                'timestamp': time.time()
            }

            # Serialize and compress data
            serialized_data = pickle.dumps(data)
            compressed_data = zlib.compress(serialized_data)

            # Send Data Size
            size_header = struct.pack('!Q', len(compressed_data))
            conn.sendall(size_header)

            # Send Data Size
            conn.sendall(compressed_data)

            # Receive result size
            result_size = struct.unpack('!Q', conn.recv(8))[0]

            # Receive result data
            result_data = b''
            while len(result_data) < result_size:
                chunk = conn.recv(min(8192, result_size - len(result_data)))
                if not chunk:
                    raise ConnectionError("connection interruption")
                result_data += chunk

            # Decompression and deserialization results
            decompressed_result = zlib.decompress(result_data)
            result = pickle.loads(decompressed_result)

            # Check Result Status
            if result['status'] == 'success':
                # Converting numpy arrays back to tensor
                processed_features = torch.from_numpy(result['features']).to(self.device)

                # Updating statistical information
                with self.stats_lock:
                    self.communication_stats['total_processed_images'] += 1
                    self.communication_stats['total_processing_time'] += (time.time() - data['timestamp'])

                return processed_features
            else:
                raise Exception(f"Work node processing failure: {result.get('error', 'unknown error')}")

        except Exception as e:
            self.logger.error(f"Send image to worker node {worker_ip} Failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

        finally:
            # Update connection last used time
            if worker_ip in self.device_connections:
                self.device_connections[worker_ip]['last_used'] = time.time()

    def _process_blocks(self, blocks):
        """process block"""
        try:
            processed_blocks = []

            for block in blocks:
                # Clearing the GPU cache
                torch.cuda.empty_cache()

                if isinstance(block, torch.Tensor):
                    block = block.contiguous()

                # Handling of individual blocks
                processed_block = self._process_single_block(block)
                if processed_block is not None:
                    processed_blocks.append(processed_block)

                # Cleaning up the memory
                del block

            return processed_blocks

        except Exception as e:
            self.logger.error(f"block processing error: {str(e)}")
            raise
        finally:
            torch.cuda.empty_cache()

    def _process_single_block(self, block):
        """Handling of individual blocks"""
        try:
            with torch.no_grad():
                # Get original shape
                if len(block.shape) == 4:  # [1, C, H, W]
                    block = block.squeeze(0)  # [C, H, W]

                C, H, W = block.shape

                # 1. subchannel processing
                processed_channels = []
                for c in range(C):
                    channel = block[c]  # [H, W]

                    # 2. Perform SVD for each channel
                    try:
                        U, S, V = torch.svd(channel)

                        # 3. Selection of singular values based on energy retention strategies
                        total_energy = torch.sum(S ** 2)
                        energy_ratio = torch.cumsum(S ** 2, dim=0) / total_energy
                        # Retains 95% of its energy
                        k = torch.where(energy_ratio >= 0.95)[0][0].item() + 1

                        self.logger.debug(f"Channel {c}: total number of singular values = {len(S)}, number of reservations = {k}, "
                                   f"Retained energy ratio = {energy_ratio[k-1]:.4f}")

                        # 4. Reconstruction using a selected number of singular values
                        reconstructed = torch.matmul(
                            torch.matmul(
                                U[:, :k],
                                torch.diag(S[:k])
                            ),
                            V[:, :k].t()
                        )

                        processed_channels.append(reconstructed)

                    except Exception as e:
                        self.logger.error(f"Channel {c} SVD processing error: {str(e)}")
                        processed_channels.append(channel)

                # 5. Combined processed channels
                processed_block = torch.stack(processed_channels, dim=0)

                # 6. Make sure the output is shaped correctly
                if processed_block.shape != block.shape:
                    self.logger.warning(f"Warning: processed shapes do not match. Expected {block.shape}, got {processed_block.shape}")
                    processed_block = block

                # 7. Normalized blocks
                processed_block = torch.clamp(processed_block, 0, 1)

                # 8. Restoring the original dimension
                processed_block = processed_block.unsqueeze(0)  # [1, C, H, W]

                return processed_block

        except Exception as e:
            self.logger.error(f"Single block processing error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return block
        finally:
            torch.cuda.empty_cache()

    def _split_image_into_blocks(self, image):
        """
        Segment images into chunks to increase processing power for large chunks
        Args:
            image: Input Image Tensor [B, C, H, W]
        Returns:
            blocks: Image Block List
        """
        try:
            B, C, H, W = image.shape
            block_h, block_w = self.block_size

            # Check if block_size is appropriate
            if block_h > H or block_w > W:
                self.logger.warning(f"Block size ({block_h}, {block_w}) Larger than image size ({H}, {W})")
                # Automatically adjusts block_size to half or less of the image size
                block_h = min(block_h, H // 2)
                block_w = min(block_w, W // 2)
                self.block_size = (block_h, block_w)
                self.logger.info(f"Automatically adjusts block_size to: {self.block_size}")

            # Calculating the required fill
            pad_h = (block_h - H % block_h) % block_h
            pad_w = (block_w - W % block_w) % block_w

            # Add Fill
            if pad_h > 0 or pad_w > 0:
                image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')

            # Update height and width
            _, _, H_pad, W_pad = image.shape

            # Calculate the number of blocks
            num_blocks_h = H_pad // block_h
            num_blocks_w = W_pad // block_w

            # Initialization Block List
            blocks = []
            
            # Using Batching to Reduce Memory Usage
            batch_size = 4  # This value can be adjusted according to the available memory
            
            for b in range(B):
                for i in range(0, num_blocks_h, batch_size):
                    for j in range(0, num_blocks_w, batch_size):
                        # Calculate the block range for the current batch
                        i_end = min(i + batch_size, num_blocks_h)
                        j_end = min(j + batch_size, num_blocks_w)
                        
                        # Extract all blocks of the current batch
                        for ii in range(i, i_end):
                            for jj in range(j, j_end):
                                h_start = ii * block_h
                                h_end = (ii + 1) * block_h
                                w_start = jj * block_w
                                w_end = (jj + 1) * block_w
                                
                                block = image[b:b+1, :, h_start:h_end, w_start:w_end]
                                
                                # Check that the block size is correct
                                if block.shape[-2:] != (block_h, block_w):
                                    self.logger.warning(f"Block size mismatch: expected {(block_h, block_w)}, actual {block.shape[-2:]}")
                                    continue
                                    
                                blocks.append(block)
                                
                        # Proactive memory cleanup
                        torch.cuda.empty_cache()

            return blocks

        except Exception as e:
            self.logger.error(f"Image chunking error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []

    def _reconstruct_image(self, processed_blocks, B, C, H, W):
        """
        Reconstruct the processed block as an image
        Args:
            processed_blocks: List of processed image blocks
            B: Batch Size
            C: Number of Channels
            H: Original Height
            W: Original Width
        Returns:
            reconstructed: Reconstructed Image [B, C, H, W]
        """
        try:
            block_h, block_w = self.block_size

            # Calculate the size after filling
            pad_h = (block_h - H % block_h) % block_h
            pad_w = (block_w - W % block_w) % block_w
            H_pad = H + pad_h
            W_pad = W + pad_w

            # Calculate the number of blocks
            num_blocks_h = H_pad // block_h
            num_blocks_w = W_pad // block_w

            # Initialize the output tensor
            reconstructed = torch.zeros((B, C, H_pad, W_pad), device=processed_blocks[0].device)

            # Reconstructed image
            block_idx = 0
            for b in range(B):
                for i in range(num_blocks_h):
                    for j in range(num_blocks_w):
                        if block_idx >= len(processed_blocks):
                            break

                        h_start = i * block_h
                        h_end = (i + 1) * block_h
                        w_start = j * block_w
                        w_end = (j + 1) * block_w

                        reconstructed[b:b+1, :, h_start:h_end, w_start:w_end] = processed_blocks[block_idx]
                        block_idx += 1

            # Remove Fill
            if pad_h > 0 or pad_w > 0:
                reconstructed = reconstructed[:, :, :H, :W]

            return reconstructed

        except Exception as e:
            self.logger.error(f"Image reconstruction error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _process_block(self, block):
        """
        Processing a single image block
        Args.
            block: input image block [H, W, C]
        Returns.
            Processed image block [H, W, C]
        """
        try:
            # Get original shape
            H, W, C = block.shape

            # 1. subchannel processing
            processed_channels = []
            for c in range(C):
                channel = block[:, :, c]  # [H, W]

                # 2. Perform SVD for each channel
                try:
                    U, S, V = torch.svd(channel)

                    # 3. Reconstruction using the first k singular values
                    k = min(10, len(S))
                    reconstructed = torch.matmul(
                        torch.matmul(
                            U[:, :k],
                            torch.diag(S[:k])
                        ),
                        V[:, :k].t()
                    )

                    processed_channels.append(reconstructed)

                except Exception as e:
                    print(f"通道 {c} SVD处理错误: {str(e)}")
                    processed_channels.append(channel)

            # 4. Combined processed channels
            processed_block = torch.stack(processed_channels, dim=2)

            # 5. Make sure the output is shaped correctly
            if processed_block.shape != block.shape:
                print(f"Warning: processed shapes do not match. Expected {block.shape}, got {processed_block.shape}")
                processed_block = block

            # 6. Normalized blocks
            processed_block = torch.clamp(processed_block, 0, 1)

            return processed_block

        except Exception as e:
            print(f"block processing error: {str(e)}")
            return block

    def _process_block_batch(self, blocks):
        """Batch processing of image blocks"""
        processed_blocks = []
        for block in blocks:
            processed = self._process_block(block)
            processed_blocks.append(processed)
        return processed_blocks

    def _get_worker_connection(self, worker_ip):
        """Get or create a persistent connection to a worker node"""
        with self.connection_lock:
            if worker_ip in self.connections and self.connections[worker_ip]['socket']:
                # Check that the connection is valid
                try:
                    self.connections[worker_ip]['socket'].sendall(b'STATUS')
                    response = self.connections[worker_ip]['socket'].recv(1024)
                    if response == b'READY':
                        return self.connections[worker_ip]['socket']
                except:
                    # Connection Disconnected, Close and Remove
                    try:
                        self.connections[worker_ip]['socket'].close()
                    except:
                        pass
                    del self.connections[worker_ip]

            # Creating a New Connection
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFFER_SIZE)

                # Setting the TCP keepalive parameter
                if hasattr(socket, 'TCP_KEEPIDLE'):
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
                if hasattr(socket, 'TCP_KEEPINTVL'):
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                if hasattr(socket, 'TCP_KEEPCNT'):
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)

                s.settimeout(30)
                s.connect((worker_ip, self.config.port))

                self.connections[worker_ip] = {
                    'socket': s,
                    'last_used': time.time()
                }
                return s
            except Exception as e:
                self.logger.error(f"创建到 {worker_ip} 的连接失败: {str(e)}")
                return None

    def _send_blocks_to_worker(self, worker_ip, blocks, max_retries=3):
        """Optimized work node data sending"""
        for retry in range(max_retries):
            try:
                start_time = time.time()

                # Getting or creating a connection
                s = self._get_worker_connection(worker_ip)
                if not s:
                    raise Exception(f"Unable to connect to a worker node {worker_ip}")

                # Prepare data
                blocks_data = []
                shapes = []
                for block in blocks:
                    block_numpy = block.cpu().numpy()
                    blocks_data.append(block_numpy.tobytes())
                    shapes.append(block_numpy.shape)

                task_data = {
                    'task_data': {
                        'blocks': blocks_data,
                        'shapes': shapes
                    }
                }

                # Serialization and compression
                serialized = pickle.dumps(task_data)
                compressed = zlib.compress(serialized)

                # Updating statistical information
                with self.stats_lock:
                    self.communication_stats['total_raw_size'] += len(serialized)
                    self.communication_stats['total_compressed_size'] += len(compressed)
                    self.communication_stats['total_blocks'] += len(blocks)

                # Send Data Size
                size_header = struct.pack('!Q', len(compressed))
                s.sendall(size_header)

                # Waiting for confirmation
                ack = s.recv(2)
                if ack != b'OK':
                    raise Exception(f"Correct acknowledgement response not received: {ack}")

                # Send data
                s.sendall(compressed)

                # Receive response size
                response_size = struct.unpack('!Q', s.recv(8))[0]

                # Send confirmation
                s.sendall(b'OK')

                # Receive response data
                response_data = b''
                while len(response_data) < response_size:
                    # Using a larger receive buffer
                    chunk = s.recv(min(BUFFER_SIZE, response_size - len(response_data)))
                    if not chunk:
                        raise Exception("connection interruption")
                    response_data += chunk

                # Update Statistics
                end_time = time.time()
                transfer_time = end_time - start_time
                with self.stats_lock:
                    self.communication_stats['total_data_size'] += len(compressed) + response_size
                    self.communication_stats['total_transfer_time'] += transfer_time

                # Decompression and Deserialization Response
                response = pickle.loads(zlib.decompress(response_data))

                # Processing Response
                if response.get('status') == 'success':
                    processed_blocks = [torch.tensor(block).to(self.device)
                                     for block in response.get('processed_blocks', [])]
                    return processed_blocks
                else:
                    raise Exception(f"processing failure: {response.get('error', 'unknown error')}")

            except Exception as e:
                self.logger.error(f"Send to {worker_ip} failed (retry {retry+1}/{max_retries}): {str(e)}")
                if retry == max_retries - 1:
                    raise

    def _check_worker_status(self, ip: str) -> bool:
        """Improved job node status checking"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)  # Increase timeout
                s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                try:
                    s.connect((ip, self.config.port))
                except (ConnectionRefusedError, socket.timeout):
                    self.logger.debug(f"Unable to connect to a worker node {ip}")
                    return False

                try:
                    s.sendall(b'STATUS')
                    response = s.recv(1024)
                    return response == b'READY'
                except socket.error:
                    self.logger.debug(f"Worker node {ip} communication failure")
                    return False

        except Exception as e:
            self.logger.debug(f"Checking worker node {ip} state failed: {str(e)}")
            return False

    def _calculate_communication_stats(self):
        """Calculation of communication statistics indicators"""
        stats = {
            'total_data_mb': self.communication_stats['total_data_size'] / (1024 * 1024),
            'total_time': self.communication_stats['total_transfer_time'],
            'avg_speed': 0,
            'avg_size_per_image': 0,
            'avg_time_per_image': 0,
            'compression_ratio': 0
        }

        # Calculate average speed
        if stats['total_time'] > 0:
            stats['avg_speed'] = stats['total_data_mb'] / stats['total_time']

        # Calculate the average value for each graph
        if self.communication_stats['total_blocks'] > 0:
            stats['avg_size_per_image'] = stats['total_data_mb'] / self.communication_stats['total_blocks']
            stats['avg_time_per_image'] = stats['total_time'] / self.communication_stats['total_blocks']

        # Calculate compression ratio
        if self.communication_stats['total_raw_size'] > 0:
            stats['compression_ratio'] = (
                self.communication_stats['total_raw_size'] -
                self.communication_stats['total_compressed_size']
            ) / self.communication_stats['total_raw_size'] * 100

        return stats

    def _log_communication_stats(self):
        """Output communication statistics report"""
        stats = self._calculate_communication_stats()

        self.logger.info("\n=== Communication Cost Report ===")
        self.logger.info(f"Total data transferred: {stats['total_data_mb']:.2f} MB")
        self.logger.info(f"Total transmission time: {stats['total_time']:.2f} s")
        self.logger.info(f"Average transmission speed: {stats['avg_speed']:.2f} MB/s")
        self.logger.info(f"Average transmission per image: {stats['avg_size_per_image']:.2f} MB/image")
        self.logger.info(f"Average transfer time per image: {stats['avg_time_per_image']:.4f} 秒/image")
        self.logger.info(f"Data compression ratio: {stats['compression_ratio']:.2f}%")
        self.logger.info(f"Number of data blocks processed: {self.communication_stats['total_blocks']}")
        self.logger.info("")

    def _distribute_blocks(self, blocks, num_workers):
        """Distribute data blocks to worker nodes"""
        if not blocks:
            return []

        try:
            if num_workers <= 0:
                self.logger.warning("Invalid number of job nodes, use single batch")
                return [blocks]

            blocks_per_worker = len(blocks) // num_workers
            extra_blocks = len(blocks) % num_workers

            distributed_blocks = []
            start_idx = 0

            for i in range(num_workers):
                num_blocks = blocks_per_worker + (1 if i < extra_blocks else 0)
                end_idx = start_idx + num_blocks
                worker_blocks = blocks[start_idx:end_idx]
                if len(worker_blocks) > 0:
                    distributed_blocks.append(worker_blocks)
                start_idx = end_idx

            self.logger.debug(f"Allocate {len(blocks)} blocks to {num_workers} worker nodes")
            return distributed_blocks if distributed_blocks else [blocks]

        except Exception as e:
            self.logger.error(f"Error allocating data block: {str(e)}")
            return [blocks]

    def _process_blocks_distributed(self, blocks, active_workers):
        """Distributed processing of data blocks"""
        if not blocks or not active_workers:
            return self._process_blocks_locally(blocks)

        try:
            num_workers = len(active_workers)

            distributed_blocks = self._distribute_blocks(blocks=blocks, num_workers=num_workers)

            results = []
            for worker_ip, worker_blocks in zip(active_workers, distributed_blocks):
                try:
                    processed_blocks = self._send_blocks_to_worker(worker_ip, worker_blocks)
                    results.extend(processed_blocks)
                    self.failure_counts[worker_ip] = 0
                except Exception as e:
                    self.logger.error(f"Worker node {worker_ip} Processing failure: {str(e)}")
                    self.failure_counts[worker_ip] += 1
                    if self.failure_counts[worker_ip] >= self.max_failures:
                        self.logger.warning(f"Worker node {worker_ip} too many consecutive failures")
                    results.extend(self._process_blocks_locally(worker_blocks))

            return results

        except Exception as e:
            self.logger.error(f"Distributed Processing Failure: {str(e)}")
            return self._process_blocks_locally(blocks)

    def train_classifier(self, train_loader=None, val_loader=None, test_loader=None, num_epochs=10, learning_rate=0.001, device=None):
        """Train the classifier"""
        try:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Output the communication cost report of the feature extraction phase before the training starts
            if hasattr(self, 'feature_extraction_started'):
                self.logger.info("=== Output communication cost report for feature extraction phase ===")
                self._log_communication_stats()
                delattr(self, 'feature_extraction_started')
                self._reset_communication_stats()

            self.logger.info(f"Start training the classifier using the device: {device}")

            if train_loader is None or val_loader is None:
                raise ValueError("Training and validation data loaders must be provided")

            # Make sure the model is on the right device
            self.to(device)

            # Save num_epochs to instance variable
            self.num_epochs = num_epochs

            # Setting the training start flag for triggering communication cost reports for the feature extraction phase
            self.training_started = True
            self.logger.info("=== Training starts and will output a communication cost report for the feature extraction phase ===")

            # Create classifiers and move to the right device
            self.classifier = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.config.num_classes)
            ).to(device)

            # Define the loss function and optimizer
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)

            train_losses = []
            val_accuracies = []
            best_val_accuracy = 0.0
            epoch_times = []

            self.logger.info("\n=== Training begins ===")
            self.logger.info(f"Total number of rounds: {num_epochs}")
            self.logger.info(f"Learning rate: {learning_rate}")
            self.logger.info(f"Training set size: {len(train_loader.dataset)}")
            self.logger.info(f"Validation Set Size: {len(val_loader.dataset)}")

            # training cycle
            for epoch in range(num_epochs):
                epoch_start_time = time.time()

                # training phase
                self.classifier.train()
                train_loss = self._train_epoch(train_loader, optimizer, epoch, device)
                train_losses.append(train_loss)

                # validation phase
                val_accuracy = self._validate_epoch(val_loader, device, epoch, num_epochs)
                val_accuracies.append(val_accuracy)

                # Calculate and record the time spent on this round
                epoch_end_time = time.time()
                epoch_time = epoch_end_time - epoch_start_time
                epoch_times.append(epoch_time)

                # Export detailed time statistics for this round
                self.logger.info(f"\n=== Time statistics of {epoch+1} rounds ===")
                self.logger.info(f"Total time for the round: {epoch_time:.2f} s")
                self.logger.info(f"Average processing time per sample: {epoch_time/len(train_loader.dataset):.4f} s")
                self.logger.info(f"Training loss: {train_loss:.4f}")
                self.logger.info(f"Verification of accuracy: {val_accuracy:.4f}")

                # Preservation of optimal models
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self.logger.info(f"New best validation accuracy: {best_val_accuracy:.4f}")


            # Output overall statistics at the end of training
            self.logger.info("\n=== Overall training statistics ===")
            self.logger.info(f"Total training time: {sum(epoch_times):.2f} s")
            self.logger.info(f"Average time per round: {sum(epoch_times)/len(epoch_times):.2f} s")
            self.logger.info(f"Minimum round time: {min(epoch_times):.2f} s")
            self.logger.info(f"Maximum round duration: {max(epoch_times):.2f} s")
            self.logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")

            # Output communication statistics after training is complete
            self.logger.info("\n=== Communication Cost Report ===")
            self.logger.info(f"Total data transferred: {self.communication_stats['total_data_size'] / (1024 * 1024):.2f} MB")
            self.logger.info(f"total transmission time: {self.communication_stats['total_transfer_time']:.2f} s")

            if self.communication_stats['total_transfer_time'] > 0:
                avg_speed = (self.communication_stats['total_data_size'] / (1024 * 1024)) / self.communication_stats['total_transfer_time']
                self.logger.info(f"Average transmission speed: {avg_speed:.2f} MB/s")

            if self.communication_stats['total_blocks'] > 0:
                avg_size_per_image = (self.communication_stats['total_data_size'] / (1024 * 1024)) / self.communication_stats['total_blocks']
                avg_time_per_image = self.communication_stats['total_transfer_time'] / self.communication_stats['total_blocks']
                self.logger.info(f"Average transmission per image: {avg_size_per_image:.2f} MB/图")
                self.logger.info(f"Average transfer time per image: {avg_time_per_image:.4f} 秒/图")

            if self.communication_stats['total_raw_size'] > 0:
                compression_ratio = (self.communication_stats['total_raw_size'] - self.communication_stats['total_compressed_size']) / self.communication_stats['total_raw_size'] * 100
                self.logger.info(f"Data compression ratio: {compression_ratio:.2f}%")

            self.logger.info(f"Number of data blocks processed: {self.communication_stats['total_blocks']}")
            self.logger.info("")

            return train_losses, val_accuracies

        except Exception as e:
            self.logger.error(f"Training process errors: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _train_epoch(self, train_loader, optimizer, epoch, device):
        """Training an epoch"""
        try:
            self.train()
            total_loss = 0
            total_cls_loss = 0
            total_l1_loss = 0
            total_valid_loss = 0
            processed_batches = 0
            self.logger.info(f"Epoch    GPU_mem   giou_loss   cls_loss   l1_loss   valid_loss  Instances     Size")

            for batch_idx, (data, target) in enumerate(train_loader):
                # GPU Memory Usage Statistics
                gpu_mem = torch.cuda.memory_allocated() / 1024**3

                # Move data to the specified device
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                # forward propagation
                features = self.process_image(data, training=True)
                if features is None:
                    continue

                # Adjustment of feature dimensions
                if len(features.shape) > 2:
                    B = features.shape[0]
                    # Use contiguous() to make sure the memory is contiguous, then use reshape instead of view
                    features = features.contiguous().reshape(B, -1)  # Spread as [B, N]
                    if features.size(1) != 768:
                        feature_adapter = nn.Linear(features.size(1), 768).to(device)
                        features = feature_adapter(features)

                # By means of a classifier
                output = self.classifier(features)  # [B, num_classes]

                # Make sure the target is the right shape
                target = target.view(-1)  # Make sure target is 1D [B]

                # Calculation of losses
                cls_loss = F.cross_entropy(output, target)
                l1_loss = torch.mean(torch.abs(features))
                loss = cls_loss + 0.01 * l1_loss
                valid_loss = F.cross_entropy(output, target)

                # backward propagation
                loss.backward()
                optimizer.step()

                # Cumulative losses
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_l1_loss += l1_loss.item()
                total_valid_loss += valid_loss.item()
                processed_batches += 1

                # Calculation of average loss
                avg_loss = total_loss / processed_batches
                avg_cls_loss = total_cls_loss / processed_batches
                avg_l1_loss = total_l1_loss / processed_batches
                avg_valid_loss = total_valid_loss / processed_batches

                # Get current batch size
                batch_size = data.size(0)

                # Output training information in image format
                if batch_idx % 10 == 0:
                    self.logger.info(
                        f"{epoch+1}/{self.num_epochs}  {gpu_mem:.3f}G  {avg_loss:.4f}  "
                        f"{avg_cls_loss:.4f}  {avg_l1_loss:.4f}  {avg_valid_loss:.4f}  "
                        f"{batch_size:3d}  {data.shape[-1]:4d}x{data.shape[-2]}"
                    )

                # Cleaning up the memory
                torch.cuda.empty_cache()

            return total_loss / processed_batches if processed_batches > 0 else float('inf')

        except Exception as e:
            self.logger.error(f"Training process errors: {str(e)}")
            self.logger.error(traceback.format_exc())
            return float('inf')

    def _validate_epoch(self, val_loader, device, epoch, num_epochs):
        """Validating a round"""
        try:
            self.eval()
            correct = 0
            total = 0

            criterion = nn.CrossEntropyLoss().to(device)
            total_loss = 0
            total_cls_loss = 0
            total_l1_loss = 0
            total_valid_loss = 0
            batch_count = 0

            validation_start_time = time.time()
            batch_times = []
            accuracy = 0.0

            with torch.no_grad():
                total_batches = len(val_loader)
                self.logger.info(f"=== Verify at round {epoch+1} ===")

                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    batch_start_time = time.time()

                    # Clearing the cache
                    torch.cuda.empty_cache()

                    # Make sure the data is on the right device
                    inputs = inputs.to(device)
                    targets = targets.to(device).view(-1)  # Make sure the targets are 1D.

                    # Getting features
                    features = self.process_image(inputs)
                    if features is None:
                        continue

                    # Adjustment of feature dimensions
                    if len(features.shape) > 2:
                        B = features.shape[0]
                        # Using contiguous() and reshape instead of view
                        features = features.contiguous().reshape(B, -1)  # Spread as [B, N]
                        if features.size(1) != 768:
                            feature_adapter = nn.Linear(features.size(1), 768).to(device)
                            features = feature_adapter(features)

                    # anticipate
                    outputs = self.classifier(features)  # [B, num_classes]

                    # Calculation of different types of losses
                    cls_loss = criterion(outputs, targets)
                    l1_loss = torch.mean(torch.abs(features))
                    loss = cls_loss + 0.01 * l1_loss
                    valid_loss = F.cross_entropy(outputs, targets)

                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                    total_loss += loss.item()
                    total_cls_loss += cls_loss.item()
                    total_l1_loss += l1_loss.item()
                    total_valid_loss += valid_loss.item()
                    batch_count += 1

                    # Record batch processing time
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    batch_times.append(batch_time)

                    # Cleaning up the memory
                    del inputs, targets, features, outputs
                    torch.cuda.empty_cache()

                # Calculation of statistical information
                validation_time = time.time() - validation_start_time

                if batch_count > 0:
                    avg_loss = total_loss / batch_count
                    avg_valid_loss = total_valid_loss / batch_count
                    accuracy = correct / total if total > 0 else 0
                    avg_batch_time = sum(batch_times) / batch_count if batch_times else 0

                    self.logger.info(f"\n=== Validation statistics for round {epoch+1} ===")
                    self.logger.info(f"Total validation time: {validation_time:.2f} s")
                    self.logger.info(f"Validation average time per batch: {avg_batch_time:.4f} s")
                    self.logger.info(f"Verification Accuracy. {accuracy:.4f}")
                    self.logger.info(f"verification loss: {avg_valid_loss:.4f}")
                else:
                    self.logger.warning("No validation batches were successfully processed")
                    accuracy = 0.0

                return accuracy

        except Exception as e:
            self.logger.error(f"Validation process error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return 0.0

    def predict(self, images):
        """
        Predicting Image Categories
        Args.
            images: input image tensor [B, C, H, W]
        Returns.
            predictions: predicted categories [B]
        """
        try:
            self.eval()
            with torch.no_grad():
                # Getting features
                features = self.process_image(images)
                B = features.shape[0]

                # Converting Characteristic Dimensions
                features = features.flatten(2)  # [B, C, H*W]
                features = features.permute(0, 2, 1)  # [B, H*W, C]

                # Ensure that the feature dimensions are correct
                if features.size(-1) != self.embed_dim:
                    features = nn.Linear(features.size(-1),
                                       self.embed_dim).to(features.device)(features)

                # Adding a CLS token
                cls_tokens = self.cls_token.expand(B, -1, -1)
                features = torch.cat((cls_tokens, features), dim=1)

                # Handling of position codes
                if features.size(1) > self.pos_embed.size(1):
                    features = features[:, :self.pos_embed.size(1), :]
                elif features.size(1) < self.pos_embed.size(1):
                    pos_embed = self.pos_embed[:, :features.size(1), :]
                    features = features + pos_embed
                else:
                    features = features + self.pos_embed

                # Applying attention and norm
                features = features.transpose(0, 1)
                features, _ = self.attention(features, features, features)
                features = features.transpose(0, 1)
                features = self.norm(features)

                # Get CLS token output and categorize it
                cls_token_final = features[:, 0]
                outputs = self.classifier(cls_token_final)

                # Access to forecast categories
                _, predictions = outputs.max(1)

                return predictions

        except Exception as e:
            self.logger.error(f"error in forecasting: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.train()  # Recovery training mode

    def _batch_blocks(self, blocks, batch_size=16):
        """Splitting blocks into batches"""
        for i in range(0, len(blocks), batch_size):
            yield blocks[i:i + batch_size]

    def _batch_reconstruct_image(self, processed_blocks, B, C, H, W):
        """Batch reconstruction of images"""
        try:
            block_h, block_w = self.block_size

            # Calculate the size after filling
            pad_h = (block_h - H % block_h) % block_h
            pad_w = (block_w - W % block_w) % block_w
            H_pad = H + pad_h
            W_pad = W + pad_w

            # Rebuilding with Batch Processing
            num_blocks_h = H_pad // block_h
            num_blocks_w = W_pad // block_w

            # Creating an output tensor
            reconstructed = torch.zeros((B, C, H_pad, W_pad),
                                     device=processed_blocks[0].device)

            # batch file
            block_batch_size = 16
            for b in range(B):
                for start_idx in range(0, len(processed_blocks), block_batch_size):
                    end_idx = min(start_idx + block_batch_size, len(processed_blocks))
                    batch_blocks = processed_blocks[start_idx:end_idx]

                    # Calculate the position of the block
                    positions = []
                    for idx in range(start_idx, end_idx):
                        i = (idx // num_blocks_w) % num_blocks_h
                        j = idx % num_blocks_w
                        h_start = i * block_h
                        h_end = (i + 1) * block_h
                        w_start = j * block_w
                        w_end = (j + 1) * block_w
                        positions.append((h_start, h_end, w_start, w_end))

                    # Batch Placement Blocks
                    for block, (h_start, h_end, w_start, w_end) in zip(batch_blocks, positions):
                        reconstructed[b:b+1, :, h_start:h_end, w_start:w_end] = block

            # Remove Fill
            if pad_h > 0 or pad_w > 0:
                reconstructed = reconstructed[:, :, :H, :W]

            return reconstructed

        except Exception as e:
            self.logger.error(f"Image reconstruction error: {str(e)}")
            raise

    def _batch_svd_process(self, batch):
        """Optimized batch SVD processing"""
        try:
            # Using Cache
            cache_key = hash(str(batch))
            if cache_key in self.svd_cache:
                return self.svd_cache[cache_key]

            # Creating CUDA Events
            if self.enable_profiling:
                self.cuda_events['svd']['start'] = torch.cuda.Event(enable_timing=True)
                self.cuda_events['svd']['end'] = torch.cuda.Event(enable_timing=True)
                self.cuda_events['svd']['start'].record()

            # batch file
            batch_tensor = torch.stack(batch)
            batch_size = len(batch)

            # Parallel SVD calculation
            results = []
            for i in range(0, batch_size, self.svd_batch_size):
                sub_batch = batch_tensor[i:i+self.svd_batch_size]
                U, S, V = torch.svd(sub_batch)
                results.append((U, S, V))

            # Updating the cache
            if len(self.svd_cache) >= self.cache_size:
                self.svd_cache.pop(next(iter(self.svd_cache)))
            self.svd_cache[cache_key] = results

            return results

        except Exception as e:
            self.logger.error(f"Batch SVD processing errors: {str(e)}")
            return None

    def _parallel_svd(self, sub_batch):
        """Parallel SVD calculation"""
        try:
            device = sub_batch.device
            U_list, S_list, V_list = [], [], []

            # Using the batch processing capabilities of torch.svd
            U, S, V = torch.svd(sub_batch.reshape(-1, sub_batch.shape[-2], sub_batch.shape[-1]))

            # Processing each result
            for i in range(len(U)):
                # energy conservation
                total_energy = torch.sum(S[i] ** 2)
                energy_ratio = torch.cumsum(S[i] ** 2, dim=0) / total_energy
                k = torch.where(energy_ratio >= 0.95)[0][0].item() + 1

                U_list.append(U[i, :, :k])
                S_list.append(S[i, :k])
                V_list.append(V[i, :, :k])

            return U_list, S_list, V_list

        except Exception as e:
            self.logger.error(f"Parallel SVD calculation error: {str(e)}")
            return [], [], []

    def _get_active_workers(self):
        """Get a list of active worker nodes"""
        active_workers = []
        if hasattr(self, 'config') and hasattr(self.config, 'raspberry_ips'):
            for ip in self.config.raspberry_ips:
                if self._check_worker_status(ip):
                    active_workers.append(ip)
        return active_workers

    def __del__(self):
        """Clear connections"""
        for ip in self.connections:
            try:
                self.connections[ip]['socket'].close()
            except:
                pass

    def process_blocks(self, blocks):
        """Processing of data blocks with local fallback support"""
        if self.local_processing_mode:
            return self._process_blocks_locally(blocks)

        try:
            active_workers = self._get_active_workers()
            if not active_workers or len(active_workers) < len(self.config.raspberry_ips) / 2:
                self.logger.warning("Insufficient available worker nodes, switch to local processing mode")
                self.local_processing_mode = True
                return self._process_blocks_locally(blocks)

            num_workers = len(active_workers)
            distributed_blocks = self._distribute_blocks(blocks=blocks, num_workers=num_workers)

            results = []
            for worker_ip, worker_blocks in zip(active_workers, distributed_blocks):
                try:
                    processed_blocks = self._send_blocks_to_worker(worker_ip, worker_blocks)
                    results.extend(processed_blocks)
                    self.failure_counts[worker_ip] = 0
                except Exception as e:
                    self.logger.error(f"Worker node {worker_ip} Processing failure: {str(e)}")
                    self.failure_counts[worker_ip] += 1
                    if self.failure_counts[worker_ip] >= self.max_failures:
                        self.logger.warning(f"Worker node {worker_ip} too many consecutive failures")
                    results.extend(self._process_blocks_locally(worker_blocks))

            return results

        except Exception as e:
            self.logger.error(f"Distributed error handling: {str(e)}")
            self.logger.info("Switch to local processing mode")
            self.local_processing_mode = True
            return self._process_blocks_locally(blocks)

    def _process_blocks_locally(self, blocks):
        """Local processing of data blocks"""
        try:
            self.logger.info("Use of local processing mode")
            processed_blocks = []

            for block in blocks:
                # Make sure the data is on the right device
                if not isinstance(block, torch.Tensor):
                    block = torch.tensor(block, device=self.device)
                elif block.device != self.device:
                    block = block.to(self.device)

                # Getting shape information
                if len(block.shape) == 4:  # [B, C, H, W]
                    B, C, H, W = block.shape
                    block = block.permute(0, 2, 3, 1)  # [B, H, W, C]
                    block = block.reshape(B, H*W, C)
                elif len(block.shape) == 3:  # [C, H, W]
                    C, H, W = block.shape
                    block = block.permute(1, 2, 0)  # [H, W, C]
                    block = block.reshape(1, H*W, C)

                # Applying local attention mechanisms
                attention_output, _ = self.attention(
                    block,
                    block,
                    block
                )

                # Restore original shape
                if len(block.shape) == 3:  # [B, H*W, C]
                    attention_output = attention_output.reshape(B, H, W, C)
                    attention_output = attention_output.permute(0, 3, 1, 2)  # [B, C, H, W]
                else:  # [H*W, C]
                    attention_output = attention_output.reshape(H, W, C)
                    attention_output = attention_output.permute(2, 0, 1)  # [C, H, W]

                processed_blocks.append(attention_output)

            return processed_blocks

        except Exception as e:
            self.logger.error(f"本地处理错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return blocks

    def _reset_communication_stats(self):
        """Reset communication statistics"""
        self.communication_stats = {
            'total_data_size': 0,
            'total_transfer_time': 0,
            'total_blocks': 0,
            'total_raw_size': 0,
            'total_compressed_size': 0
        }

def process_image(image_path, processor):
    try:
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"process image: {os.path.basename(image_path)}, size: {torch.tensor(image).shape}")
        
        # Ensure that it is called directly without passing any keyword arguments
        result = processor(torch.tensor(image).float())  # The correct way to call
        
        return result
        
    except Exception as e:
        print(f"Errors in processing: {str(e)}")
        return None
