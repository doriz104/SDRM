"""
Distributed System Configuration Files
"""
from typing import Dict, List, Tuple, Optional
import json
import os
import socket
import time
import torch

class DistributedConfig:
    def __init__(self, 
                 raspberry_ips=None,
                 port=5000,
                 num_channels=3,
                 num_classes=10,
                 image_size=224,
                 batch_size=32,
                 num_workers=4):
        """
        Distributed Configuration
        Args.
            raspberry_ips: list of Raspberry Pi IP addresses
            port: communication port
            num_channels: number of image channels
            num_classes: Number of classification categories
            image_size: image size
            batch_size: batch size
            num_workers: number of workers
        """
        self.raspberry_ips = raspberry_ips if raspberry_ips else []
        self.port = port
        self.timeout = 30
        self.max_retries = 3
        
        # Training configuration
        self.batch_size = batch_size
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 0.01
        
        # Model Configuration
        self.embed_dim = 768
        self.num_heads = 12
        self.dropout = 0.1
        self.num_layers = 12
        
        # Data Configuration
        self.image_size = image_size
        self.patch_size = 16
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        # Optimizer Configuration
        self.warmup_steps = 10000
        self.max_lr = 0.001
        self.min_lr = 1e-5
        
        # Equipment Configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Log Configuration
        self.log_interval = 10
        self.save_interval = 1000
        self.eval_interval = 100
        
        # Path Configuration
        self.save_dir = "checkpoints"
        self.log_dir = "logs"
        
        # Other configurations
        self.num_workers = num_workers
        
    def update(self, **kwargs):
        """Updating configuration parameters"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Parameters do not exist in the configuration: {k}")

    def run_worker(self, worker_ip, task_name, task_data):
        """
        Running remote worker node tasks
        Args.
            worker_ip: IP address of the worker node
            task_name: Task name
            task_data: task data
        Returns.
            Processing results
        """
        try:
            # Creating a socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)  # Use the configured timeout
            
            # Connecting to a worker node
            sock.connect((worker_ip, self.port))
            print(f"Successful connection to the worker node: {worker_ip}:{self.port}")
            
            # Preparation of mission data
            task = {
                'task_name': task_name,
                'task_data': task_data
            }
            
            # Send data
            data = json.dumps(task).encode('utf-8')
            sock.sendall(len(data).to_bytes(8, byteorder='big'))
            sock.sendall(data)
            
            # Receive response size
            size_data = sock.recv(8)
            if not size_data:
                raise ConnectionError("Failed to receive response size")
            
            size = int.from_bytes(size_data, byteorder='big')
            
            # Receive response data
            data = bytearray()
            remaining = size
            start_time = time.time()
            
            while remaining > 0 and (time.time() - start_time) < self.timeout:
                chunk = sock.recv(min(self.chunk_size, remaining))
                if not chunk:
                    raise ConnectionError("connection interruption")
                data.extend(chunk)
                remaining -= len(chunk)
                
            if remaining > 0:
                raise TimeoutError("Receive data timeout")
                
            # parse the response
            response = json.loads(data.decode('utf-8'))
            
            # Checking the response status
            if response.get('status') == 'error':
                raise Exception(response.get('error', 'unknown error'))
                
            return response
            
        except Exception as e:
            print(f"Remote processing failure: {str(e)}")
            return None
            
        finally:
            try:
                sock.close()
            except:
                pass

    def measure_latency(self, ip):
        """
        Measure network latency to a specified IP
        Args.
            ip: Destination IP address
        Returns.
            Latency in milliseconds
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            
            start_time = time.time()
            sock.connect((ip, self.load_port))
            
            # Receive load information
            size_data = sock.recv(8)
            if not size_data:
                raise ConnectionError("Receive data size failure")
                
            size = int.from_bytes(size_data, byteorder='big')
            data = sock.recv(size)
            
            latency = (time.time() - start_time) * 1000
            return latency
            
        except Exception as e:
            print(f"Failed to measure network latency to {ip: {str(e)}}")
            return float('inf')
            
        finally:
            try:
                sock.close()
            except:
                pass

    def get_best_worker(self):
        """
        Get the best working node
        Returns.
            IP of the worker node with the lowest latency
        """
        if not self.raspberry_ips:
            return None
            
        latencies = [(ip, self.measure_latency(ip)) for ip in self.raspberry_ips]
        return min(latencies, key=lambda x: x[1])[0]
    
    def process_attention(self, x):
        """
        Handling Attention
        Args.
            x: input features [B, H*W, C]
        Returns.
            Processed Features [B, H*W, C]
        """
        try:
            # Applying attention mechanisms
            x, _ = self.attention(x, x, x)
            return x
        except Exception as e:
            print(f"attention processing error: {str(e)}")
            return x
        
    def _load_raspberry_config(self) -> Dict:
        """Loading Raspberry Pi configuration information"""
        config_path = os.path.join(os.path.dirname(__file__), 'raspberry_config.json')
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            return self._create_default_config()
        except Exception as e:
            print(f"Error loading configuration file: {str(e)}")
            return self._create_default_config()
            
    def _create_default_config(self) -> Dict:
        """Creating a Default Configuration"""
        default_config = {}
        for i, ip in enumerate(self.raspberry_ips):
            default_config[f"raspberry_{i+1}"] = {
                "ip": ip,
                "port": self.port,
                "num_cores": 4,
                "memory": "1GB",
                "status": "unknown"
            }
        return default_config
        
    def get_raspberry_config(self) -> Dict:
        """Getting Raspberry Pi configuration information"""
        return self.raspberry_config
        
    def get_worker_address(self, worker_id: int) -> Optional[Tuple[str, int]]:
        """Get the address of the specified worker node"""
        if worker_id < len(self.raspberry_ips):
            return (self.raspberry_ips[worker_id], self.port)
        return None
        
    def get_all_workers(self) -> List[Tuple[str, int]]:
        """Get addresses of all worker nodes"""
        return [(ip, self.port) for ip in self.raspberry_ips]
        
    def update_raspberry_status(self, raspberry_id: str, status: str) -> None:
        """Update Raspberry Pi Status"""
        if raspberry_id in self.raspberry_config:
            self.raspberry_config[raspberry_id]["status"] = status
            
    def get_available_workers(self) -> List[Tuple[str, int]]:
        """Get all available worker nodes"""
        available_workers = []
        for config in self.raspberry_config.values():
            if config["status"] == "active":
                available_workers.append((config["ip"], config["port"]))
        return available_workers
        
    def save_config(self) -> None:
        """Save configuration to file"""
        config_path = os.path.join(os.path.dirname(__file__), 'raspberry_config.json')
        try:
            with open(config_path, 'w') as f:
                json.dump(self.raspberry_config, f, indent=4)
        except Exception as e:
            print(f"Error saving configuration file: {str(e)}")