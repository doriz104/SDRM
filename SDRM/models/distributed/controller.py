"""
Distributed Attention Mechanism Controller
"""
import socket
import json
import torch
import numpy as np
from typing import List, Dict, Tuple
import threading
from queue import Queue
from pathlib import Path
import sys
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import time

# Add the project root directory to the system path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.scale_dot_product_attention import ScaleDotProductAttention

class DistributedAttentionController:
    def __init__(self, 
                 raspberry_ips: List[str],
                 d_model: int,
                 n_head: int):
        """
        Initializing the Distributed Attention Controller
        """
        if not raspberry_ips:
            raise ValueError("The edge device IP list cannot be empty")
        self.raspberry_ips = raspberry_ips
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.results_queue = Queue()
        self.port = 5000
        
        # Print initialization information
        print(f"Initializing the Distributed Controller:")
        print(f"- Available Raspberry Pi nodes: {self.raspberry_ips}")
        print(f"- model dimension: {self.d_model}")
        print(f"- attention span: {self.n_head}")
        
        # Initialize the linear transformation layer
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_concat = torch.nn.Linear(d_model, d_model)
        
    def split_head(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split the tensor by the number of attention heads
        
        Args:
            tensor: [batch_size, length, d_model]
            
        Returns:
            [batch_size, n_head, length, head_dim]
        """
        batch_size, length, d_model = tensor.size()
        tensor = tensor.view(batch_size, length, self.n_head, self.head_dim)
        return tensor.transpose(1, 2)
        
    def _prepare_attention_task(self, 
                              head_id: int, 
                              q: torch.Tensor,
                              k: torch.Tensor,
                              v: torch.Tensor) -> Dict:
        """
        Preparing task data for a single attention head

        Args.
            head_id: attention head ID
            q: query tensor [batch_size, length, head_dim]
            k: key tensor [batch_size, length, head_dim]
            v: Value tensor [batch_size, length, head_dim]

        Returns.
            Dictionary with task data
        """
        return {
            'head_id': head_id,
            'task_data': {
                'query': q[..., head_id, :, :].numpy().tolist(),
                'key': k[..., head_id, :, :].numpy().tolist(),
                'value': v[..., head_id, :, :].numpy().tolist(),
                'head_dim': self.head_dim
            }
        }
        
    def _send_task(self, pi_ip: str, task: Dict):
        """Send a task to the specified Raspberry Pi"""
        print(f"Try sending a task to: {pi_ip}")  # Add Log
        if pi_ip not in self.raspberry_ips:
            print(f"Warning: {pi_ip} is not in the list of configured Raspberry Pi's")
            return
            
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect((pi_ip, self.port))
                
                # Send data
                data = json.dumps(task).encode('utf-8')
                s.sendall(len(data).to_bytes(8, byteorder='big'))  # Send data length first
                s.sendall(data)  # Send actual data
                
                # Receive results
                size_data = s.recv(8)
                size = int.from_bytes(size_data, byteorder='big')
                
                response = b""
                while len(response) < size:
                    chunk = s.recv(min(4096, size - len(response)))
                    if not chunk:
                        break
                    response += chunk
                    
                result = json.loads(response.decode('utf-8'))
                self.results_queue.put((task['head_id'], result))
                
        except Exception as e:
            print(f"Error communicating with Raspberry Pi {pi_ip}: {str(e)}")
            self.results_queue.put((task['head_id'], None))
            
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Distributed Forward Propagation

        Args.
            q: query tensor [batch_size, length, d_model]
            k: key tensor [batch_size, length, d_model]
            v: value tensor [batch_size, length, d_model]
            mask: mask tensor

        Returns.
            Attention Output [batch_size, length, d_model]
        """
        # 1. linear transformation
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # 2. splitter head
        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)
        
        # 3. Preparation of tasks for each head
        tasks = [
            self._prepare_attention_task(i, q, k, v)
            for i in range(self.n_head)
        ]
        
        # 4. Create a thread pool to distribute tasks
        threads = []
        for i, task in enumerate(tasks):
            # Make sure to use addresses from the configured IP list
            pi_ip = self.raspberry_ips[i % len(self.raspberry_ips)]
            print(f"Assign task {i} to Raspberry Pi: {pi_ip}")
            thread = threading.Thread(
                target=self._send_task,
                args=(pi_ip, task)
            )
            threads.append(thread)
            thread.start()
            
        # 5. Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # 6. Collection of results
        results = []
        while not self.results_queue.empty():
            head_id, result = self.results_queue.get()
            if result is not None:
                results.append((head_id, torch.tensor(result)))
                
        # 7. Sort by head_id
        results.sort(key=lambda x: x[0])
        
        # 8. Combined results
        batch_size = q.size(0)
        length = q.size(2)
        combined = torch.zeros(batch_size, self.n_head, length, self.head_dim)
        
        for head_id, result in results:
            combined[:, head_id] = result
            
        # 9. Convert dimensions and connect
        combined = combined.transpose(1, 2).contiguous()
        combined = combined.view(batch_size, length, self.d_model)
        
        # 10. The final linear transformation
        output = self.w_concat(combined)
        
        return output 
    
    def process_blocks(self, batch_data: Dict) -> List[torch.Tensor]:
        """Distributed Computing for Processing Image Blocks"""
        blocks = batch_data['blocks']
        total_blocks = len(blocks)
        workers = len(self.raspberry_ips)
        
        print(f"Starting distributed processing:")
        print(f"- Total blocks: {total_blocks}")
        print(f"- Number of working nodes available: {workers}")
        print(f"- Work Node List: {self.raspberry_ips}")
        
        # Division of tasks
        blocks_per_worker = total_blocks // workers
        results = []
        
        # Creating a Thread Pool
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            
            # Distribute tasks to individual Raspberry Pi
            for i, ip in enumerate(self.raspberry_ips):
                start_idx = i * blocks_per_worker
                end_idx = start_idx + blocks_per_worker if i < workers - 1 else total_blocks
                
                worker_blocks = blocks[start_idx:end_idx]
                print(f"Assign tasks to {ip}: block {start_idx} to {end_idx}")
                
                future = executor.submit(
                    self._send_blocks_to_worker,
                    ip,
                    worker_blocks
                )
                futures.append(future)
            
            # Collection of results
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30-second timeout
                    results.extend(result)
                except Exception as e:
                    print(f"Error while processing block: {str(e)}")
        
        return results
    
    def _send_blocks_to_worker(self, ip: str, blocks: List[torch.Tensor]) -> List[torch.Tensor]:
        """Sends a block of data to the specified Raspberry Pi"""
        # Verify that the IP is in the allowed list
        if ip not in self.raspberry_ips:
            raise ValueError(f"IP {ip} not in configured Raspberry Pi list")
            
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"Trying to connect to a worker node {ip} (try {retry_count + 1}/{max_retries})")
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(10)  # Setting the 10-second timeout
                    s.connect((ip, self.port))
                    
                    # Prepare data
                    data = {
                        'blocks': [b.numpy().tolist() for b in blocks],
                        'operation': 'attention'
                    }
                    
                    # Send data
                    json_data = json.dumps(data).encode('utf-8')
                    s.sendall(len(json_data).to_bytes(8, byteorder='big'))
                    s.sendall(json_data)
                    
                    # Receive results
                    size_data = s.recv(8)
                    size = int.from_bytes(size_data, byteorder='big')
                    
                    result_data = b""
                    while len(result_data) < size:
                        chunk = s.recv(min(4096, size - len(result_data)))
                        if not chunk:
                            break
                        result_data += chunk
                    
                    # parsing result
                    result = json.loads(result_data.decode('utf-8'))
                    return [torch.tensor(block) for block in result]
                    
            except Exception as e:
                retry_count += 1
                print(f"Connection to worker node {ip} failed: {str(e)}")
                if retry_count < max_retries:
                    time.sleep(2)  # Wait 2 seconds and retry
                    continue
                else:
                    raise ConnectionError(f"Unable to connect to worker node {ip}")
                    
        raise RuntimeError(f"Sending data to worker node {ip} failed, maximum retries reached")