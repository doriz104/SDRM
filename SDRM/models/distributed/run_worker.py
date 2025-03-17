"""
Edge device worker node startup script
"""
import os
import sys
import logging
import socket
import json
import numpy as np
import psutil
import time
from threading import Thread
import zlib
import threading
import traceback
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import struct
import pickle

# Setup Log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a special temporary directory in the user directory
home_dir = os.path.expanduser('~')
worker_temp_dir = os.path.join(home_dir, '.worker_temp')

try:
    # Create a catalog
    os.makedirs(worker_temp_dir, mode=0o700, exist_ok=True)
    
    # Setting environment variables
    os.environ['TMPDIR'] = worker_temp_dir
    os.environ['TEMP'] = worker_temp_dir
    os.environ['TMP'] = worker_temp_dir
    
    # Verification Catalog
    test_file = os.path.join(worker_temp_dir, 'test_write')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    
    print(f"Successfully created and verified temporary catalog: {worker_temp_dir}")
    
except Exception as e:
    print(f"Failed to create temporary directory: {str(e)}")
    sys.exit(1)

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(models_dir)
layers_dir = os.path.join(models_dir, 'layers')
utils_dir = os.path.join(models_dir, 'utils')

# Add all relevant directories to Python paths
sys.path.insert(0, project_root)
sys.path.insert(0, models_dir)
sys.path.insert(0, layers_dir)
sys.path.insert(0, utils_dir)
sys.path.insert(0, current_dir)

# Verify Path
print("Python Path:")
for path in sys.path:
    print(f"- {path}")

try:
    from layers.scale_dot_product_attention import ScaleDotProductAttention
    from layers.multi_head_attention import MultiHeadAttention
    print("Successful introduction of the Attention module")
except Exception as e:
    print(f"import error: {str(e)}")
    print(f"Current Directory Structure:")
    for root, dirs, files in os.walk(project_root):
        print(f"\n{root}")
        for d in dirs:
            print(f"  dir: {d}")
        for f in files:
            print(f"  file: {f}")
    sys.exit(1)

# Increase buffer size to 1MB
BUFFER_SIZE = 1024 * 1024

class LoadMonitor:
    def __init__(self):
        self.cpu_usage = 0
        self.memory_usage = 0
        self.last_update = 0
        self._running = True
        
    def start_monitoring(self):
        """Start the monitor thread"""
        Thread(target=self._monitor_loop, daemon=True).start()
        
    def _monitor_loop(self):
        """control loop"""
        while self._running:
            self.cpu_usage = psutil.cpu_percent(interval=1)
            self.memory_usage = psutil.virtual_memory().percent
            self.last_update = time.time()
            time.sleep(2)
            
    def get_load_info(self):
        """Getting load information"""
        return {
            'cpu': self.cpu_usage,
            'memory': self.memory_usage,
            'timestamp': self.last_update
        }

class Worker:
    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Setting the socket buffer
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFFER_SIZE)
        
        # Modify the embedding dimension to match the input
        self.embed_dim = 3
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=1,
            batch_first=True,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
    def process_task(self, task_data):
        """Processing of mission data"""
        try:
            # Ensure that the data structure is correct
            if 'task_data' not in task_data:
                raise KeyError("Missing 'task_data' field")
            
            task_content = task_data['task_data']
            if 'blocks' not in task_content or 'shapes' not in task_content:
                raise KeyError("Missing 'blocks' or 'shapes' field")
            
            blocks_data = task_content['blocks']
            shapes = task_content['shapes']
            self.logger.info(f"Start processing {len(blocks_data)} blocks.")
            
            # Convert to tensor and move to edge device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            processed_blocks = []
            
            for block_idx, (block_bytes, shape) in enumerate(zip(blocks_data, shapes)):
                try:
                    # Converting from bytes to numpy arrays
                    block_array = np.frombuffer(block_bytes, dtype=np.float32)
                    block_array = block_array.reshape(shape)
                    
                    # Convert to tensor
                    tensor_block = torch.from_numpy(block_array).to(device)
                    
                    # Printing the original shape
                    self.logger.info(f"Data block {block_idx} Shape: {tensor_block.shape}")
                    
                    # Adjustment dimension [B, C, H, W] -> [H*W, B, C]
                    if len(tensor_block.shape) == 4:  # [B, C, H, W]
                        B, C, H, W = tensor_block.shape
                        tensor_block = tensor_block.permute(0, 2, 3, 1)  # [B, H, W, C]
                        tensor_block = tensor_block.reshape(B, H*W, C)  # [B, H*W, C]
                        
                    elif len(tensor_block.shape) == 3:  # [C, H, W]
                        C, H, W = tensor_block.shape
                        tensor_block = tensor_block.permute(1, 2, 0)  # [H, W, C]
                        tensor_block = tensor_block.reshape(1, H*W, C)  # [1, H*W, C]
                    
                    # Applying Multiple Attention
                    attention_output, _ = self.attention(
                        tensor_block,
                        tensor_block,
                        tensor_block
                    )
                    
                    # Restore original shape
                    if len(shape) == 4:  # If the original is 4D
                        attention_output = attention_output.reshape(B, H, W, C)
                        attention_output = attention_output.permute(0, 3, 1, 2)
                    else:  # If the original is 3D
                        attention_output = attention_output.reshape(H, W, C)
                        attention_output = attention_output.permute(2, 0, 1)
                    
                    # Transfer back to CPU and convert to list
                    processed_block = attention_output.cpu().detach().numpy().tolist()
                    processed_blocks.append(processed_block)
                    
                    self.logger.info(f"Successful processing of data blocks {block_idx + 1}/{len(blocks_data)}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing block {block_idx}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    # Returns the original data block
                    processed_blocks.append(block_array.tolist())
                
                # Cleaning up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return {
                'status': 'success',
                'processed_blocks': processed_blocks
            }
            
        except Exception as e:
            self.logger.error(f"Handling task errors: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'error': str(e)
            }
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def start(self):
        """Initiation of work nodes"""
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.running = True
            self.logger.info(f"The worker node starts at {self.host}:{self.port}")
            
            while self.running:
                try:
                    client, addr = self.socket.accept()
                    self.logger.info(f"Accepts connections from {addr}")
                    client_thread = Thread(target=self.handle_client, args=(client, addr))
                    client_thread.start()
                except Exception as e:
                    self.logger.error(f"Handling connection errors: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"activation error: {str(e)}")
            raise
            
    def handle_client(self, client_socket, addr):
        """Handling client connections"""
        self.logger.info(f"Accepts connections from {addr}")
        client_socket.settimeout(30)  # Setting the 30-second timeout
        
        # Setting the client socket buffer
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFFER_SIZE)
        
        try:
            while True:  # Continuous processing of requests from the same client
                # Receive command or data size
                initial_data = client_socket.recv(8)
                if not initial_data:
                    self.logger.info("Client closes the connection")
                    break
                
                # Check if it is a status check request
                if initial_data == b'STATUS':
                    self.logger.info("Status check request received")
                    client_socket.sendall(b'READY')
                    continue
                
                try:
                    # Parsing Data Size
                    data_size = struct.unpack('!Q', initial_data)[0]
                    self.logger.info(f"Expected received data size: {data_size} bytes")
                    
                    # Send confirmation
                    client_socket.sendall(b'OK')
                    
                    # receive data
                    received_data = b''
                    remaining = data_size
                    
                    while remaining > 0:
                        # Using a larger receive buffer
                        chunk = client_socket.recv(min(BUFFER_SIZE, remaining))
                        if not chunk:
                            raise Exception("connection interruption")
                        received_data += chunk
                        remaining -= len(chunk)
                        self.logger.info(f"Accepted: {len(received_data)}/{data_size} bytes")
                    
                    # Processing data
                    try:
                        # Unzip the data
                        decompressed_data = zlib.decompress(received_data)
                        self.logger.info("Data unzipped successfully")
                        
                        # deserialization
                        task_data = pickle.loads(decompressed_data)
                        self.logger.info("Data Deserialization Successful")
                        
                        # Processing tasks
                        result = self.process_task(task_data)
                        
                        # Serialized Response
                        response_data = pickle.dumps(result)
                        compressed_response = zlib.compress(response_data)
                        response_size = len(compressed_response)
                        
                        # Send Response Size
                        size_header = struct.pack('!Q', response_size)
                        client_socket.sendall(size_header)
                        
                        # Waiting for confirmation
                        ack = client_socket.recv(2)
                        if ack != b'OK':
                            raise Exception(f"Unexpected confirmation response received: {ack}")
                        
                        # Send response data
                        total_sent = 0
                        while total_sent < response_size:
                            sent = client_socket.send(compressed_response[total_sent:total_sent + 8192])
                            if sent == 0:
                                raise Exception("connection interruption")
                            total_sent += sent
                        
                        self.logger.info("Response data sending complete")
                        
                    except Exception as e:
                        self.logger.error(f"Error while processing data: {str(e)}")
                        self._send_error_response(client_socket, {
                            'status': 'error',
                            'error': str(e)
                        })
                        
                except struct.error as e:
                    self.logger.error(f"Parsing data size error: {str(e)}")
                    break
                
                except Exception as e:
                    self.logger.error(f"Handling request errors: {str(e)}")
                    break
                
        except Exception as e:
            self.logger.error(f"Handling client-side errors: {str(e)}")
            self.logger.error(traceback.format_exc())
            
        finally:
            try:
                client_socket.close()
            except:
                pass
            self.logger.info(f"Close the connection to {addr}")

    def _send_error_response(self, client_socket, error_data):
        """Send error response"""
        try:
            error_bytes = pickle.dumps(error_data)
            compressed_error = zlib.compress(error_bytes)
            size = len(compressed_error)
            
            # Send Size
            size_header = struct.pack('!Q', size)
            client_socket.sendall(size_header)
            
            # Waiting for confirmation
            try:
                ack = client_socket.recv(2)
                if ack == b'OK':
                    # Send error data
                    total_sent = 0
                    while total_sent < size:
                        sent = client_socket.send(compressed_error[total_sent:total_sent + 8192])
                        if sent == 0:
                            raise Exception("connection interruption")
                        total_sent += sent
            except Exception as e:
                self.logger.error(f"Error while sending error response: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Preparation error response failed: {str(e)}")

    def process_blocks(self, blocks: List) -> List:
        """process data block"""
        try:
            processed_blocks = []
            for block in blocks:
                # Convert to tensor
                tensor_block = torch.tensor(block)
                processed_block = self.transformer_process(tensor_block)
                # Convert back to list
                processed_blocks.append(processed_block.tolist())
            return processed_blocks
        except Exception as e:
            print(f"Error while processing block: {e}")
            return blocks
            
    def transformer_process(self, block: torch.Tensor) -> torch.Tensor:
        """Processing blocks with the transformer"""
        try:
            # Get the shape of the block
            H, W, C = block.shape
            # Spreading as a Sequence
            flat_block = block.reshape(-1, C)
            processed_block = flat_block + 0.1
            return processed_block.reshape(H, W, C)
        except Exception as e:
            print(f"Transformer processing error: {e}")
            return block
            
    def stop(self):
        """Stopping the server"""
        self.running = False
        try:
            self.socket.close()
        except:
            pass

def main():
    try:
        # Setting the log level
        logging.getLogger().setLevel(logging.INFO)
        worker = Worker()
        worker.start()
    except Exception as e:
        logger.error(f"Main Program Error. {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 