import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models.utils.parallel_transformer_processor import ParallelTransformerProcessor
from models.distributed.config import DistributedConfig
import time
import logging
from torchvision import datasets
import traceback
from datetime import datetime

# Creating a log directory
log_dir = "/home/wxg/dri/sdrm-dyz/logs"
os.makedirs(log_dir, exist_ok=True)

# Generate log file names (using timestamps)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"sdrm_{current_time}.log")

# Remove all existing processors
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configuring the Root Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Getting the Root Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Ensure that all module loggers inherit the settings of the root logger
logging.getLogger('models').setLevel(logging.INFO)
logging.getLogger('models.utils').setLevel(logging.INFO)

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        
        # Check if the directory exists
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image catalog does not exist: {image_dir}")
        
        # Traversing the Anomal and Normal subdirectories
        for class_id, class_name in enumerate(['Normal', 'Anomal']):
            class_dir = os.path.join(image_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            # Get all image files under this category
            files = [
                f for f in os.listdir(class_dir)
                if f.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'))
            ]
            
            # Add full path and tags
            for f in files:
                self.image_files.append(os.path.join(class_name, f))
                self.labels.append(class_id)
        
        # Check if the image file is found
        if not self.image_files:
            raise ValueError(f"No image files were found in the directory {image_dir}")
            
        logger.info(f"find {len(self.image_files)} image files in {image_dir}")
        logger.info(f"Category distribution: Normal: {self.labels.count(0)}, Anomal: {self.labels.count(1)}")
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

def main():
    try:
        # 1. Configuration initialization
        config = DistributedConfig(
            raspberry_ips=['192.168.3.28', '192.168.3.26', '192.168.3.29'],
            num_classes=2
        )
        
        config.update(
            batch_size=4,
            num_epochs=5,
            learning_rate=0.0001
        )
        
        device = torch.device(config.device)
        logger.info(f"Utilization equipment: {device}")
        
        # 2. Data preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 3. Load Dataset
        # Test Data Set
        dataset_root = "/home/wxg/dri/dataset/VisA_ImageNet_224"
        # pcb dataset
        # dataset_root = "/home/wxg/dri/dataset/VisA_pcb_224"
        # Full data set
        # dataset_root = "/home/wxg/chen/Galaxy/dataset/VisA_ImageNet_224"
        train_path = os.path.join(dataset_root, "train")
        test_path = os.path.join(dataset_root, "test")
        
        logger.info(f"Training set path: {train_path}")
        logger.info(f"Test Set Path: {test_path}")
        
        # Check if the directory exists
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training set catalog does not exist: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test set directory does not exist: {test_path}")
            
        train_dataset = ImageDataset(train_path, transform=transform)
        test_dataset = ImageDataset(test_path, transform=transform)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # 4. Initializing the Processor
        processor = ParallelTransformerProcessor(config).to(device)
        logger.info("Processor initialization complete")
        
        # Adding Performance Statistics Variables
        svd_times = []
        attention_times = []
        total_images = 0
        epoch_stats = []
        
        # 5. feature extraction stage
        logger.info("=== Starting the feature extraction phase ===")
        feature_start_time = time.time()
        
        try:
            for batch_idx, (images, labels) in enumerate(train_loader):
                with torch.no_grad():
                    images = images.to(device)
                    labels = labels.to(device)
                    total_images += images.size(0)
                    
                    # Record SVD start time
                    svd_start_time = time.time()
                    
                    try:
                        # SVD processing using the _process_image_locally method
                        features = processor._process_image_locally(images, training=False)
                        
                        if features is None:
                            logger.warning("Local image processing fails, downgrade to raw feature extraction")
                            features = processor.feature_extraction(images)
                        
                        features = features.to(device)
                        
                    except Exception as e:
                        logger.error(f"Image Processing Errors: {str(e)}")
                        logger.error(traceback.format_exc())
                        # Downgrade to raw feature extraction
                        features = processor.feature_extraction(images)
                        features = features.to(device)
                    
                    svd_end = time.time()
                    svd_times.append(svd_end - svd_start_time)
                    
                    # Recording attention span
                    attention_start = time.time()
                    features = processor.process_image(images, training=False)  # Change to False to output communication cost
                    attention_end = time.time()
                    attention_times.append(attention_end - attention_start)
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"Feature extraction：Processed {batch_idx * config.batch_size} images")
                        # Ensure logs are written immediately
                        for handler in logger.handlers:
                            handler.flush()
                        
                        # Clean the memory regularly
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        
                        # Monitoring Memory Usage
                        if torch.cuda.is_available():
                            memory_used = torch.cuda.memory_allocated() / 1024**2
                            memory_cached = torch.cuda.memory_reserved() / 1024**2
                            logger.info(f"GPU Memory Usage: {memory_used:.2f} MB (allocated) / {memory_cached:.2f} MB (cache)")
                    
                    # Remove unwanted tensor
                    del images, labels, features
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
        feature_extraction_time = time.time() - feature_start_time
        
        # 6. Train the classifier
        logger.info("\n=== Starting the classifier training phase ===")
        training_start_time = time.time()
        
        try:
            train_losses, val_accuracies = processor.train_classifier(
                train_loader=train_loader,
                val_loader=test_loader,
                num_epochs=config.num_epochs,
                learning_rate=config.learning_rate
            )
            
        except Exception as e:
            logger.error(f"Classifier training error: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        training_time = time.time() - training_start_time
        
        # Computational performance statistics
        total_svd_time = sum(svd_times)
        total_attention_time = sum(attention_times)
        avg_svd_time = total_svd_time / len(svd_times) if svd_times else 0
        avg_attention_time = total_attention_time / len(attention_times) if attention_times else 0
        
        # 8. Output detailed performance reports
        logger.info("\n=== Detailed Performance Report ===")
        logger.info(f"Feature extraction time: {feature_extraction_time:.2f} s")
        logger.info(f"Total SVD time: {total_svd_time:.2f} s")
        logger.info(f"Long Attention Total Time: {total_attention_time:.2f} s")
        logger.info(f"Average SVD time per map: {avg_svd_time:.4f} s")
        logger.info(f"Average attention span per chart: {avg_attention_time:.4f} s")
        logger.info(f"Classifier training time: {training_time:.2f} s")
        
        # Add communication cost report
        total_data_size = 0
        total_transfer_time = 0
        
        try:
            # Get the communication data recorded in the processor
            if hasattr(processor, 'communication_stats'):
                total_data_size = processor.communication_stats.get('total_data_size', 0)
                total_transfer_time = processor.communication_stats.get('total_transfer_time', 0)
            
            # Convert data size to a more readable format
            data_size_mb = total_data_size / (1024 * 1024)  # Convert to MB
            transfer_speed = data_size_mb / total_transfer_time if total_transfer_time > 0 else 0
            
            logger.info("\n=== Communication Cost Report ===")
            logger.info(f"Total data transferred: {data_size_mb:.2f} MB")
            logger.info(f"total transmission time: {total_transfer_time:.2f} s")
            logger.info(f"Average transmission speed: {transfer_speed:.2f} MB/s")
            logger.info(f"Average transmission per image: {(data_size_mb/total_images):.2f} MB/Picture")
            logger.info(f"Average transfer time per image: {(total_transfer_time/total_images):.4f} s/Picture")
            
        except Exception as e:
            logger.warning(f"Error getting communication cost data: {str(e)}")
        
        logger.info(f"\nTotal processing time: {feature_extraction_time + training_time:.2f} s")
        logger.info(f"Total number of images processed: {total_images}")
        logger.info("===============")
        
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")
        logger.error(f"Error Details: {traceback.format_exc()}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Program execution complete, resources cleaned up")

    logger.handlers[0].flush()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProgram interrupted by user")
    except Exception as e:
        logger.error(f"Abnormal program exit: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("program exit")