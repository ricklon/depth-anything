import torch
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from dotenv import load_dotenv

class DepthEstimator:
    def __init__(self, model_size="Small", device=None):
        """
        Initialize the Depth Estimator with specified model size and device.
        
        Args:
            model_size (str): Size of the model ('Small', 'Base', 'Large')
            device (str): Device to run inference on ('cuda', 'cpu', or None for auto-detection)
        """
        self.output_dir = self._setup_environment()
        self.device = self._setup_device(device)
        self.model, self.processor = self._load_model(model_size)
    
    def _setup_environment(self):
        """Setup environment variables and create necessary directories."""
        load_dotenv()
        output_dir = Path("depth_outputs")
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def _setup_device(self, device):
        """Setup the computation device (CPU/GPU) with detailed checking."""
        print("\n=== Device Setup ===")
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Checking possible issues:")
            
            # Check CUDA installation
            if not hasattr(torch, 'cuda'):
                print("- PyTorch not compiled with CUDA support")
            else:
                print("- PyTorch CUDA support is available")
            
            # Check NVIDIA driver
            try:
                subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print("- NVIDIA driver is installed")
            except FileNotFoundError:
                print("- NVIDIA driver not found or nvidia-smi not in PATH")
            
            print("Falling back to CPU.")
            return 'cpu'
        
        if device is None:
            # Auto-detect with verbose output
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"CUDA is available:")
                print(f"- GPU Device: {torch.cuda.get_device_name(0)}")
                print(f"- CUDA Version: {torch.version.cuda}")
                print(f"- PyTorch CUDA: {torch.version.cuda}")
            else:
                device = 'cpu'
                print("CUDA is not available, using CPU")
        
        if device == 'cuda':
            # Additional GPU checks
            try:
                # Test GPU memory
                print("\nGPU Memory Status:")
                print(f"- Total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
                print(f"- Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                print(f"- Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                
                # Test GPU tensor creation
                test_tensor = torch.tensor([1.0], device='cuda')
                del test_tensor  # Clean up test tensor
                print("Successfully created test tensor on GPU")
                
            except Exception as e:
                print(f"Warning: GPU test failed: {e}")
                print("Falling back to CPU")
                return 'cpu'
        
        return device
    
    def _load_model(self, model_size):
        """Load the Depth Anything V2 model and processor."""
        model_ids = {
            "Small": "depth-anything/Depth-Anything-V2-Small-hf",
            "Base": "depth-anything/Depth-Anything-V2-Base-hf",
            "Large": "depth-anything/Depth-Anything-V2-Large-hf"
        }
        
        model_id = model_ids.get(model_size)
        if not model_id:
            raise ValueError(f"Invalid model size. Choose from: {', '.join(model_ids.keys())}")
        
        print(f"Loading model {model_id}...")
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        
        # Move model to appropriate device
        model = model.to(self.device)
        
        return model, image_processor
    
    def process_frame(self, frame):
        """Process a single frame and return depth map."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        post_processed_output = self.processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)]
        )
        
        predicted_depth = post_processed_output[0]["predicted_depth"]
        
        # Normalize for visualization
        depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min()) * 255
        depth = depth.cpu().numpy().astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
        return depth_colored
    
    def process_webcam(self):
        """Process webcam feed in real-time."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        
        print("\nWebcam Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            depth_map = self.process_frame(frame)
            
            # Display original and depth side by side
            combined = np.hstack((frame, depth_map))
            cv2.imshow('Original | Depth Map (Press q to quit, s to save)', combined)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                frame_count += 1
                output_path = self.output_dir / f"depth_webcam_{frame_count}.jpg"
                cv2.imwrite(str(output_path), combined)
                print(f"Saved frame to {output_path}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path, output_path=None):
        """Process a single image and return/save depth map."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        frame = cv2.imread(str(image_path))
        depth_map = self.process_frame(frame)
        
        # Display original and depth side by side
        combined = np.hstack((frame, depth_map))
        
        if output_path:
            cv2.imwrite(str(output_path), combined)
            print(f"Saved depth map to {output_path}")
        
        cv2.imshow('Original | Depth Map (Press any key to close)', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return depth_map
        
    @property
    def device_info(self):
        """Get information about the current device being used."""
        if self.device == 'cuda':
            return {
                'device': 'GPU',
                'name': torch.cuda.get_device_name(0),
                'memory_allocated': f"{torch.cuda.memory_allocated(0)/1024**2:.2f} MB",
                'memory_cached': f"{torch.cuda.memory_reserved(0)/1024**2:.2f} MB"
            }
        return {
            'device': 'CPU',
            'name': 'CPU'
        }