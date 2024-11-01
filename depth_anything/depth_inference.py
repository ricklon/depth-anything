import torch
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime
import time

class DepthEstimator:
    def __init__(self, model_size="Small", device=None, input_size=518, grayscale=False):
        """
        Initialize the Depth Estimator with specified model size and device.
        
        Args:
            model_size (str): Size of the model ('Small', 'Base', 'Large', 'Giant')
            device (str): Device to run inference on ('cuda', 'cpu', or None for auto-detection)
            input_size (int): Input size for model inference (default: 518)
            grayscale (bool): If True, output grayscale depth maps without color palette
        """
        self.input_size = input_size
        self.grayscale = grayscale
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
            
            print("Falling back to CPU.")
            return 'cpu'
        
        if device is None:
            # Auto-detect with verbose output
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"CUDA is available:")
                print(f"- GPU Device: {torch.cuda.get_device_name(0)}")
                print(f"- CUDA Version: {torch.version.cuda}")
            else:
                device = 'cpu'
                print("CUDA is not available, using CPU")
        
        return device
    
    def _load_model(self, model_size):
        """Load the Depth Anything V2 model and processor."""
        model_ids = {
            "Small": "depth-anything/Depth-Anything-V2-Small-hf",
            "Base": "depth-anything/Depth-Anything-V2-Base-hf",
            "Large": "depth-anything/Depth-Anything-V2-Large-hf",
            "Giant": "depth-anything/Depth-Anything-V2-Giant-hf"
        }
        
        model_id = model_ids.get(model_size)
        if not model_id:
            raise ValueError(f"Invalid model size. Choose from: {', '.join(model_ids.keys())}")
        
        print(f"Loading model {model_id}...")
        try:
            image_processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id)
            
            # Move model to appropriate device
            model = model.to(self.device)
            return model, image_processor
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_id}: {e}")
    
    def process_frame(self, frame, return_raw=False):
        """
        Process a single frame and return depth map.
        
        Args:
            frame: numpy array in BGR format (from cv2)
            return_raw (bool): If True, return raw depth values instead of visualization
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Process image
            inputs = self.processor(
                images=image, 
                size={"height": self.input_size, "width": self.input_size}, 
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = self.processor.post_process_depth_estimation(
                    outputs,
                    target_sizes=[(image.height, image.width)]
                )[0]["predicted_depth"]
            
            if return_raw:
                return predicted_depth.cpu().numpy()
            
            # Normalize for visualization
            depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min()) * 255
            depth = depth.cpu().numpy().astype(np.uint8)
            
            # Apply colormap if not grayscale
            if not self.grayscale:
                depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            else:
                # Convert grayscale to 3-channel for display compatibility
                depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            
            return depth
            
        except Exception as e:
            raise RuntimeError(f"Error processing frame: {e}")

    def process_video(self, video_path, output_dir, pred_only=False):
        """Process a video file and save depth estimation results."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        try:
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup output video writer
            output_path = Path(output_dir) / f"depth_{Path(video_path).stem}.mp4"
            if pred_only:
                out_width = frame_width
                out_height = frame_height
            else:
                out_width = frame_width * 2  # Side by side
                out_height = frame_height
            
            # Try different codecs
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            except:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # fallback to MP4V
                except:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # last resort
            
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height), True)
            if not out.isOpened():
                raise IOError(f"Failed to create video writer for {output_path}")
            
            with tqdm(total=total_frames, desc="Processing video") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    depth_map = self.process_frame(frame)
                    
                    # Prepare output frame
                    if pred_only:
                        out_frame = depth_map
                    else:
                        out_frame = np.hstack((frame, depth_map))
                    
                    # Write frame
                    out.write(out_frame)
                    pbar.update(1)
                    
        except Exception as e:
            raise RuntimeError(f"Error processing video: {e}")
            
        finally:
            cap.release()
            out.release()
            print(f"Saved processed video to {output_path}")

    def process_image(self, image_path, output_path=None, pred_only=False, display=True):
        """Process a single image and save/display results."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Read image
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise IOError(f"Failed to read image: {image_path}")
            
            depth_map = self.process_frame(frame)
            
            if output_path:
                # Prepare output image
                if pred_only:
                    output_image = depth_map
                else:
                    output_image = np.hstack((frame, depth_map))
                
                # Save output
                success = cv2.imwrite(str(output_path), output_image)
                if not success:
                    raise IOError(f"Failed to save output image to {output_path}")
                print(f"Saved depth map to {output_path}")
            
            if display:
                # Display original and depth side by side
                combined = np.hstack((frame, depth_map))
                cv2.imshow('Original | Depth Map (Press any key to close)', combined)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return depth_map
            
        except Exception as e:
            raise RuntimeError(f"Error processing image: {e}")

    def save_frame(self, frame, depth_map, save_type='combined'):
        """
        Save the current frame and depth map with improved error handling.
        
        Args:
            frame: Original webcam frame
            depth_map: Processed depth map
            save_type: Type of save operation ('combined', 'separate', or 'depth_only')
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if save_type == 'combined':
                combined = np.hstack((frame, depth_map))
                output_path = self.output_dir / f"depth_webcam_combined_{timestamp}.jpg"
                success = cv2.imwrite(str(output_path), combined)
                if not success:
                    raise IOError("Failed to save combined frame")
                print(f"Saved combined frame to {output_path}")
                
            elif save_type == 'separate':
                frame_path = self.output_dir / f"webcam_original_{timestamp}.jpg"
                depth_path = self.output_dir / f"depth_map_{timestamp}.jpg"
                
                success1 = cv2.imwrite(str(frame_path), frame)
                success2 = cv2.imwrite(str(depth_path), depth_map)
                
                if not success1 or not success2:
                    raise IOError("Failed to save one or both frames")
                    
                print(f"Saved original frame to {frame_path}")
                print(f"Saved depth map to {depth_path}")
                
            elif save_type == 'depth_only':
                depth_path = self.output_dir / f"depth_only_{timestamp}.jpg"
                success = cv2.imwrite(str(depth_path), depth_map)
                if not success:
                    raise IOError("Failed to save depth map")
                print(f"Saved depth map to {depth_path}")
                
        except Exception as e:
            print(f"Error saving frame: {e}")
            return False
        
        return True

    def start_recording(self, frame_width, frame_height, save_type='combined'):
        """Initialize video recording with improved quality settings."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Try different codecs in order of preference
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        except:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # fallback to MP4V
            except:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # last resort
        
        fps = 30.0
        
        if save_type == 'combined':
            output_path = self.output_dir / f"depth_webcam_recording_{timestamp}.mp4"
            out = cv2.VideoWriter(
                str(output_path), 
                fourcc, 
                fps, 
                (frame_width * 2, frame_height),
                isColor=True
            )
            if not out.isOpened():
                raise IOError(f"Failed to create video writer for {output_path}")
            return out, output_path
            
        elif save_type == 'separate':
            orig_path = self.output_dir / f"webcam_original_{timestamp}.mp4"
            depth_path = self.output_dir / f"depth_recording_{timestamp}.mp4"
            
            out_orig = cv2.VideoWriter(
                str(orig_path), 
                fourcc, 
                fps, 
                (frame_width, frame_height),
                isColor=True
            )
            out_depth = cv2.VideoWriter(
                str(depth_path), 
                fourcc, 
                fps, 
                (frame_width, frame_height),
                isColor=True
            )
            
            if not out_orig.isOpened() or not out_depth.isOpened():
                raise IOError("Failed to create video writers")
                
            return (out_orig, out_depth), (orig_path, depth_path)
            
        elif save_type == 'depth_only':
            output_path = self.output_dir / f"depth_only_{timestamp}.mp4"
            out = cv2.VideoWriter(
                str(output_path), 
                fourcc, 
                fps, 
                (frame_width, frame_height),
                isColor=True
            )
            if not out.isOpened():
                raise IOError(f"Failed to create video writer for {output_path}")
            return out, output_path

    def process_webcam(self):
        """Process webcam feed in real-time."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        
        print("\nWebcam Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame (combined view)")
        print("- Press 'd' to save depth map only")
        print("- Press 'b' to save both frames separately")
        print("- Press 'r' to start/stop recording")
        print("- Press 'm' to change recording mode while recording")
        print("- Press 'p' to pause/resume")
        
        frame_count = 0
        recording = False
        paused = False
        out = None
        out_depth = None  # Separate writer for depth in "both" mode
        recording_mode = "combined"  # can be "combined", "depth", or "both"
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    depth_map = self.process_frame(frame)
                    
                    # Display original and depth side by side
                    combined = np.hstack((frame, depth_map))
                    cv2.imshow('Original | Depth Map (Press q:quit, s/d/b:save, r:record, m:mode, p:pause)', combined)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save combined view
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = self.output_dir / f"depth_webcam_combined_{timestamp}.jpg"
                    cv2.imwrite(str(output_path), combined)
                    print(f"Saved combined frame to {output_path}")
                elif key == ord('d'):
                    # Save depth map only
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = self.output_dir / f"depth_webcam_depth_{timestamp}.jpg"
                    cv2.imwrite(str(output_path), depth_map)
                    print(f"Saved depth map to {output_path}")
                elif key == ord('b'):
                    # Save both frames separately
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    original_path = self.output_dir / f"depth_webcam_original_{timestamp}.jpg"
                    depth_path = self.output_dir / f"depth_webcam_depth_{timestamp}.jpg"
                    cv2.imwrite(str(original_path), frame)
                    cv2.imwrite(str(depth_path), depth_map)
                    print(f"Saved original to {original_path}")
                    print(f"Saved depth map to {depth_path}")
                elif key == ord('r'):
                    if not recording:
                        # Start recording
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        try:
                            # Try different codecs in order of preference
                            codecs = [
                                ('mp4v', '.mp4'),  # MP4 codec
                                ('XVID', '.avi'),  # XVID codec
                                ('MJPG', '.avi'),  # Motion JPEG
                            ]
                            
                            for codec, ext in codecs:
                                try:
                                    fourcc = cv2.VideoWriter_fourcc(*codec)
                                    
                                    if recording_mode == "combined":
                                        output_path = self.output_dir / f"depth_webcam_recording_{timestamp}{ext}"
                                        out = cv2.VideoWriter(str(output_path), fourcc, 20.0, 
                                                            (combined.shape[1], combined.shape[0]))
                                    elif recording_mode == "depth":
                                        output_path = self.output_dir / f"depth_webcam_depth_recording_{timestamp}{ext}"
                                        out = cv2.VideoWriter(str(output_path), fourcc, 20.0,
                                                            (depth_map.shape[1], depth_map.shape[0]))
                                    else:  # both
                                        output_path = self.output_dir / f"depth_webcam_original_recording_{timestamp}{ext}"
                                        output_path_depth = self.output_dir / f"depth_webcam_depth_recording_{timestamp}{ext}"
                                        out = cv2.VideoWriter(str(output_path), fourcc, 20.0,
                                                            (frame.shape[1], frame.shape[0]))
                                        out_depth = cv2.VideoWriter(str(output_path_depth), fourcc, 20.0,
                                                                (depth_map.shape[1], depth_map.shape[0]))
                                    
                                    # Test if writer initialized successfully
                                    if recording_mode == "both":
                                        if out is not None and out_depth is not None and out.isOpened() and out_depth.isOpened():
                                            break
                                    else:
                                        if out is not None and out.isOpened():
                                            break
                                            
                                except Exception as e:
                                    print(f"Failed to initialize {codec} codec: {e}")
                                    if out is not None:
                                        out.release()
                                    if out_depth is not None:
                                        out_depth.release()
                                    out = None
                                    out_depth = None
                                    continue
                            
                            if (recording_mode == "both" and (out is None or out_depth is None)) or \
                            (recording_mode != "both" and out is None):
                                print("Failed to initialize any video codec")
                                continue
                                
                            recording = True
                            print(f"Started recording in {recording_mode} mode")
                            if recording_mode == "both":
                                print(f"Recording to {output_path} and {output_path_depth}")
                            else:
                                print(f"Recording to {output_path}")
                                
                        except Exception as e:
                            print(f"Error starting recording: {e}")
                            if out is not None:
                                out.release()
                            if out_depth is not None:
                                out_depth.release()
                            out = None
                            out_depth = None
                            recording = False
                    else:
                        # Stop recording
                        if out is not None:
                            out.release()
                        if out_depth is not None:
                            out_depth.release()
                        out = None
                        out_depth = None
                        recording = False
                        print("Stopped recording")
                elif key == ord('m') and recording:
                    # Change recording mode while recording
                    modes = ["combined", "depth", "both"]
                    current_idx = modes.index(recording_mode)
                    recording_mode = modes[(current_idx + 1) % len(modes)]
                    print(f"Switched to {recording_mode} recording mode")
                    # Stop current recording and start new one with new mode
                    if recording:
                        if out is not None:
                            out.release()
                        if out_depth is not None:
                            out_depth.release()
                        out = None
                        out_depth = None
                        recording = False
                        key = ord('r')  # Simulate pressing 'r' to start new recording
                        continue
                elif key == ord('p'):
                    # Toggle pause
                    paused = not paused
                    status = "Paused" if paused else "Resumed"
                    print(f"{status} processing")
                
                # Write frame if recording
                if recording and not paused:
                    if recording_mode == "combined" and out is not None:
                        out.write(combined)
                    elif recording_mode == "depth" and out is not None:
                        out.write(depth_map)
                    elif recording_mode == "both" and out is not None and out_depth is not None:
                        out.write(frame)
                        out_depth.write(depth_map)
        
        finally:
            # Clean up
            if out is not None:
                out.release()
            if out_depth is not None:
                out_depth.release()
            cap.release()
            cv2.destroyAllWindows()
                        
    @property
    def device_info(self):
        """Get information about the current device being used."""
        try:
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
        except Exception as e:
            print(f"Error getting device info: {e}")
            return {'device': 'Unknown', 'error': str(e)}