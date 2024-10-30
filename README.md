# Depth Anything V2

Depth estimation using the Depth Anything V2 model. This implementation provides a user-friendly CLI interface for both real-time and batch processing of images and videos.

## Features

- Multiple model sizes (Small, Base, Large, Giant)
- Real-time webcam depth estimation
- Batch processing of images and videos
- Customizable input resolution
- Grayscale or colored depth maps
- GPU acceleration support
- Progress tracking for batch operations

## Installation

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support (optional, but recommended)
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Install uv if you haven't already:
```bash
pipx install uv
```

2. Create and activate virtual environment:
```bash
# Create venv
uv venv

# Activate (Windows)
.venv/Scripts/activate

# Activate (Unix/macOS)
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

### Verify Installation

Test your setup by running:
```bash
python -m depth_anything.gpu_check
```

## Usage

### Command Line Interface

The package provides two main commands:
- `webcam`: Process real-time webcam feed
- `process`: Process images or videos

#### Webcam Mode

```bash
# Basic webcam usage
uv run -m depth_anything.cli webcam

# High-resolution webcam with Large model
uv run -m depth_anything.cli webcam --model-size Large --input-size 1024

# Grayscale output
uv run -m depth_anything.cli webcam --grayscale

# Force CPU or GPU usage
uv run -m depth_anything.cli webcam --device cpu  # or gpu
```

Webcam Controls:
- Press 'q' to quit
- Press 's' to save current frame

#### Processing Images and Videos

```bash
# Process a single image
uv run -m depth_anything.cli process image.jpg

# Process with custom settings
uv run -m depth_anything.cli process image.jpg \
    --model-size Large \
    --input-size 768 \
    --grayscale

# Process a directory of images
uv run -m depth_anything.cli process ./images_directory \
    --outdir ./depth_results

# Process a video
uv run -m depth_anything.cli process video.mp4 \
    --pred-only \
    --outdir ./video_results

# Process images listed in a text file
uv run -m depth_anything.cli process image_list.txt \
    --outdir ./batch_results
```

### Options

#### Global Options
- `--model-size`: Choose model size ['Small', 'Base', 'Large', 'Giant']
- `--device`: Force CPU or GPU usage ['cpu', 'gpu']
- `--input-size`: Input resolution for inference (default: 518)
- `--grayscale`: Output grayscale depth maps without color palette

#### Process Command Options
- `--outdir`: Output directory for results
- `--pred-only`: Save only depth maps without original images

### Input Types Support

The `process` command supports various input types:
1. Single image file (jpg, jpeg, png, bmp)
2. Directory containing multiple images
3. Text file with image paths (one per line)
4. Video file (mp4, avi, mov, mkv)

## Examples

### Basic Image Processing
```bash
# Process a single image with default settings
uv run -m depth_anything.cli process path/to/image.jpg

# Save results in a specific directory
uv run -m depth_anything.cli process path/to/image.jpg --outdir results
```

### Advanced Processing
```bash
# High-quality processing with Large model
uv run -m depth_anything.cli process path/to/image.jpg \
    --model-size Large \
    --input-size 1024 \
    --outdir high_quality_results

# Batch process a directory with grayscale output
uv run -m depth_anything.cli process path/to/images \
    --grayscale \
    --outdir grayscale_results

# Process video with depth-only output
uv run -m depth_anything.cli process path/to/video.mp4 \
    --pred-only \
    --outdir video_results
```

### Real-time Applications
```bash
# High-resolution webcam feed
uv run -m depth_anything.cli webcam \
    --model-size Large \
    --input-size 1024

# Efficient real-time processing
uv run -m depth_anything.cli webcam \
    --model-size Small \
    --input-size 384
```

## Model Details

Available model sizes and their characteristics:

| Model                      | Parameters | Best For                          |
|---------------------------|------------|-----------------------------------|
| Depth-Anything-V2-Small   | 24.8M      | Real-time, resource-constrained   |
| Depth-Anything-V2-Base    | 97.5M      | Balanced performance              |
| Depth-Anything-V2-Large   | 335.3M     | High-quality, detailed results    |
| Depth-Anything-V2-Giant   | 1.3B       | Maximum quality (experimental)    |

## Output

The tool generates:
- For images: JPEG files with original and depth map side by side (or depth only with `--pred-only`)
- For videos: MP4 files with original and depth frames (or depth only)
- For webcam: Real-time display and option to save frames

Output files are named using the pattern:
- Images: `depth_map_[original_name].jpg`
- Videos: `depth_[original_name].mp4`
- Webcam: `depth_webcam_[frame_number].jpg`

## Development

1. Install development dependencies:
```bash
uv pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest
```

3. Format code:
```bash
black .
isort .
```

## License

This project is licensed under the Apache-2.0 License. Note that the Depth-Anything-V2 models have their own licensing:
- Depth-Anything-V2-Small: Apache-2.0 license
- Depth-Anything-V2-Base/Large/Giant: CC-BY-NC-4.0 license

## Acknowledgments

This implementation is based on the [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) project. Please see their repository for the original implementation and research paper.
