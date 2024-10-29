# Depth Anything

Depth estimation using the Depth Anything model.

## Installation

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Install uv if you haven't already:
```bash
pip install uv
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
uv pip install -r requirements.txt
```

For development, install additional dependencies:
```bash
uv pip install -r requirements-dev.txt
```

### Verify Installation

Run the verification script:
```bash
python verify_install.py
```

## Usage

Process an image:
```bash
python depth_anything/cli.py image path/to/image.jpg
```

Use webcam:
```bash
python depth_anything/cli.py webcam
```

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

4. Update dependencies:
```bash
uv pip compile requirements.txt -o requirements.lock
uv pip install -r requirements.lock
```