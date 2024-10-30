import click
from pathlib import Path
import cv2
from .depth_inference import DepthEstimator

def validate_device(ctx, param, value):
    if value == 'gpu':
        return 'cuda'
    elif value == 'cpu':
        return 'cpu'
    elif value is None:
        return None
    raise click.BadParameter("Device must be either 'gpu' or 'cpu'")

def validate_input_size(ctx, param, value):
    if value is None:
        return 518  # Default size from original repo
    if value < 64:
        raise click.BadParameter("Input size must be at least 64 pixels")
    return value

def validate_save_format(ctx, param, value):
    formats = ['jpg', 'png']
    if value not in formats:
        raise click.BadParameter(f"Save format must be one of: {', '.join(formats)}")
    return value

@click.group()
def cli():
    """Depth Anything V2 - Depth Estimation Tool"""
    pass

@cli.command()
@click.option('--model-size', 
              type=click.Choice(['Small', 'Base', 'Large', 'Giant'], case_sensitive=True),
              default='Small',
              help='Model size to use for depth estimation')
@click.option('--device',
              type=str,
              callback=validate_device,
              help='Device to use for inference (gpu/cpu). Default: auto-detect')
@click.option('--input-size',
              type=int,
              callback=validate_input_size,
              default=518,
              help='Input size for model inference. Default: 518')
@click.option('--grayscale',
              is_flag=True,
              help='Save grayscale depth map without color palette')
@click.option('--save-format',
              type=str,
              default='jpg',
              callback=validate_save_format,
              help='Format for saving images (jpg, png). Default: jpg')
@click.option('--quality',
              type=int,
              default=95,
              help='Image save quality (0-100, for JPEG only). Default: 95')
def webcam(model_size, device, input_size, grayscale, save_format, quality):
    """Process webcam feed in real-time.
    
    Controls:
    \b
    - Press 'q' to quit
    - Press 's' to save current frame (combined view)
    - Press 'd' to save depth map only
    - Press 'b' to save both frames separately
    - Press 'r' to start/stop recording
    - Press 'm' to change recording mode while recording
    - Press 'p' to pause/resume
    """
    try:
        estimator = DepthEstimator(
            model_size=model_size, 
            device=device,
            input_size=input_size,
            grayscale=grayscale
        )
        click.echo(f"Device info: {estimator.device_info}")
        click.echo("\nStarting webcam processing...")
        estimator.process_webcam()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--model-size', 
              type=click.Choice(['Small', 'Base', 'Large', 'Giant'], case_sensitive=True),
              default='Small',
              help='Model size to use for depth estimation')
@click.option('--device',
              type=str,
              callback=validate_device,
              help='Device to use for inference (gpu/cpu). Default: auto-detect')
@click.option('--input-size',
              type=int,
              callback=validate_input_size,
              default=518,
              help='Input size for model inference. Default: 518')
@click.option('--outdir',
              type=click.Path(),
              default='depth_outputs',
              help='Output directory for depth maps')
@click.option('--pred-only',
              is_flag=True,
              help='Only save the predicted depth map, without raw image')
@click.option('--grayscale',
              is_flag=True,
              help='Save grayscale depth map without color palette')
@click.option('--save-format',
              type=str,
              default='jpg',
              callback=validate_save_format,
              help='Format for saving images (jpg, png). Default: jpg')
@click.option('--quality',
              type=int,
              default=95,
              help='Image save quality (0-100, for JPEG only). Default: 95')
@click.option('--batch-size',
              type=int,
              default=1,
              help='Batch size for processing multiple images. Default: 1')
@click.option('--no-display',
              is_flag=True,
              help='Disable preview window when processing')
def process(input_path, model_size, device, input_size, outdir, pred_only, 
           grayscale, save_format, quality, batch_size, no_display):
    """Process images or video for depth estimation.
    
    INPUT_PATH can be:
    - A single image file
    - A directory containing images
    - A text file with image paths
    - A video file
    """
    try:
        estimator = DepthEstimator(
            model_size=model_size, 
            device=device,
            input_size=input_size,
            grayscale=grayscale
        )
        click.echo(f"Device info: {estimator.device_info}")
        
        input_path = Path(input_path)
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True)
        
        if input_path.is_file():
            if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                # Process video
                click.echo("\nProcessing video...")
                estimator.process_video(str(input_path), outdir, pred_only)
            elif input_path.suffix.lower() == '.txt':
                # Process list of images from text file
                click.echo("\nProcessing images from list...")
                with open(input_path) as f:
                    image_paths = [line.strip() for line in f if line.strip()]
                with click.progressbar(image_paths, label='Processing images') as images:
                    for img_path in images:
                        if Path(img_path).exists():
                            output_path = outdir / f"depth_map_{Path(img_path).stem}.{save_format}"
                            estimator.process_image(
                                img_path, 
                                output_path, 
                                pred_only,
                                display=not no_display
                            )
                        else:
                            click.echo(f"Warning: Image not found: {img_path}", err=True)
            else:
                # Process single image
                click.echo("\nProcessing single image...")
                output_path = outdir / f"depth_map_{input_path.stem}.{save_format}"
                estimator.process_image(
                    str(input_path), 
                    output_path,
                    pred_only,
                    display=not no_display
                )
        elif input_path.is_dir():
            # Process directory of images
            click.echo("\nProcessing directory of images...")
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = [f for f in input_path.iterdir() 
                         if f.suffix.lower() in image_extensions]
            
            with click.progressbar(image_files, label='Processing images') as images:
                for img_path in images:
                    output_path = outdir / f"depth_map_{img_path.stem}.{save_format}"
                    estimator.process_image(
                        str(img_path),
                        output_path,
                        pred_only,
                        display=not no_display
                    )
        else:
            raise click.BadParameter(f"Invalid input path: {input_path}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

def main():
    cli()

if __name__ == '__main__':
    main()