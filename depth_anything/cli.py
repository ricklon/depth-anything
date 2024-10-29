import click
from pathlib import Path
from .depth_inference import DepthEstimator

def validate_device(ctx, param, value):
    if value == 'gpu':
        return 'cuda'
    elif value == 'cpu':
        return 'cpu'
    elif value is None:
        return None
    raise click.BadParameter("Device must be either 'gpu' or 'cpu'")

@click.group()
def cli():
    """Depth Anything V2 - Depth Estimation Tool"""
    pass

@cli.command()
@click.option('--model-size', 
              type=click.Choice(['Small', 'Base', 'Large'], case_sensitive=True),
              default='Small',
              help='Model size to use for depth estimation')
@click.option('--device',
              type=str,
              callback=validate_device,
              help='Device to use for inference (gpu/cpu). Default: auto-detect')
def webcam(model_size, device):
    """Process webcam feed in real-time."""
    try:
        estimator = DepthEstimator(model_size=model_size, device=device)
        click.echo(f"Device info: {estimator.device_info}")
        estimator.process_webcam()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--model-size', 
              type=click.Choice(['Small', 'Base', 'Large'], case_sensitive=True),
              default='Small',
              help='Model size to use for depth estimation')
@click.option('--device',
              type=str,
              callback=validate_device,
              help='Device to use for inference (gpu/cpu). Default: auto-detect')
def image(image_path, model_size, device):
    """Process a single image file."""
    try:
        estimator = DepthEstimator(model_size=model_size, device=device)
        click.echo(f"Device info: {estimator.device_info}")
        output_path = Path("depth_outputs") / f"depth_map_{Path(image_path).stem}.jpg"
        estimator.process_image(image_path, output_path)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

if __name__ == '__main__':
    cli()