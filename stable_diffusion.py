import torch
from diffusers import StableDiffusionPipeline
import os

def text_to_picture(
    prompt: str,
    output_path: str,
    model_name: str = "runwayml/stable-diffusion-v1-5",
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = None
) -> str:
    """
    Generates an image from a text prompt using a Stable Diffusion model.

    This method automatically detects and uses the most suitable device:
    CUDA (for NVIDIA GPUs), MPS (for Apple Silicon Macs), or CPU.

    Args:
        prompt (str): The text prompt to generate the image from.
        output_path (str): The file path where the generated image will be saved (e.g., "output_image.png").
        model_name (str): The name of the pre-trained Stable Diffusion model to use.
                          Defaults to "runwayml/stable-diffusion-v1-5".
        height (int): The height of the generated image in pixels. Defaults to 512.
        width (int): The width of the generated image in pixels. Defaults to 512.
        num_inference_steps (int): The number of denoising steps. Higher steps generally
                                   result in higher quality but take longer. Defaults to 50.
        guidance_scale (float): A higher guidance scale encourages the model to generate
                                images closely related to the prompt, but it can also
                                reduce diversity. Defaults to 7.5.
        seed (int, optional): A random seed for reproducibility. If None, a random seed is used.

    Returns:
        str: The path to the saved image file if successful, an empty string otherwise.
    """
    # 1. Determine the device to use
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available. Using GPU for generation.")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("MPS is available. Using Apple Silicon GPU (M3) for generation.")
    else:
        device = "cpu"
        print("Neither CUDA nor MPS found. Falling back to CPU for generation.")

    # 2. Load the Stable Diffusion pipeline
    try:
        # Load the model from Hugging Face Hub
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16 if device != "cpu" else torch.float32)
        pipe.to(device) # Move the model to the detected device

        # If using MPS, enable attention slicing for potentially lower memory usage
        if device == "mps":
            pipe.enable_attention_slicing()

        # 3. Set up the random generator for reproducibility if a seed is provided
        generator = torch.Generator(device=device)
        if seed is not None:
            generator = generator.manual_seed(seed)
        else:
            # If no seed is provided, generate a random one for this run
            seed = torch.seed()
            generator = generator.manual_seed(seed)
            print(f"Using random seed: {seed}")


        # 4. Generate the image
        print(f"Generating image for prompt: '{prompt}'...")
        image = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        # 5. Save the image
        image.save(output_path)
        print(f"Image saved successfully to: {os.path.abspath(output_path)}") # Print absolute path for clarity
        return output_path

    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        return ""
