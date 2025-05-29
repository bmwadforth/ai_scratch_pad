import torch
from diffusers import DiffusionPipeline # More general pipeline for SD3 models
import os

def text_to_picture(
    prompt: str,
    output_path: str,
    # *** Updated to the best available Stable Diffusion 3.5 Medium model for 2025 ***
    model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers",
    # SD3 models generally perform well at 1024x1024
    height: int = 1024,
    width: int = 1024,
    # SD3 models can often achieve good quality with fewer steps than SD1.5,
    # but more than Turbo models. 28-50 is a good range.
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0, # Standard guidance scale often works well
    seed: int = None
) -> str:
    """
    Generates an image from a text prompt using a Stable Diffusion model.

    This method automatically detects and uses the most suitable device:
    CUDA (for NVIDIA GPUs), MPS (for Apple Silicon Macs), or CPU.

    Args:
        prompt (str): The text prompt to generate the image from.
        output_path (str): The filename for the generated image (e.g., "output_image.png").
                           The image will be saved in the current executing script's directory.
        model_name (str): The name of the pre-trained Stable Diffusion model to use.
                          Defaults to "stabilityai/stable-diffusion-3-medium-diffusers".
        height (int): The height of the generated image in pixels. Defaults to 1024 for SD3.
        width (int): The width of the generated image in pixels. Defaults to 1024 for SD3.
        num_inference_steps (int): The number of denoising steps. SD3 often performs well
                                   between 28 and 50 steps. Defaults to 28.
        guidance_scale (float): A higher guidance scale encourages the model to generate
                                images closely related to the prompt, but it can also
                                reduce diversity. Defaults to 7.0.
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
        # SD3 models are best loaded with torch_dtype=torch.float16 for GPU/MPS
        # and generally use DiffusionPipeline
        print(f"Attempting to load model: {model_name} on {device}...")
        pipe = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            # For SD3, you might need to handle text encoders if memory is very tight
            # For example, if you get OOM, you could try:
            # text_encoder_3=None, tokenizer_3=None,
            # This would disable the larger T5 text encoder, reducing VRAM but potentially prompt adherence.
        )

        # Move the model to the detected device
        pipe.to(device)

        # Enable optimizations for lower memory usage on GPU/MPS
        if device == "cuda":
            try:
                import xformers
                pipe.enable_xformers_memory_efficient_attention()
                print("xFormers enabled for CUDA.")
            except ImportError:
                print("xFormers not installed. Install with 'pip install xformers' for better performance.")
            pipe.enable_attention_slicing()
            # If still OOM, try CPU offload (will be slower)
            # pipe.enable_model_cpu_offload()
            # print("Model CPU offload enabled (will be slower).")
        elif device == "mps":
            pipe.enable_attention_slicing()
            # For MPS, enable_model_cpu_offload can also be very helpful for large models
            # pipe.enable_model_cpu_offload()
            # print("Model CPU offload enabled (will be slower on MPS).")

        # 3. Set up the random generator for reproducibility if a seed is provided
        generator = torch.Generator(device=device)
        if seed is not None:
            generator = generator.manual_seed(seed)
        else:
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

        # 5. Save the image in the current directory
        image.save(output_path)
        print(f"Image saved successfully to: {os.path.abspath(output_path)}")
        return output_path

    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA Out of Memory Error: {e}")
        print("This typically means your GPU does not have enough VRAM to load or process the model.")
        print(f"The model '{model_name}' (SD3.5 Medium) generally requires ~10GB VRAM at 1024x1024 resolution.")
        print("Suggestions: 1. Ensure 'torch_dtype=torch.float16'. 2. Enable 'xformers' (for CUDA). 3. Try 'pipe.enable_model_cpu_offload()'. 4. Reduce 'height' and 'width'.")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred during image generation: {e}")
        return ""