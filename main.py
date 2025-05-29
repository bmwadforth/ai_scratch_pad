from stable_diffusion import text_to_picture

# Example 1: Basic usage
print("\n--- Running Example 1 ---")
output_file_1 = "my_generated_image_1.png"
text_to_picture(
    prompt="A picture of a hill",
    output_path=output_file_1,
    height=512,  # Or even lower for testing, like 384
    width=512,  # Or even lower for testing, like 384
    seed=42
)

# Example 2: Different dimensions and steps
print("\n--- Running Example 2 ---")
output_file_2 = "my_generated_image_2.png"
text_to_picture(
    prompt="A cozy cottage in a snowy forest, digital painting",
    output_path=output_file_2,
    height=768,
    width=768,
    num_inference_steps=75,
    guidance_scale=8.0,
    seed=123
)