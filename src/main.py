import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device}")

model_id = "microsoft/Florence-2-base"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch_dtype).to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code = True)
print("Model and processor loaded successfully.")

def process_image(image_path, query = "eyeglass frames"):
    image = Image.open(image_path).convert("RGB")
    
    # Florence 2 prompt
    prompt = f"<REFERRING_EXPRESSION_SEGMENTATION> {query}"
    # Model inputs
    inputs = processor(text = prompt, images = image, return_tensors = "pt").to(device, torch_dtype)
    
    with torch.no_grad():
        generated_ids = model.generate(input_ids = inputs["input_ids"],
                                       pixel_values = inputs["pixel_values"],
                                       max_new_tokens = 1024,
                                       num_beams = 3
                                       )
        
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens = False)[0]
    
    results = processor.post_process_generation(
        generated_text,
        task = "<REFERRING_EXPRESSION_SEGMENTATION>",
        image_size = (image.width, image.height)
    )
    
    return image, results["<REFERRING_EXPRESSION_SEGMENTATION>"]

def draw_segmentation(image, data):
    image_rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", image_rgba.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
        
    if not data['polygons']:
        return image.convert("RGB")
    
    for polygons, label in zip(data['polygons'], data['labels']):
        for polygon in polygons:
            draw.polygon(polygon, outline="red", fill=(255, 0, 0, 80))
    
    combined = Image.alpha_composite(image_rgba, overlay)
    return combined.convert("RGB")
    
if __name__ == "__main__":
    input_img = "image.jpg"
    #Trying to exclude the glass lenses from segmentation, but didn't work.
    prompt = "the solid rim and temples of the eyeglasses, strictly excluding the glass lenses"
    
    if os.path.exists(input_img):
        print(f"Processing image: {input_img}")
        original_img , results = process_image(input_img, query=prompt)
        
        output_img = draw_segmentation(original_img, results)
        output_img.show()
        output_img.save("result.jpg")
        print("Processing completed. Result saved as result.jpg")
    else:
        print(f"Input image '{input_img}' not found. Please place the image in the same directory as this script and name it 'test.jpg'.")