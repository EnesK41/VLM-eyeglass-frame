import torch
from transformers import AutoProcessor, AutoModelForCausalLM, SamModel, SamProcessor
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import re
import traceback
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"[Init] Device: {device}, dtype: {torch_dtype}")

florence_id = "microsoft/Florence-2-base"
print("[Init] Loading Florence-2...")
florence_model = AutoModelForCausalLM.from_pretrained(florence_id, trust_remote_code=True, torch_dtype=torch_dtype).to(device)
florence_processor = AutoProcessor.from_pretrained(florence_id, trust_remote_code=True)
print("[Init] Florence-2 ready.")

sam_id = "facebook/sam-vit-base"
print("[Init] Loading SAM...")
sam_model = SamModel.from_pretrained(sam_id).to(device)
sam_processor = SamProcessor.from_pretrained(sam_id)
print("[Init] SAM ready.")

MAX_BBOX_AREA_RATIO = 0.28

def parse_vqa_yes_no(text):
    normalized = text.lower().strip()
    if re.search(r"\byes\b", normalized):
        return True
    if re.search(r"\bno\b", normalized):
        return False
    return None

def check_has_glasses(image):
    prompt = "<VQA> Are there eyeglasses or sunglasses in this image? Answer only yes or no."
    inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=64,
            num_beams=3
        )
    answer_raw = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    answer_clean = florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parsed_answer = florence_processor.post_process_generation(answer_raw, task="<VQA>", image_size=(image.width, image.height))
    parsed_text = str(parsed_answer.get("<VQA>", "")).strip()

    # Florence-2 sometimes returns location tokens instead of a strict yes/no answer.
    if "<loc_" in parsed_text.lower():
        return None, parsed_text

    decision = parse_vqa_yes_no(parsed_text)
    if decision is None:
        decision = parse_vqa_yes_no(answer_clean)

    return decision, (parsed_text or answer_clean)

def get_florence_bboxes(image, query):
    prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {query}"
    inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    
    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    results = florence_processor.post_process_generation(generated_text, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(image.width, image.height))
    return results['<CAPTION_TO_PHRASE_GROUNDING>'].get('bboxes', [])

def filter_large_bboxes(bboxes, image_width, image_height, max_area_ratio):
    filtered = []
    image_area = float(image_width * image_height)

    for idx, bbox in enumerate(bboxes, start=1):
        if len(bbox) != 4:
            print(f"[Filter] Box {idx} invalid format. Rejected.")
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox]

        # Keep coordinates inside image and enforce proper min/max order.
        x1 = min(max(x1, 0.0), image_width - 1.0)
        x2 = min(max(x2, 0.0), image_width - 1.0)
        y1 = min(max(y1, 0.0), image_height - 1.0)
        y2 = min(max(y2, 0.0), image_height - 1.0)

        x_min, x_max = sorted((x1, x2))
        y_min, y_max = sorted((y1, y2))

        width = x_max - x_min
        height = y_max - y_min
        if width <= 1 or height <= 1:
            print(f"[Filter] Box {idx} near-zero area. Rejected.")
            continue

        area_ratio = (width * height) / image_area
        if area_ratio > max_area_ratio:
            print(f"[Filter] Box {idx} too large (ratio={area_ratio:.3f}). Rejected.")
            continue

        filtered.append([x_min, y_min, x_max, y_max])
        print(f"[Filter] Box {idx} kept (ratio={area_ratio:.3f}).")

    return filtered

def get_sam_mask(image, bbox, confidence_threshold=0.10):
    inputs = sam_processor(image, input_boxes=[[[bbox]]], return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = sam_model(**inputs)
        
    iou_scores = outputs.iou_scores[0, 0].cpu().numpy()
    best_idx = iou_scores.argmax()
    best_score = iou_scores[best_idx]
    
    if best_score < confidence_threshold:
        return None
        
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu()
    )
    return masks[0][0][best_idx]

def create_3panel_visualization(image, bboxes, masks):
    W, H = image.width, image.height
    
    p1 = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(p1)
    for bbox in bboxes:
        draw.rectangle(bbox, outline="white", width=4)
    
    combined_mask = torch.zeros((H, W), dtype=torch.bool)
    for mask in masks:
        combined_mask = combined_mask | mask

    mask_np = combined_mask.numpy().astype(np.uint8) * 255
    p2 = Image.fromarray(mask_np, mode='L').convert("RGBA")
    
    p3_overlay = np.array(image.convert("RGBA"))
    p3_overlay[combined_mask.numpy(), 0] = 255
    p3_overlay[combined_mask.numpy(), 1] = 0
    p3_overlay[combined_mask.numpy(), 2] = 0
    p3_overlay[combined_mask.numpy(), 3] = 180
    p3 = Image.fromarray(p3_overlay).convert("RGBA")

    canvas = Image.new("RGBA", (W * 3, H + 40), (255, 255, 255, 255))
    canvas.paste(p1, (0, 40))
    canvas.paste(p2, (W, 40))
    canvas.paste(p3, (W * 2, 40))
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        
    draw_label = ImageDraw.Draw(canvas)
    draw_label.text((10, 10), "1. Florence BBoxes", fill="black", font=font)
    draw_label.text((W + 10, 10), "2. SAM Pure Masks", fill="black", font=font)
    draw_label.text((W * 2 + 10, 10), "3. SAM Final Seg", fill="black", font=font)
    
    return canvas.convert("RGB")

if __name__ == "__main__":
    print("[Run] Pipeline started.")
    input_dir = Path("data/inputs")
    output_dir = Path("data/outputs")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Run] Input dir: {input_dir}")
    print(f"[Run] Output dir: {output_dir}")
    
    image_paths = sorted([str(p) for p in input_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    print(f"[Run] Found {len(image_paths)} image(s).")
    
    if not image_paths:
        print("[Run] No images found. Exiting.")
        exit()
        
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"[Image] Processing: {filename}")
        
        try:
            original_image = Image.open(img_path).convert("RGB")
            print("[Image] Loaded.")
            
            has_glasses, vqa_text = check_has_glasses(original_image)
            if has_glasses is True:
                print("[VQA] Answer: yes")
            elif has_glasses is False:
                print("[VQA] Answer: no")
            else:
                print("[VQA] Answer: uncertain")
            print(f"[VQA] Raw: {vqa_text}")
            
            raw_frames = get_florence_bboxes(original_image, "eyeglasses, sunglasses")
            print(f"[Florence] Boxes found: {len(raw_frames)}")
            frames = filter_large_bboxes(
                raw_frames,
                image_width=original_image.width,
                image_height=original_image.height,
                max_area_ratio=MAX_BBOX_AREA_RATIO
            )
            print(f"[Florence] Boxes after size filter: {len(frames)}")

            if has_glasses is False and frames:
                print("[VQA] No, but grounding found boxes. Continuing with boxes.")
            if has_glasses is True and not frames:
                print("[VQA] Yes, but grounding found no boxes.")
            
            if frames:
                all_masks = []
                valid_boxes = []
                for idx, box in enumerate(frames, start=1):
                    print(f"[SAM] Box {idx}/{len(frames)}")
                    mask = get_sam_mask(original_image, box, confidence_threshold=0.60)
                    if mask is not None:
                        all_masks.append(mask)
                        valid_boxes.append(box)
                    else:
                        print(f"[SAM] Box {idx} rejected.")
                print(f"[SAM] Valid masks: {len(all_masks)}")
                
                if all_masks:
                    result_image = create_3panel_visualization(original_image, valid_boxes, all_masks)
                    
                    output_path = output_dir / f"viz_{filename}"
                    result_image.save(str(output_path))
                    print(f"[Save] Saved: {output_path}")
                else:
                    print("[Save] No valid masks. Nothing saved.")
            else:
                print("[Florence] No boxes found. Skipped.")
                
        except Exception as e:
            print(f"[Error] Failed on: {filename}")
            traceback.print_exc()

    print("[Run] Pipeline finished.")