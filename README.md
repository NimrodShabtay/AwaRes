# AwaRes: Look Where It Matters

**High-Resolution Crops Retrieval for Efficient VLMs**

[Paper](https://arxiv.org/abs/2603.16932) | [Project Page](https://nimrodshabtay.github.io/AwaRes/) | [Model](https://huggingface.co/Kimhi/AWARES-Qwen2.5-VL-7B) | [Dataset](https://huggingface.co/datasets/NimrodShabtay1986/AwaRes)

AwaRes is a spatial-on-demand VLM inference framework that processes low-resolution images first and selectively retrieves high-resolution crops via tool-calling. It achieves comparable performance to Qwen2.5-VL-7B across six benchmarks while using only ~36% of the visual tokens, offering a 4.4x speedup over dynamic high-resolution methods.

## Quick Start

```python
import re
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

CROPS_MAP = {
    '0': 'top-left', '1': 'top-right', '2': 'bottom-left', '3': 'bottom-right',
    '4': 'center', '5': 'top', '6': 'bottom', '7': 'left', '8': 'right',
}

SYSTEM_PROMPT = (
    "You are a vision-language model that analyzes images and answers questions about them. "
    "If the image resolution is too low for accurate analysis, respond with GET_CROPS: "
    "followed by a list of crop numbers in square brackets (e.g., GET_CROPS: ['3'] or "
    f"GET_CROPS: ['0', '5']), where the available crop numbers and their corresponding "
    f"areas are {CROPS_MAP}, otherwise provide your answer."
)

model_name = "Kimhi/AWARES-Qwen2.5-VL-7B"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)


def extract_crop_indices(text):
    """Parse GET_CROPS: ['0', '5'] from model output."""
    match = re.search(r"GET_CROPS\s*:\s*\[([^\]]*)\]", text)
    if not match:
        return None
    raw = match.group(1)
    indices = re.findall(r"['\"]?(\d)['\"]?", raw)
    return indices


def get_crop(image: Image.Image, crop_id: str) -> Image.Image:
    """Extract a crop region from a high-res image given a crop index."""
    w, h = image.size
    hw, hh = w // 2, h // 2
    crop_boxes = {
        '0': (0, 0, hw, hh),           # top-left
        '1': (hw, 0, w, hh),           # top-right
        '2': (0, hh, hw, h),           # bottom-left
        '3': (hw, hh, w, h),           # bottom-right
        '4': (w//4, h//4, 3*w//4, 3*h//4),  # center
        '5': (0, 0, w, hh),            # top half
        '6': (0, hh, w, h),            # bottom half
        '7': (0, 0, hw, h),            # left half
        '8': (hw, 0, w, h),            # right half
    }
    return image.crop(crop_boxes[crop_id])


def generate(messages):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


# Load your high-res image (the model sees a low-res version first)
high_res_image = Image.open("your_image.jpg")
low_res_image = high_res_image.copy()
low_res_image.thumbnail((512, 512))

question = "What text is written on the small sign in the bottom-right corner?"

# Turn 1: ask with low-res image
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": [
        {"type": "image", "image": low_res_image},
        {"type": "text", "text": question},
    ]},
]

response = generate(messages)
print(f"Turn 1: {response}")

# Check if the model requested crops
crop_indices = extract_crop_indices(response)
if crop_indices:
    print(f"Model requested crops: {crop_indices}")

    # Prepare crop images
    crop_images = [get_crop(high_res_image, idx) for idx in crop_indices]
    crop_content = [{"type": "image", "image": img} for img in crop_images]

    # Turn 2: provide crops and get final answer
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": crop_content})

    final_response = generate(messages)
    print(f"Turn 2 (final answer): {final_response}")
else:
    print("Model answered directly (no crops needed)")
```

## Evaluation with lmms-eval

We provide a custom [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) model type (`qwen2_5_vl_awares`) that handles the full AWARES multi-turn pipeline automatically during evaluation — including low-res input, `GET_CROPS` parsing, crop extraction, and second-turn generation with KV-cache reuse.

### Setup

```bash
git clone https://github.ibm.com/ai-models-architectures/lmms-eval.git
cd lmms-eval
pip install -e .
pip install qwen-vl-utils
```

### Single benchmark

```bash
accelerate launch --num_processes=8 --main_process_port=12346 \
    -m lmms_eval \
    --model qwen2_5_vl_awares \
    --model_args pretrained=Kimhi/AWARES-Qwen2.5-VL-7B \
    --tasks chartqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix awares \
    --output_path ./logs/awares_eval/
```

### Multiple benchmarks

```bash
accelerate launch --num_processes=8 --main_process_port=12346 \
    -m lmms_eval \
    --model qwen2_5_vl_awares \
    --model_args pretrained=Kimhi/AWARES-Qwen2.5-VL-7B \
    --tasks chartqa,ocrbench,realworldqa,mmmu_val,infovqa_val,docvqa_val \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix awares \
    --output_path ./logs/awares_eval/
```

### Single GPU

```bash
python -m lmms_eval \
    --model qwen2_5_vl_awares \
    --model_args pretrained=Kimhi/AWARES-Qwen2.5-VL-7B \
    --tasks chartqa \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/awares_eval/
```

## Citation

```bibtex
@article{shabtay2026look,
  title={Look Where It Matters: High-Resolution Crops Retrieval for Efficient VLMs},
  author={Shabtay, Nimrod and Kimhi, Moshe and Spector, Artem and Haray, Sivan and Rivlin, Ehud and Baskin, Chaim and Giryes, Raja and Schwartz, Eli},
  journal={arXiv preprint arXiv:2603.16932},
  year={2026}
}
```
