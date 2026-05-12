import base64
from io import BytesIO
from typing import Any, List, Optional, Tuple, Union
import time

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.generation import GenerationConfig
import re
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64
from typing import Tuple, Optional
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")

MAX_IMAGE_SIZE = 20000
INDS_NOT_WORDS = True
FIXED_LR_SIZE = False
TOOL_CALL_PATTERN = "GET_CROPS"
CROPS_MAP = {
        '0': 'top-left',
        '1': 'top-right',
        '2': 'bottom-left',
        '3': 'bottom-right',
        '4': 'center',
        '5': 'top',
        '6': 'bottom',
        '7': 'left',
        '8': 'right',
        'all': 'all',
    }
if INDS_NOT_WORDS:
    VISION_MODEL_SHORT_SYSTEM_PROMPT = f"""You are a vision-language model that analyzes images and answers questions about them. If the image resolution is too low for accurate analysis, respond with {TOOL_CALL_PATTERN} [<crop_number>] to get a higher resolution version, the avalible <crop_numbers> and their corresponding area of the high resolution image are {CROPS_MAP} ,otherwise provide your answer."""
else:
    VISION_MODEL_SHORT_SYSTEM_PROMPT = f"""You are a vision-language model that analyzes images and answers questions about them. If the image resolution is too low for accurate analysis, respond with {TOOL_CALL_PATTERN}: [<relevant_crop>] to get a higher resolution version, otherwise provide your answer."""
    
tool_call_present_threshold = 4 # To have MORE as a valid tool call #len(TOOL_CALL_PATTERN) // 2


def parse_crop_pattern(pattern: str) -> Optional[List[str]]:
    """
    Parse a crop pattern string and return the crop name(s).
    
    Args:
        pattern: String in format "GET_CROPS:[<crop name>]" or "GET_CROPS:[<crop1>, <crop2>, ...]"
    
    Returns:
        List of crop names if valid (list of 1 if only 1 crop), None otherwise
    """    
    match = re.match(r"GET_CROPS:?\[(.+?)\]", pattern)
    
    if not match:
        return None
    
    content = match.group(1).strip()
    if INDS_NOT_WORDS:
        content = content.strip("'")        
        
    valid_crops = {
        'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center',
        'top', 'bottom', 'left', 'right', 'all'
    }
    
    # Split by comma and process each crop
    raw_crops = [c.strip().strip("'\"") for c in content.split(',')]
    raw_crops = [CROPS_MAP.get(c) for c in content]
    result = []
    for crop in raw_crops:
        if crop not in valid_crops:
            return None  # Invalid crop found
        result.append(crop)
    
    return result if result else None


def is_image_too_small(width: int, height: int, min_size: int = 100) -> bool:
    """
    Check if the image is too small to crop.
    
    Args:
        width: Image width
        height: Image height
        min_size: Minimum size threshold (default 100)
    
    Returns:
        True if image is smaller than min_size in either dimension
    """
    return width < min_size or height < min_size


def get_crop_box(crop_names: List[str], width: int, height: int) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Get the crop box coordinates for given crop names.
    
    Args:
        crop_names: List of crop names
        width: Image width
        height: Image height
    
    Returns:
        List of tuples (left, top, right, bottom) or None for 'all' or small images
    """
    # Return None if 'all' is in the list or for small images
    if 'all' in crop_names:
        return None
    
    if is_image_too_small(width, height):
        return None
    
    crop_size_w = max(100, width // 2)
    crop_size_h = max(100, height // 2)
    
    crop_boxes = {
        'top-left': (0, 0, crop_size_w, crop_size_h),
        'top-right': (width - crop_size_w, 0, width, crop_size_h),
        'bottom-left': (0, height - crop_size_h, crop_size_w, height),
        'bottom-right': (width - crop_size_w, height - crop_size_h, width, height),
        'center': (
            (width - crop_size_w) // 2,
            (height - crop_size_h) // 2,
            (width + crop_size_w) // 2,
            (height + crop_size_h) // 2
        ),
        'top': (0, 0, width, crop_size_h),
        'bottom': (0, height - crop_size_h, width, height),
        'left': (0, 0, crop_size_w, height),
        'right': (width - crop_size_w, 0, width, height),
    }
    
    result = []
    for crop_name in crop_names:
        box = crop_boxes.get(crop_name)
        if box:
            result.append(box)
    
    return result if result else None


def get_crop_from_pattern(pattern: str, img: Image.Image) -> List[Image.Image]:
    """
    Extract crop(s) from a PIL Image based on a pattern string.
    
    Args:
        pattern: String in format "GET_CROPS:[<crop name>]" or "GET_CROPS:[<crop1>, <crop2>, ...]"
                 where crop names are: top-left, top-right, bottom-left, bottom-right, center,
                 top, bottom, left, right, all
        img: PIL Image to crop
    
    Returns:
        List of cropped PIL Images in RGB mode (or list with full image if pattern is invalid, 
        'all', or image is too small)
    """
    
    def convert_mode(image: Image.Image) -> Image.Image:
        if image.mode not in ('RGB', 'RGBA', 'L', 'LA'):
            return image.convert('RGB')
        return image
    
    crop_names = parse_crop_pattern(pattern)
    width, height = img.size
    
    # Return full image for invalid pattern, 'all', or small images
    if crop_names is None or 'all' in crop_names or is_image_too_small(width, height):
        return [convert_mode(img.copy())]
    
    boxes = get_crop_box(crop_names, width, height)
    
    if boxes is None:
        return [convert_mode(img.copy())]
    
    result = []
    for box in boxes:
        crop = img.crop(box)
        result.append(convert_mode(crop))
    
    return result


def compute_pixel_ratio(hr_image_size: Tuple[int, int], pattern: str, low_res_size: Tuple[int, int] = (384, 384)) -> float:
    """
    Compute the ratio of pixels used (crops + low resolution image) compared to the full image.
    
    Args:
        hr_image_size: Tuple of (width, height) from PIL Image
        pattern: Crop pattern string in format "GET_CROPS:[<crop name>]" or "GET_CROPS:[<crop1>, <crop2>, ...]"
        low_res_size: Low resolution image size to add to each crop (default 384x384)
    
    Returns:
        Ratio of (sum_of_crop_pixels + low_res_pixels_per_crop) / full_image_pixels
        - Usually < 1 for single crops
        - Can be > 1 for 'all' (full image), small images, or multiple crops
    """
    width, height = hr_image_size
    full_image_pixels = width * height
    low_res_pixels = low_res_size[0] * low_res_size[1]
    
    crop_names = parse_crop_pattern(pattern)
    
    # For invalid pattern, 'all', or small images, use full image size
    if crop_names is None or 'all' in crop_names or is_image_too_small(width, height):
        crop_pixels = full_image_pixels
        total_pixels = crop_pixels + low_res_pixels
    else:
        boxes = get_crop_box(crop_names, width, height)
        if boxes is None:
            crop_pixels = full_image_pixels
            total_pixels = crop_pixels + low_res_pixels
        else:
            # Sum up pixels from all crops + low resolution pixels for each crop
            total_pixels = 0
            for box in boxes:
                left, top, right, bottom = box
                crop_pixels = (right - left) * (bottom - top)
                total_pixels += crop_pixels + low_res_pixels
    
    ratio = total_pixels / full_image_pixels
    if ratio > 50:
        print('Warning: Unusually high pixel ratio computed:', ratio)
    return ratio


def resize_image(path_or_img, target_size=384, fixed_size=True):
    resized_high = False
    if isinstance(path_or_img, str):
        img = Image.open(path_or_img)
    else:
        img = path_or_img
        
    width, height = img.size
    original_max_side = max(width, height)
    if original_max_side > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / original_max_side
        width, height = int(width * ratio), int(height * ratio)
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        original_max_side = MAX_IMAGE_SIZE
        resized_high = True
    if fixed_size:
        if original_max_side <= target_size:
            return img, False, width, height, None, False
        ratio = target_size / original_max_side
    else:
        ratio = 0.5
    new_width, new_height = int(width * ratio), int(height * ratio)
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img, True, width, height, img, resized_high

    
@register_model("qwen2_5_vl_awares")
class Qwen2_5_VL_AwaRes(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = True,
        min_pixels: int = 32 * 28 * 28, # 224 X 224
        max_pixels: int = 1605632, #2048 * 28 * 28, # 224 X 224 X 32
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        lora_path: Optional[str] = None,  # Path to a LoRA/PEFT adapter directory
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if use_flash_attention_2:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map).eval()

        if lora_path is not None:
            from peft import PeftModel
            eval_logger.info(f"Loading LoRA adapter from {lora_path}")
            self._model = PeftModel.from_pretrained(self._model, lora_path)
            self._model = self._model.merge_and_unload()
            self._model = self._model.to(torch.bfloat16)
            eval_logger.info("LoRA adapter merged and unloaded")

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self.processor.tokenizer.padding_side = "left"
        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.model_version = "_".join(pretrained.split("/")[-2:])

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1                    

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            sample_id = f'{task}_{split}_{doc_id[0]}'
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")


            messages = []
            tool_call_inds = []
            eff_factor = []
            inference_times = []
            low_img_sizes = []
            org_img_sizes = []
            hr_images = []
            for i, context in enumerate(contexts):
                message = [{"role": "system", "content": VISION_MODEL_SHORT_SYSTEM_PROMPT}]                
                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None                            
                    if isinstance(visual, Image.Image):  # Single image
                        org_img = visual.convert("RGB")
                        hr_images.append(org_img)
                        low_res_image, was_resized, org_width, org_height, resized_high, was_high_resized = resize_image(
                            Image.fromarray(org_img) if isinstance(org_img, np.ndarray) else org_img, fixed_size=FIXED_LR_SIZE)                        
                        low_img_sizes.append(low_res_image.size)
                        org_img_sizes.append(org_img.size)
                        message.append({"role": "user", "content": [{"type": "image", "image": low_res_image}, {"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        raise ValueError("visual as list of images is not supported yet")
                        image_content = []
                        for v in visual:
                            org_img = v.convert("RGB")
                            hr_image = v.convert("RGB")
                            low_res_image = resize_image(
                                Image.fromarray(org_img) if isinstance(org_img, np.ndarray) else org_img)                            
                            image_content.append({"type": "image", "image": low_res_image})
                        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                    else:
                        raise ValueError("visual type not supported, supported only PIL.Image.Image")
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})

                else:                    
                    raise ValueError("len(visuals) > 1 is not supported yet")
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})

                messages.append(message)
                            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,                
                padding=True,
                return_tensors="pt",
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = 0.9
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
                                    
            pad_token_id=self.processor.tokenizer.pad_token_id
            eos_token_id=self.processor.tokenizer.eos_token_id
            
            # Start timing for inference
            start_time = time.time()
            
            cont = self.model.generate(
                **inputs,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for i, answer in enumerate(answers):
                has_tool_call = TOOL_CALL_PATTERN in answer 
                if has_tool_call:                
                    if is_image_too_small(org_img_sizes[i][0], org_img_sizes[i][1]):  # Use full image if too small
                        answer = f"{TOOL_CALL_PATTERN}:[all]"
                        eff_fac = 1
                    else:
                        eff_fac = compute_pixel_ratio(hr_images[i].size, answer, low_img_sizes[i])
                    
                    eff_factor.append(eff_fac)
                    tool_call_inds.append([answer])
                    messages[i].append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
                    requested_crops = get_crop_from_pattern(answer, hr_images[i])
                    crops_content = []
                    for crop in requested_crops:
                        crops_content.append({"type": "image", "image": crop})
                    messages[i].append({"role": "user", "content": crops_content})                                    
                
                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=text,
                        images=image_inputs,
                        videos=video_inputs,                    
                        padding=True,
                        return_tensors="pt",
                    )

                    if self.device_map == "auto":
                        inputs = inputs.to("cuda")
                    else:
                        inputs = inputs.to(self.device) 
                    
                    cont = self.model.generate(
                        **inputs,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                    )

                    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                    final_answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)                                        

                else:
                    tool_call_inds.append([''])
                    eff_factor.append(0.25)
                    final_answers = answers                    

            # End timing after second inference
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            for fans, context, cur_tool_call_ind, cur_eff_factor, cur_inference_time in zip(final_answers, contexts, tool_call_inds, eff_factor, inference_times):
                # res.append([fans, cur_tool_call_ind, cur_eff_factor, cur_inference_time])
                res.append([fans])
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), fans)
                pbar.update(1)
                
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()        
            
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
