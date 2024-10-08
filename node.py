from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from typing import Optional, Union, List
from urllib.request import urlopen
import comfy.diffusers_convert
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.clip_vision
import comfy.model_management
import numpy as np
import os
import subprocess
import sys
import torch
import torch

import os
import sys
import json
import hashlib
import traceback
import math
import time
import random
import logging
import uuid
import folder_paths
import nodes
# from ..ComfyUI_IPAdapter_plus import IPAdapterPlus
from PIL import Image, ImageOps, ImageSequence, ImageFile,UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo

import numpy as np
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths
import latent_preview
import node_helpers
from .libs.utils import TaggedCache, any_typ

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
settings_file = os.path.join(root_dir, 'cache_settings.json')
try:
    with open(settings_file) as f:
        cache_settings = json.load(f)
except Exception as e:
    print(e)
    cache_settings = {}
cache = TaggedCache(cache_settings)
cache_count = {}


def update_cache(k, tag, v):
    cache[k] = (tag, v)
    cnt = cache_count.get(k)
    if cnt is None:
        cnt = 0
        cache_count[k] = cnt
    else:
        cache_count[k] += 1


def cache_weak_hash(k):
    cnt = cache_count.get(k)
    if cnt is None:
        cnt = 0

    return k, cnt

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
# 导入 IPAdapterPlus 模块
from custom_nodes.ComfyUI_IPAdapter_plus import IPAdapterPlus


class cstr(str):
    class color:
        END = '\33[0m'
        BOLD = '\33[1m'
        ITALIC = '\33[3m'
        UNDERLINE = '\33[4m'
        BLINK = '\33[5m'
        BLINK2 = '\33[6m'
        SELECTED = '\33[7m'

        BLACK = '\33[30m'
        RED = '\33[31m'
        GREEN = '\33[32m'
        YELLOW = '\33[33m'
        BLUE = '\33[34m'
        VIOLET = '\33[35m'
        BEIGE = '\33[36m'
        WHITE = '\33[37m'

        BLACKBG = '\33[40m'
        REDBG = '\33[41m'
        GREENBG = '\33[42m'
        YELLOWBG = '\33[43m'
        BLUEBG = '\33[44m'
        VIOLETBG = '\33[45m'
        BEIGEBG = '\33[46m'
        WHITEBG = '\33[47m'

        GREY = '\33[90m'
        LIGHTRED = '\33[91m'
        LIGHTGREEN = '\33[92m'
        LIGHTYELLOW = '\33[93m'
        LIGHTBLUE = '\33[94m'
        LIGHTVIOLET = '\33[95m'
        LIGHTBEIGE = '\33[96m'
        LIGHTWHITE = '\33[97m'

        GREYBG = '\33[100m'
        LIGHTREDBG = '\33[101m'
        LIGHTGREENBG = '\33[102m'
        LIGHTYELLOWBG = '\33[103m'
        LIGHTBLUEBG = '\33[104m'
        LIGHTVIOLETBG = '\33[105m'
        LIGHTBEIGEBG = '\33[106m'
        LIGHTWHITEBG = '\33[107m'

        @staticmethod
        def add_code(name, code):
            if not hasattr(cstr.color, name.upper()):
                setattr(cstr.color, name.upper(), code)
            else:
                raise ValueError(f"'cstr' object already contains a code with the name '{name}'.")

    def __new__(cls, text):
        return super().__new__(cls, text)

    def __getattr__(self, attr):
        if attr.lower().startswith("_cstr"):
            code = getattr(self.color, attr.upper().lstrip("_cstr"))
            modified_text = self.replace(f"__{attr[1:]}__", f"{code}")
            return cstr(modified_text)
        elif attr.upper() in dir(self.color):
            code = getattr(self.color, attr.upper())
            modified_text = f"{code}{self}{self.color.END}"
            return cstr(modified_text)
        elif attr.lower() in dir(cstr):
            return getattr(cstr, attr.lower())
        else:
            raise AttributeError(f"'cstr' object has no attribute '{attr}'")

    def print(self, **kwargs):
        print(self, **kwargs)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Freeze PIP modules
def packages(versions=False):
    import sys
    import subprocess
    return [( r.decode().split('==')[0] if not versions else r.decode() ) for r in subprocess.check_output([sys.executable, '-s', '-m', 'pip', 'freeze']).split()]
def install_package(package, uninstall_first: Union[List[str], str] = None):
    if os.getenv("WAS_BLOCK_AUTO_INSTALL", 'False').lower() in ('true', '1', 't'):
        cstr(f"Preventing package install of '{package}' due to WAS_BLOCK_INSTALL env").msg.print()
    else:
        if uninstall_first is None:
            return

        if isinstance(uninstall_first, str):
            uninstall_first = [uninstall_first]

        cstr(f"Uninstalling {', '.join(uninstall_first)}..")
        subprocess.check_call([sys.executable, '-s', '-m', 'pip', 'uninstall', *uninstall_first])
        cstr("Installing package...").msg.print()
        subprocess.check_call([sys.executable, '-s', '-m', 'pip', '-q', 'install', package])

class Image_Blending_Mode_Mask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mode": ([
                    "add",
                    "color",
                    "color_burn",
                    "color_dodge",
                    "darken",
                    "difference",
                    "exclusion",
                    "hard_light",
                    "hue",
                    "lighten",
                    "multiply",
                    "overlay",
                    "screen",
                    "soft_light"
                ],),
                "blend_percentage": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask": ("MASK",),  # 添加遮罩输入
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_blending_mode_mask"

    CATEGORY = "WWL"

    def image_blending_mode_mask(self, image_a, image_b, mode='add', blend_percentage=1.0, mask=None):

        # Install Pilgram
        if 'pilgram' not in packages():
            install_package("pilgram")

        # Import Pilgram module
        import pilgram

        # Convert images to PIL
        img_a = tensor2pil(image_a)
        img_b = tensor2pil(image_b)

        # Ensure images are the same size
        if img_a.size != img_b.size:
            raise ValueError("Input images must have the same dimensions")

        # Apply blending mode
        blending_modes = {
            "color": pilgram.css.blending.color,
            "color_burn": pilgram.css.blending.color_burn,
            "color_dodge": pilgram.css.blending.color_dodge,
            "darken": pilgram.css.blending.darken,
            "difference": pilgram.css.blending.difference,
            "exclusion": pilgram.css.blending.exclusion,
            "hard_light": pilgram.css.blending.hard_light,
            "hue": pilgram.css.blending.hue,
            "lighten": pilgram.css.blending.lighten,
            "multiply": pilgram.css.blending.multiply,
            "add": pilgram.css.blending.normal,
            "overlay": pilgram.css.blending.overlay,
            "screen": pilgram.css.blending.screen,
            "soft_light": pilgram.css.blending.soft_light
        }

        out_image = blending_modes.get(mode, pilgram.css.blending.normal)(img_a, img_b)

        out_image = out_image.convert("RGB")

        # Apply mask if provided
        if mask is not None:
            mask = ImageOps.invert(tensor2pil(mask).convert('L'))
            out_image = Image.composite(img_a, out_image, mask.resize(img_a.size))

        # Blend image based on blend percentage
        blend_mask = Image.new(mode="L", size=img_a.size, color=(round(blend_percentage * 255)))
        blend_mask = ImageOps.invert(blend_mask)
        out_image = Image.composite(img_a, out_image, blend_mask)

        return (pil2tensor(out_image), )


class LoadImage_Bool:
    DEFAULT_IMAGE_NAME = "custom_nodes_empty.jpg"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        if not files:
            default_image_path = os.path.join(input_dir, LoadImage_Bool.DEFAULT_IMAGE_NAME)
            LoadImage_Bool._generate_default_image(default_image_path)
            files = [LoadImage_Bool.DEFAULT_IMAGE_NAME]
        return {"required": {"image": (sorted(files), {"image_upload": True})}}

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    FUNCTION = "load_image"
    CATEGORY = "WWL"

    def load_image(self, image=DEFAULT_IMAGE_NAME):
        if not image or image == LoadImage_Bool.DEFAULT_IMAGE_NAME:
            return self._default_response()

        try:
            image_path = folder_paths.get_annotated_filepath(image)
            img = node_helpers.pillow(Image.open, image_path)

            output_images = []
            output_masks = []
            w, h = None, None

            excluded_formats = ['MPO']

            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)

                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB")

                if len(output_images) == 0:
                    w = image.size[0]
                    h = image.size[1]

                if image.size[0] != w or image.size[1] != h:
                    continue

                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                if 'A' in i.getbands():
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
                output_images.append(image)
                output_masks.append(mask.unsqueeze(0))

            if len(output_images) > 1 and img.format not in excluded_formats:
                output_image = torch.cat(output_images, dim=0)
                output_mask = torch.cat(output_masks, dim=0)
            else:
                output_image = output_images[0]
                output_mask = output_masks[0]

            return (output_image, output_mask, True)

        except Exception as e:
            return self._default_response()

    def _default_response(self):
        black_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
        black_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
        return (black_image, black_mask, False)

    @classmethod
    def IS_CHANGED(s, image=DEFAULT_IMAGE_NAME):
        if not image or image == LoadImage_Bool.DEFAULT_IMAGE_NAME:
            return False
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image=DEFAULT_IMAGE_NAME):
        if not image or image == LoadImage_Bool.DEFAULT_IMAGE_NAME:
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

    @staticmethod
    def _generate_default_image(filepath):
        # Create a default black image (64x64)
        black_image = Image.new("RGB", (1, 1), (0, 0, 0))
        black_image.save(filepath)


class IPAdapter_Mad_Scientist_weight_type:
    CATEGORY_KEYS = ['0layer', '1layer', '2layer', '3layer', '4layer', '5layer', '6layer', '7layer', '8layer', '9layer', '10layer', '11layer']
    OPTIONS = {str(i): str(i) for i in range(0, 12)}

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **cls.get_input_types_from_keys(cls.CATEGORY_KEYS),
                "Random": (["Yes", "No"], {"default": "No"}),
                "seed": ("INT", {"default": 0, "min": -1125899906842624, "max": 1125899906842624}),
            }
        }

    @staticmethod
    def get_input_types_from_keys(keys):
        input_types = {}
        for i, key in enumerate(keys):
            # Hide the super key by not including it in the input_types dictionary
            input_types[f"{key} Weight"] = ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1})
        return input_types

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("layer_weights",)
    FUNCTION = "generate_prompt"
    CATEGORY = "WWL"

    def generate_prompt(self, **kwargs):
        prompt_parts = {str(i): 0.0 for i in range(0, 12)}
        for i, key in enumerate(self.CATEGORY_KEYS):
            # Since the super key is hidden, we use the index to find the corresponding weight
            weight_key = f"{key} Weight"
            if weight_key in kwargs and kwargs[weight_key] is not None:
                weight = kwargs[weight_key]
                prompt_parts[str(i)] = weight
        
        if kwargs.get("Random") == "Yes":
            for key in self.CATEGORY_KEYS:
                options = list(self.OPTIONS.keys())
                random_choice = random.choice(options)
                weight_key = f"{key} Weight"
                if prompt_parts[str(int(random_choice))] == 0.0:
                    weight = random.uniform(-2.0, 2.0)
                    prompt_parts[random_choice] = weight

        layer_weights = ','.join(f"{k}:{int(v)}" if v in {-2.0, -1.0, 0.0, 1.0, 2.0} else f"{k}:{v:.1f}" for k, v in prompt_parts.items())
        return (layer_weights,)
    
WEIGHT_TYPES = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer', 'composition', 'strong style transfer', 'style and composition', 'style transfer precise', 'composition precise']

class IPAdapter_FaceID_Bool():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_faceidv2": ("FLOAT", { "default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "insightface": ("INSIGHTFACE",),
            }
        }

    CATEGORY = "ipadapter/faceid"
    RETURN_TYPES = ("MODEL","IMAGE","BOOLEAN")
    RETURN_NAMES = ("MODEL", "face_image", "bool",)
    FUNCTION = "apply_ipadapter_bool"
    CATEGORY = "WWL"

    def apply_ipadapter_bool(self, model, ipadapter, start_at=0.0, end_at=1.0, weight=1.0, weight_style=1.0, weight_composition=1.0, expand_style=False, weight_type="linear", combine_embeds="concat", weight_faceidv2=None, image=None, image_style=None, image_composition=None, image_negative=None, clip_vision=None, attn_mask=None, insightface=None, embeds_scaling='V only', layer_weights=None, ipadapter_params=None, encode_batch_size=0, style_boost=None, composition_boost=None, enhance_tiles=1, enhance_ratio=1.0):
        try:
            # 实例化 IPAdapterFaceID 类
            ip_adapter_face_id = IPAdapterPlus.IPAdapterFaceID()
            # 调用 apply_ipadapter 方法
            work_model, face_image = ip_adapter_face_id.apply_ipadapter(model, ipadapter, start_at, end_at, weight, weight_style, weight_composition, expand_style, weight_type, combine_embeds, weight_faceidv2, image, image_style, image_composition, image_negative, clip_vision, attn_mask, insightface, embeds_scaling, layer_weights, ipadapter_params, encode_batch_size, style_boost, composition_boost, enhance_tiles, enhance_ratio)
            return (work_model, face_image, True)
        except Exception as e:
            print(f"Error: {e}")
            return (model, image, False)
        
class LoraLoaderShared(nodes.LoraLoader):
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        file_list = folder_paths.get_filename_list("loras")
        file_list.insert(0, "None")
        return {
            "required": { 
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (file_list, {"tooltip": "The name of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
                "key_opt": ("STRING", {"multiline": False, "placeholder": "If empty, use 'lora_name' as the key."}),
            },
            "optional": {
                "mode": (['Auto', 'Override Cache', 'Read Only'],),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "cache key")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Backend"

    def doit(self, model, clip, lora_name, strength_model, strength_clip, key_opt, mode='Auto'):
        if mode == 'Read Only':
            if key_opt.strip() == '':
                raise Exception("[LoraLoaderShared] key_opt cannot be omit if mode is 'Read Only'")
            key = key_opt.strip()
        elif key_opt.strip() == '':
            key = lora_name
        else:
            key = key_opt.strip()

        if strength_model == 0 and strength_clip == 0:
            return (model, clip, key)

        if lora_name == "None":
            return (model, clip, key)
        
        if key not in cache or mode == 'Override Cache':
            res = self.load_lora(model, clip, lora_name, strength_model, strength_clip)
            update_cache(key, "lora", (False, res))
            cache_kind = 'lora'
            print(f"[Inspire Pack] LoraLoaderShared: Lora '{lora_name}' is cached to '{key}'.")
        else:
            cache_kind, (_, res) = cache[key]
            print(f"[Inspire Pack] LoraLoaderShared: Cached lora '{key}' is loaded. (Loading skip)")

        if cache_kind == 'lora':
            model, clip = res
        else:
            raise Exception(f"[LoraLoaderShared] Unexpected cache_kind '{cache_kind}'")

        return model, clip, key

    @staticmethod
    def IS_CHANGED(model, clip, lora_name, strength_model, strength_clip, key_opt, mode='Auto'):
        if mode == 'Read Only':
            if key_opt.strip() == '':
                raise Exception("[LoraLoaderShared] key_opt cannot be omit if mode is 'Read Only'")
            key = key_opt.strip()
        elif key_opt.strip() == '':
            key = lora_name
        else:
            key = key_opt.strip()

        if mode == 'Read Only':
            return (None, cache_weak_hash(key))
        elif mode == 'Override Cache':
            return (lora_name, key)

        return (None, cache_weak_hash(key))

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)