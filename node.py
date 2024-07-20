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

    CATEGORY = "Image/Image"

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
    DEFAULT_IMAGE_NAME = "{}.jpg".format(uuid.uuid4())

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