from .node import *

NODE_CLASS_MAPPINGS = {
    "Image Blending Mode Mask" : Image_Blending_Mode_Mask,
    "Load Image With Bool" : LoadImage_Bool,
}

__all__ = ['NODE_CLASS_MAPPINGS']
