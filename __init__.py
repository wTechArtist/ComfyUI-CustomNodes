from .node import *

NODE_CLASS_MAPPINGS = {
    "Image Blending Mode Mask" : Image_Blending_Mode_Mask,
    "Load Image With Bool" : LoadImage_Bool,
    "IPAdapter Mad Scientist Weight_Type" : IPAdapter_Mad_Scientist_weight_type,
    "IPAdapter FaceID With Bool" : IPAdapter_FaceID_Bool
    
}

__all__ = ['NODE_CLASS_MAPPINGS']
