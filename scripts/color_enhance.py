import gradio as gr
import imageio.core.util
import numpy as np
import skimage.color
from PIL import Image

from modules import scripts_postprocessing
from modules.ui_components import FormRow


imageio.core.util._precision_warn = lambda *args, **kwargs: None


class ScriptPostprocessingColorEnhance(scripts_postprocessing.ScriptPostprocessing):
    name = "Color Enhance"
    order = 30000

    def ui(self):
        with FormRow():
            strength = gr.Slider(label="Color Enhance strength", minimum=0, maximum=1, step=0.01, value=0)
        return { "strength": strength }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, strength):
        if strength == 0:
            return
        
        info_bak = {} if not hasattr(pp.image, "info") else pp.image.info
        pp.image = self._color_enhance(pp.image, strength)
        pp.image.info = info_bak
        pp.info["Color Enhance"] = strength

    def _lerp(self, a: float, b: float, t: float) -> float:
        return (1 - t) * a + t * b

    def _color_enhance(self, arr, strength: float = 1) -> Image.Image:
        lch = skimage.color.lab2lch(lab=skimage.color.rgb2lab(rgb=np.array(arr, dtype=np.uint8)))
        lch[:, :, 1] *= 100/(self._lerp(100, lch[:, :, 1].max(), strength)) # Normalize chroma component
        return Image.fromarray(np.array(skimage.color.lab2rgb(lab=skimage.color.lch2lab(lch=lch)) * 255, dtype=np.uint8))