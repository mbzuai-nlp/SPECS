import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

# 如果你的项目需要一些特定路径，可以在这里追加 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
evaluation_dir = os.path.abspath(os.path.join(current_dir, "../.."))
project_root = os.path.abspath(os.path.join(evaluation_dir, ".."))
sys.path.append(project_root)

# 尝试导入 cliprec（可选）
try:
    from cliprec.models.modeling_cliprec import CLIPDecoderModel
except ImportError:
    print("警告: 无法导入 CLIPDecoderModel，cliprec 版本将不可用")
    # 定义一个空的 CLIPDecoderModel 类，以防导入失败
    class CLIPDecoderModel:
        @staticmethod
        def from_clip_decoder_pretrained(*args, **kwargs):
            raise ImportError("CLIPDecoderModel 未安装，无法使用 cliprec 版本")


# 下面是 Hugging Face Transformers 相关
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModel,            # 用于 SigLIP
    CLIPModel,            # 用于标准 CLIP
    CLIPImageProcessor,
)
from peft import PeftModel

# =========================
# 1) 定义一个包装类 Score
# =========================
class Score(nn.Module):
    """
    这个类相当于一个包装，可以选择内部用 CLIPScore 或其他 Score 实现来获取 embed。
    """

    def __init__(self, model: str, model_version: str, **kwargs):
        super().__init__()
        # 默认仍然初始化一个 CLIPScore
        self.model = CLIPScore(model, model_version, **kwargs)

    @torch.no_grad()
    def embed_text(self, text, device):
        return self.model.embed_text(text, device)

    @torch.no_grad()
    def embed_image(self, image, device):
        return self.model.embed_image(image, device)


# =========================
# 2) 定义一个 CLIPScore
# =========================
class CLIPScore(nn.Module):
    """
    原先的 CLIPScore，用于加载并使用标准 CLIP 模型来做embedding。
    如果 'version' 参数是 'pretrained', 'cliprec', 'clipdetails' 等，会在 load_clip_version 里做不同逻辑。
    """

    def __init__(self, model_name_or_path: str, version: str, **kwargs):
        super().__init__()
        # 如果用户传了 processor / tokenizer，就直接用
        if "processor" in kwargs and "tokenizer" in kwargs:
            self.processor = kwargs["processor"]
            self.tokenizer = kwargs["tokenizer"]
            self.model = None
        else:
            # 否则走自定义加载
            self.load_clip_version(model_name_or_path, version, **kwargs)

    @torch.no_grad()
    def embed_text(self, text, device):
        """
        对文本进行embedding，返回 [batch_size, hidden_dim]。
        假设 self.model 支持 get_text_features(**inputs)。
        """
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        ).to(device)
        text_features = self.model.get_text_features(**text_inputs)
        return text_features

    @torch.no_grad()
    def embed_image(self, images, device):
        """
        对图像进行embedding，返回 [batch_size, hidden_dim]。
        假设 self.model 支持 get_image_features(**inputs)。
        """
        # images 可以是 PIL 或者list
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        image_features = self.model.get_image_features(**inputs)
        return image_features

    def load_clip_version(
        self,
        model_name_or_path: str,
        version: str,
        clip_model_name_or_path: str = None,
        decoder_model_name_or_path: str = None,
    ):
        """
        根据指定 version 来加载不同的模型/processor/tokenizer:
          - pretrained: 使用 CLIPModel.from_pretrained(...)
          - cliprec: 尝试使用 cliprec decoder / PeftModel
          - clipdetails: 只是一种自定义的路径结构
          - else: 默认当做普通 CLIP
        """
        if version == "pretrained":
            # 标准 CLIP 加载方式
            self.model = CLIPModel.from_pretrained(model_name_or_path)
            self.processor = CLIPImageProcessor.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        elif version == "cliprec":
            # 尝试先走 cliprec
            try:
                base = CLIPDecoderModel.from_clip_decoder_pretrained(
                    clip_model_name_or_path,
                    decoder_model_name_or_path,
                )
                self.model = PeftModel.from_pretrained(base, model_name_or_path).clip_model
                self.processor = AutoProcessor.from_pretrained(clip_model_name_or_path)
                self.tokenizer = AutoTokenizer.from_pretrained(clip_model_name_or_path)
            except Exception as e:
                print(f"加载 cliprec 版本时出错: {e}")
                print("回退到普通 CLIP 模型")
                self.model = CLIPModel.from_pretrained(model_name_or_path)
                self.processor = CLIPImageProcessor.from_pretrained(model_name_or_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        elif version == "clipdetails":
            # 一种自定义的加载方法
            processing_path_or_name = Path(model_name_or_path).parent
            self.model = CLIPModel.from_pretrained(model_name_or_path)
            self.processor = CLIPImageProcessor.from_pretrained(processing_path_or_name)
            self.tokenizer = AutoTokenizer.from_pretrained(processing_path_or_name)

        else:
            # 默认
            print(f"未知版本 '{version}'，使用默认 CLIP 模型")
            self.model = CLIPModel.from_pretrained(model_name_or_path)
            self.processor = CLIPImageProcessor.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


# =========================
# 3) 定义一个 SigLIPScore
# =========================
class SigLIPScore(nn.Module):
    """
    原先的 SigLIPScore，用于加载并使用标准 SigLIP 模型来做embedding。
    如果 'version' 参数是 'pretrained', 'cliprec', 'clipdetails' 等，会在 load_clip_version 里做不同逻辑。
    """

    def __init__(self, model_name_or_path: str, version: str, **kwargs):
        super().__init__()
        # 如果用户传了 processor / tokenizer，就直接用
        if "processor" in kwargs and "tokenizer" in kwargs:
            self.processor = kwargs["processor"]
            self.tokenizer = kwargs["tokenizer"]
            self.model = None
        else:
            # 否则走自定义加载
            self.load_siglip_version(model_name_or_path, version, **kwargs)

    @torch.no_grad()
    def embed_text(self, text, device):
        """
        对文本进行embedding，返回 [batch_size, hidden_dim]。
        假设 self.model 支持 get_text_features(**inputs)。
        """
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        ).to(device)
        text_features = self.model.get_text_features(**text_inputs)
        return text_features

    @torch.no_grad()
    def embed_image(self, images, device):
        """
        对图像进行embedding，返回 [batch_size, hidden_dim]。
        假设 self.model 支持 get_image_features(**inputs)。
        """
        # images 可以是 PIL 或者list
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        image_features = self.model.get_image_features(**inputs)
        return image_features

    def load_siglip_version(
        self,
        model_name_or_path: str,
        version: str,
        clip_model_name_or_path: str = None,
        decoder_model_name_or_path: str = None,
    ):
        """
        根据指定 version 来加载不同的模型/processor/tokenizer:
          - pretrained: 使用 CLIPModel.from_pretrained(...)
          - cliprec: 尝试使用 cliprec decoder / PeftModel
          - clipdetails: 只是一种自定义的路径结构
          - else: 默认当做普通 CLIP
        """
        if version == "pretrained":
            # 标准 CLIP 加载方式
            self.model = AutoModel.from_pretrained(model_name_or_path)
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        elif version == "cliprec":
            # 尝试先走 cliprec
            try:
                base = CLIPDecoderModel.from_clip_decoder_pretrained(
                    clip_model_name_or_path,
                    decoder_model_name_or_path,
                )
                self.model = PeftModel.from_pretrained(base, model_name_or_path).clip_model
                self.processor = AutoProcessor.from_pretrained(clip_model_name_or_path)
                self.tokenizer = AutoTokenizer.from_pretrained(clip_model_name_or_path)
            except Exception as e:
                print(f"加载 cliprec 版本时出错: {e}")
                print("回退到普通 CLIP 模型")
                self.model = CLIPModel.from_pretrained(model_name_or_path)
                self.processor = CLIPImageProcessor.from_pretrained(model_name_or_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        elif version == "clipdetails":
            # 一种自定义的加载方法
            processing_path_or_name = Path(model_name_or_path).parent
            self.model = CLIPModel.from_pretrained(model_name_or_path)
            self.processor = CLIPImageProcessor.from_pretrained(processing_path_or_name)
            self.tokenizer = AutoTokenizer.from_pretrained(processing_path_or_name)

        else:
            # 默认
            print(f"未知版本 '{version}'，使用默认 SigLIP 模型")
            self.model = AutoModel.from_pretrained(model_name_or_path)
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)



# import torch
# from transformers import AutoProcessor, CLIPModel, AutoTokenizer, CLIPImageProcessor
# from peft import PeftModel
# import torch.nn as nn
# import os
# from pathlib import Path

# from cliprec.models.modeling_cliprec import CLIPDecoderModel

# class Score(nn.Module):
#     def __init__(self, model: str, model_version: str, **kwargs):
#         super().__init__()
#         self.model = CLIPScore(model, model_version, **kwargs)

#     @torch.no_grad
#     def embed_text(self, text, device):
#         return self.model.embed_text(text, device)

#     @torch.no_grad
#     def embed_image(self, image, device):
#         return self.model.embed_image(image, device)

# class CLIPScore(nn.Module):
#     def __init__(self, model_name_or_path: str, version: str, **kwargs):
#         super().__init__()
#         if "processor" in kwargs:
#             self.processor = kwargs["processor"]
#             self.tokenizer = kwargs["tokenizer"]
#         else:
#             self.load_clip_version(model_name_or_path, version, **kwargs)

#     @torch.no_grad
#     def embed_text(self, text, device):
#         text_inputs = self.tokenizer(
#             text,
#             padding="max_length",
#             return_tensors="pt",
#             truncation=True,
#         ).to(device)
#         text_features = self.model.get_text_features(**text_inputs)
#         return text_features

#     @torch.no_grad
#     def embed_image(self, images, device):
#         images = self.processor(images=images, return_tensors="pt").to(device)
#         image_features = self.model.get_image_features(**images)
#         return image_features

#     def load_clip_version(
#         self,
#         model_name_or_path: str,
#         version: str,
#         clip_model_name_or_path: str = None,
#         decoder_model_name_or_path: str = None,
#     ):
#         if version == "pretrained":
#             self.model = CLIPModel.from_pretrained(model_name_or_path)
#             self.processor = CLIPImageProcessor.from_pretrained(model_name_or_path)
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#         elif version == "cliprec":
#             base = CLIPDecoderModel.from_clip_decoder_pretrained(
#                 clip_model_name_or_path,
#                 decoder_model_name_or_path,
#             )
#             self.model = PeftModel.from_pretrained(base, model_name_or_path).clip_model
#             self.processor = AutoProcessor.from_pretrained(clip_model_name_or_path)
#             self.tokenizer = AutoTokenizer.from_pretrained(clip_model_name_or_path)
#         elif version == "clipdetails":
#             processing_path_or_name = Path(model_name_or_path).parent
#             self.model = CLIPModel.from_pretrained(model_name_or_path)
#             self.processor = CLIPImageProcessor.from_pretrained(processing_path_or_name)
#             self.tokenizer = AutoTokenizer.from_pretrained(processing_path_or_name)
#         elif version == "siglipdetails":
#             # 加载 Siglip2 模型
#             from transformers import SiglipModel  # 确保 transformers 版本中包含 Siglip2Model
#             processing_path_or_name = Path(model_name_or_path).parent
#             self.model = SiglipModel.from_pretrained(model_name_or_path)
#             self.processor = AutoProcessor.from_pretrained(processing_path_or_name)
#             self.tokenizer = AutoTokenizer.from_pretrained(processing_path_or_name)
#         else:
#             raise ValueError("Not implemented")
