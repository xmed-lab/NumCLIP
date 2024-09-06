import os.path as osp

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchvision.models as models
from clip import clip

from ordinalclip.utils import get_logger

from . import image_encoders
from .builder import MODELS
from .prompt_leaners import PROMPT_LEARNERS
from .prompt_leaners.plain_prompt_learner import PlainPromptLearner

import sys

logger = get_logger(__name__)


# for age estimation
bin_list_a = [0, 13, 19, 35, 65] 
bin_list_b = [0, 13, 19, 35, 65] 

bin_width_a = [13,6,16,30,36]
bin_width_b = [13,6,16,30,36]


# for image aesthetics
# bin_list_a = [0, 1, 2, 3, 4] 
# bin_list_b = [0, 1, 2, 3, 4] 

# bin_width_a = [1, 1, 1, 1, 1] 
# bin_width_b = [1, 1, 1, 1, 1] 


# for historical image dating
# bin_list_a = [0, 1, 2, 3, 4] 
# bin_list_b = [0, 1, 2, 3, 4] 

# bin_width_a = [1, 1, 1, 1, 1] 
# bin_width_b = [1, 1, 1, 1, 1] 

@MODELS.register_module()
class RegCLIPSSR(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        image_encoder_name,
        prompt_learner_cfg,
        d = 512,
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        clip_model = load_clip_to_cpu(
            text_encoder_name,
            image_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )
        clip_model.float()
        logger.info("convert `clip_model` to float32. if need fp16 model, call `clip.model.convert_weights`")

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        prompt_learner_cfg.update(dict(clip_model=clip_model))
        self.prompt_learner: PlainPromptLearner = PROMPT_LEARNERS.build(prompt_learner_cfg)
        self.psudo_sentence_tokens = self.prompt_learner.psudo_sentence_tokens
        self.logit_scale = clip_model.logit_scale

        self.embed_dims = clip_model.text_projection.shape[1]
        self.num_ranks = self.prompt_learner.num_ranks
        self.d = d

        # we first adopt CLIP-adapter based adaptation method. After experiment, we found fully finetune the image encoder could get the better performance.
        self.image_adapter = Adapter(self.d, 4)

        self.regressor = SSRModule()

    def forward(self, images):
        sentence_embeds = self.prompt_learner()
        psudo_sentence_tokens = self.psudo_sentence_tokens
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(images)
        y = self.image_adapter(image_features)
        y_ratio = 0.8
        image_features = y_ratio * y + (1 - y_ratio) * image_features


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()


        regress_age = self.regressor(logits)

        return logits, regress_age, image_features, text_features

    def forward_text_only(self):
        sentence_embeds = self.prompt_learner()
        psudo_sentence_tokens = self.psudo_sentence_tokens
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)

        return text_features

    def encode_image(self, x):
        return self.image_encoder(x)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts.type(self.dtype) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x



class SSRModule(nn.Module):
    def __init__(self, stage_num=[5, 3], d=512,
                 class_range=101, lambda_index=1., lambda_delta=1.):
        super(SSRModule, self).__init__()

        self.stage_num = stage_num
        self.lambda_index = lambda_index
        self.lambda_delta = lambda_delta
        self.class_range = class_range
        self.d = d

        self.stream1_stage2 = Adapter(self.d, 4)
        self.funsion_block_stream1_stage_2_prediction_block = nn.Linear(d, self.stage_num[1])
        self.funsion_block_stream1_stage_1_prediction_block = nn.Linear(d, self.stage_num[0])
    
        self.stream2_stage2 = Adapter(self.d, 4)
        self.funsion_block_stream2_stage_2_prediction_block = nn.Linear(d, self.stage_num[1])
        self.funsion_block_stream2_stage_1_prediction_block = nn.Linear(d, self.stage_num[0])

        self.stage2_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[1], 2 * self.stage_num[1]),
            nn.ReLU()
        )
        self.stage2_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[1], self.stage_num[1]),
            nn.ReLU()
        )
        self.stage2_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[1], self.stage_num[1]),
            nn.Tanh()
        )
        self.stage2_delta_k = nn.Sequential(
            nn.Linear(2 * self.stage_num[1], 1),
            nn.Tanh()
        )
        self.stage1_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage1_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU()
        )
        self.stage1_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage1_delta_k = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
    
    def forward(self, logits):

        prob_stage_1 = F.softmax(logits, dim=1)
        embedding_stage1_after_PB = self.stage1_FC_after_PB(logits)
        stage1_delta_k = self.stage1_delta_k(embedding_stage1_after_PB)

        stage1_regress_a = prob_stage_1[:, 0] * 0

        for index in range(self.stage_num[0]):
            width = (bin_list_a[index] / (1 + self.lambda_delta * stage1_delta_k[:, index]))
            stage1_regress_a = stage1_regress_a + prob_stage_1[:, index] * width
        stage1_regress_a = torch.unsqueeze(stage1_regress_a, 1)


        regress_age_a = stage1_regress_a
        regress_age_a = regress_age_a.squeeze(1)

        regress_age = regress_age_a

        return regress_age
    

def load_clip_to_cpu(
    text_encoder_name,
    image_encoder_name,
    root=osp.join(osp.expanduser("~/.cache/clip")),
):
    # text backbone
    if logger is not None:
        print_func = logger.info
    else:
        print_func = print

    print_func("Building CLIP model...")
    text_backbone_name = text_encoder_name
    print_func(f"Text backbone : {text_backbone_name}'s counterpart.")
    url = clip._MODELS[text_backbone_name]
    model_path = clip._download(url, root=root)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    # image backbone
    embed_dim = model.text_projection.shape[1]
    input_resolution = model.visual.input_resolution
    image_backbone_name = image_encoder_name
    print_func(f"Image backbone: {image_backbone_name}")

    if image_backbone_name != text_backbone_name:
        # remove the stochastic back-prop in vgg and alexnet
        MODEL = getattr(image_encoders, image_backbone_name, None)
        if MODEL is None:
            MODEL = getattr(models, image_backbone_name, None)
            logger.warning(f"Try PyTorch Official image model: {image_backbone_name}")
        else:
            logger.info(f"Try Custom image model: {image_backbone_name}")
        if MODEL is None:
            raise ValueError(f"Invalid torchvison model name: {image_backbone_name}")
        model.visual = MODEL(num_classes=embed_dim)
        model.visual.input_resolution = input_resolution
    else:
        print_func(f"CLIP Image encoder: {image_backbone_name}!")

    return model
