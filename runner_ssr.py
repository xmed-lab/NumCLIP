import json
from collections import defaultdict
from multiprocessing.sharedctypes import Value
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from ordinalclip.models import MODELS
from ordinalclip.models.ordinalclip import OrdinalCLIP
from ordinalclip.utils.logging import get_logger

from .optim import build_lr_scheduler, build_optimizer, build_staged_lr_param_groups
from .utils import freeze_param, load_pretrained_weights

import sys
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logger = get_logger(__name__)



class Runner(pl.LightningModule):
    def __init__(
        self,
        model_cfg,
        output_dir: str,
        optimizer_and_scheduler_cfg,
        load_weights_cfg,
        seed: int,
        loss_weights=dict(
            ce_loss=1.0,
            kl_loss=1.0,
            reg_loss=1.0
        ),
        ckpt_path="",
    ) -> None:
        super().__init__()

        self.module = MODELS.build(model_cfg)
        self.ce_loss_func = nn.CrossEntropyLoss()
        self.kl_loss_func = nn.KLDivLoss(reduction="sum")
        self.reg_loss_func = nn.L1Loss()
        self.loss_weights = loss_weights
        self.num_ranks = self.module.num_ranks
        self.register_buffer("rank_output_value_array", torch.arange(0, self.num_ranks).float(), persistent=False)
        self.output_dir = Path(output_dir)
        self._custom_logger = get_logger(__name__)

        self.load_weights(**load_weights_cfg)
        self._optimizer_and_scheduler_cfg = optimizer_and_scheduler_cfg
        self.seed = seed
        self.ckpt_path = ckpt_path

    # Model Forward
    def forward(self, images):
        return self.module(images)

    def forward_text_only(self):
        return self.forward_text_only()

    # Running Steps
    def run_step(self, batch, batch_idx):
        x, y = batch
        y = y-1
 
        logits, regression_age, *_ = self.module(x)

        losses = self.compute_losses(logits, regression_age, y)
        loss = sum([weight * losses[k] for k, weight in self.loss_weights.items()])

        metrics_exp = self.compute_per_example_metrics(logits, regression_age, y, "exp")
        metrics_max = self.compute_per_example_metrics(logits, regression_age, y, "max")

        # losses = self.compute_losses(logits, y)
        # loss = sum([weight * losses[k] for k, weight in self.loss_weights.items()])

        # metrics_exp = self.compute_per_example_metrics(logits, y, "exp")
        # metrics_max = self.compute_per_example_metrics(logits, y, "max")

        return {"loss": loss, **losses, **metrics_exp, **metrics_max}

    def training_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)

        self.logging(outputs, "train", on_step=True, on_epoch=True)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)

        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)

        return outputs

    # Epoch Eval
    def eval_epoch_end(self, outputs, run_type):
        """_summary_

        Args:
            outputs (_type_): _description_
            run_type (_type_): _description_
            moniter_key: "{val/test}_epoch_{mae/acc}_{exp/max}_metric"
        """
        stats = defaultdict(list)
        for _outputs in outputs:
            for k, v in _outputs.items():
                if self._valid_key(k):
                    stats[k].append(v)
        for k, _stats in stats.items():
            try:
                stats[k] = torch.cat(_stats).mean().item()
            except RuntimeError:
                stats[k] = torch.stack(_stats).mean().item()
            self.log(f"{run_type}_{k}", stats[k], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        stats["epoch"] = self.current_epoch
        stats["output_dir"] = str(self.output_dir)
        stats["ckpt_path"] = str(self.ckpt_path)
        with open(str(self.output_dir / f"{run_type}_stats.json"), "a") as f:
            f.write(json.dumps(stats) + "\n")

    def validation_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "test")

    def on_train_epoch_start(self) -> None:
        param_group_lrs = {pg["name"]: (pg["lr"], len(list(pg["params"]))) for pg in self.optimizers().param_groups}
        logger.info(f"check optimizer `param_groups` lr @ epoch {self.current_epoch}: {param_group_lrs}")

    def on_fit_start(self) -> None:
        pl.seed_everything(self.seed, workers=True)

    # Logging Utils
    loggings_suffix = {"metric", "loss"}

    def _valid_key(self, key: str):
        for suffix in self.loggings_suffix:
            if key.endswith(suffix):
                return True
        else:
            return False

    def logging(self, outputs: dict, run_type: str, on_step=True, on_epoch=True):
        for k, v in outputs.items():
            if self._valid_key(k):
                self.log(f"{run_type}_{k}", v.mean(), on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True)

    # Loss & Metrics
    def compute_losses(self, logits, regression_age, y):
        losses = {}

        y_label = self.num2label(y)

        # losses["ce_loss"] = self.ce_loss_func(logits, y)
        # losses["kl_loss"] = self.compute_kl_loss(logits, y)

        # losses["ce_loss"] = self.ce_loss_func(logits, y_label)
        # losses["kl_loss"] = self.compute_kl_loss(logits, y_label)

        losses["ce_loss"] = self.compute_ce_dis_loss(logits, y_label,5)
        losses["kl_loss"] = self.compute_kl_dis_loss(logits, y_label,5)

        # losses["ce_loss"] = self.compute_ce_dis_loss(logits, y,5)
        # losses["kl_loss"] = self.compute_kl_dis_loss(logits, y,5)


        # losses["ce_loss"] = torch.from_numpy(np.zeros(1)).to(logits.device)
        # losses["kl_loss"] = torch.from_numpy(np.zeros(1)).to(logits.device)
        # losses["reg_loss"] = torch.from_numpy(np.zeros(1)).to(logits.device)


        losses["reg_loss"] = self.reg_loss_func(regression_age, y)

        return losses

    def compute_kl_loss(self, logits, y):
        y_t = F.one_hot(y, self.num_ranks).t()
        y_t_row_ind = y_t.sum(-1) > 0
        num_slots = y_t_row_ind.sum()
        y_t_reduction = (y_t * 10.0).softmax(-1)
        y_t_reduction[y_t_row_ind <= 0] = 0

        logits_t = logits.t()
        kl_loss = self.kl_loss_func(F.log_softmax(logits_t, dim=-1), y_t_reduction) / num_slots
        return kl_loss


    def compute_kl_dis_loss(self, logits, y, d):
        y_t = F.one_hot(y, d).t()
        y_t_row_ind = y_t.sum(-1) > 0
        num_slots = y_t_row_ind.sum()
        y_t_reduction = (y_t * 10.0).softmax(-1)
        y_t_reduction[y_t_row_ind <= 0] = 0
        logits_t = logits.t()

        y_column = y.T
        y_column = torch.unsqueeze(y_column,1)

        ls_weight = []

        for i in range(logits_t.shape[0]):
            if y_t_row_ind[i] > 0:
                label_inv_ranks = (torch.abs(i - y_column).transpose(0,1))
                label_inv_ranks_norm = (torch.abs(i - y_column).transpose(0,1)) / torch.sum(label_inv_ranks,dim=1) * (d-1)
                label_inv_ranks_norm = torch.squeeze(label_inv_ranks_norm,0)
                label_inv_ranks_norm[y_t[i]==1] = 1.0
                ls_label_inv_ranks_norm = label_inv_ranks_norm.detach().cpu().numpy().tolist()

            else:
                label_inv_ranks_norm = torch.ones(logits_t.shape[1]).to('cuda:0')
                ls_label_inv_ranks_norm = label_inv_ranks_norm.detach().cpu().numpy().tolist()
            
            ls_weight.append(ls_label_inv_ranks_norm)

        weight = torch.Tensor(ls_weight).to('cuda:0')
        logits_weight = logits_t * weight

        kl_loss = self.kl_loss_func(F.log_softmax(logits_weight, dim=-1), y_t_reduction) / num_slots
        return kl_loss


    def compute_ce_dis_loss(self,logits,y,d):

        list_target = list(range(d))
        target = torch.Tensor(list_target).to('cuda:0')
        target = torch.unsqueeze(target,1)

        ls_weight = []

        for i in range(len(y)):
            label_inv_ranks = (torch.abs(y[i] - target).transpose(0,1))
            label_inv_ranks_norm = (torch.abs(y[i] - target).transpose(0,1)) / torch.sum(label_inv_ranks,dim=1) * (d-1)
            label_inv_ranks_norm = torch.squeeze(label_inv_ranks_norm,0)
            label_inv_ranks_norm[y[i]] = 1.0
            ls_label_inv_ranks_norm = label_inv_ranks_norm.detach().cpu().numpy().tolist()
            ls_weight.append(ls_label_inv_ranks_norm)

        weight = torch.Tensor(ls_weight).to('cuda:0')

        logits_weight = logits * weight
        loss = self.ce_loss_func(logits_weight, y)

        return loss


    def num2label(self, y):
        y_num = y.detach().cpu().numpy()
        y_label = np.zeros(len(y_num))

        for i in range(len(y_num)):
            if 0<= y_num[i] <=12:
                y_label[i] = 0
            elif 13<= y_num[i] <=18:
                y_label[i] = 1
            elif 19<= y_num[i] <=34:
                y_label[i] = 2
            elif 35<= y_num[i] <=64:
                y_label[i] = 3
            else:
                y_label[i] = 4
        y_label = torch.from_numpy(y_label).to(y.device).type(y.dtype)
        return y_label


    def compute_per_example_metrics(self, logits, regression_age, y, gather_type="exp"):
        
        probs = F.softmax(logits[:,0:5], -1)
        dtype = logits.dtype

        if gather_type == "exp":
            
            rank_output_value_array = self.rank_output_value_array.type(dtype)
            predict_y_label = torch.sum(probs * rank_output_value_array, dim=-1)

        elif gather_type == "max":
            predict_y_label = torch.argmax(probs, dim=-1).type(dtype)

        else:
            raise ValueError(f"Invalid gather_type: {gather_type}")

        y = y.type(dtype)
        y_label = self.num2label(y)

        # for age estimation
        mae = torch.abs(regression_age - y)
        acc = (torch.round(predict_y_label) == y_label).type(logits.dtype)
        return {f"mae_{gather_type}_metric": mae, f"acc_{gather_type}_metric": acc, "predict_y": regression_age}

        # for image aesthetics and historical image 
        # mae = torch.abs(predict_y_label - y)
        # acc = (torch.round(predict_y_label) == y).type(logits.dtype)
        # return {f"mae_{gather_type}_metric": mae, f"acc_{gather_type}_metric": acc, "predict_y": predict_y_label}



    # Optimizer & Scheduler
    def configure_optimizers(self):
        return self.build_optmizer_and_scheduler(**self._optimizer_and_scheduler_cfg)

    def build_optmizer_and_scheduler(
        self,
        param_dict_cfg=None,
        optimizer_cfg=None,
        lr_scheduler_cfg=None,
    ):
        param_dict_ls = self.build_param_dict(**param_dict_cfg)

        optim = build_optimizer(
            model=param_dict_ls,
            **optimizer_cfg,
        )
        sched = build_lr_scheduler(optimizer=optim, **lr_scheduler_cfg)
        return [optim], [sched]

    # Model IO
    def load_weights(
        self,
        init_model_weights=None,
        init_prompt_learner_weights=None,
        init_image_encoder_weights=None,
        init_image_adapter_weights=None,
        init_text_encoder_weights=None,
        init_regressor_weights=None,

    ):
        if init_model_weights is not None:
            self._custom_logger.info("init_model_weights")
            load_pretrained_weights(self.module, init_model_weights)
            return

        if init_prompt_learner_weights is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            load_pretrained_weights(self.module.prompt_learner, init_prompt_learner_weights)
        if init_image_encoder_weights is not None:
            self._custom_logger.info("init_image_encoder_weights")
            load_pretrained_weights(self.module.image_encoder, init_image_encoder_weights)
        if init_image_adapter_weights is not None:
            self._custom_logger.info("init_image_adapter_weights")
            load_pretrained_weights(self.module.image_adapter, init_image_adapter_weights)
        if init_text_encoder_weights is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            load_pretrained_weights(self.module.text_encoder, init_text_encoder_weights)
        if init_regressor_weights is not None:
            self._custom_logger.info("init_regressor_weights")
            load_pretrained_weights(self.module.regressor, init_regressor_weights)
        return

    def build_param_dict(
        self,
        lr_prompt_learner_context,
        lr_prompt_learner_ranks,
        lr_image_encoder,
        lr_image_adapter,
        lr_text_encoder,
        lr_logit_scale,
        staged_lr_image_encoder,
        lr_regressor
    ):
        param_dict_ls = []
        if lr_prompt_learner_context > 0 and self.module.prompt_learner is not None:
            param_dict_ls.append(
                {
                    "params": self.module.prompt_learner.context_embeds,
                    "lr": lr_prompt_learner_context,
                    "init_lr": lr_prompt_learner_context,
                    "name": "lr_prompt_learner_context",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.prompt_learner.context_embeds)")
            try:
                freeze_param(self.module.prompt_learner.context_embeds)
            except AttributeError:
                pass

        if lr_prompt_learner_ranks > 0 and self.module.prompt_learner is not None:
            param_dict_ls.append(
                {
                    "params": self.module.prompt_learner.rank_embeds,
                    "lr": lr_prompt_learner_ranks,
                    "init_lr": lr_prompt_learner_ranks,
                    "name": "lr_prompt_learner_ranks",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.prompt_learner.rank_embeds)")
            try:
                freeze_param(self.module.prompt_learner.rank_embeds)
            except AttributeError:
                pass

        if lr_image_encoder > 0 and self.module.image_encoder is not None:
            if staged_lr_image_encoder is not None:
                self._custom_logger.info("staged_lr_image_encoder activated")
                image_encoder_param_groups = build_staged_lr_param_groups(
                    model=self.module.image_encoder,
                    lr=lr_image_encoder,
                    **staged_lr_image_encoder,
                )
                param_dict_ls.extend(image_encoder_param_groups)
            else:
                param_dict_ls.append(
                    {
                        "params": self.module.image_encoder.parameters(),
                        "lr": lr_image_encoder,
                        "init_lr": lr_image_encoder,
                        "name": "image_encoder",
                    }
                )

        else:
            self._custom_logger.info("freeze_param(self.model.image_encoder)")
            freeze_param(self.module.image_encoder)

        if lr_image_adapter > 0 and self.module.image_adapter is not None:
            param_dict_ls.append(
                {
                    "params": self.module.image_adapter.parameters(),
                    "lr": lr_image_adapter,
                    "init_lr": lr_image_adapter,
                    "name": "image_adapter",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.image_adapter)")
            freeze_param(self.module.image_adapter)

        if lr_text_encoder > 0 and self.module.text_encoder is not None:
            param_dict_ls.append(
                {
                    "params": self.module.text_encoder.parameters(),
                    "lr": lr_text_encoder,
                    "init_lr": lr_text_encoder,
                    "name": "text_encoder",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.text_encoder)")
            freeze_param(self.module.text_encoder)

        if lr_logit_scale > 0 and self.module.logit_scale is not None:
            param_dict_ls.append(
                {
                    "params": self.module.logit_scale,
                    "lr": lr_logit_scale,
                    "init_lr": lr_logit_scale,
                    "name": "logit_scale",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.logit_scale)")
            freeze_param(self.module.logit_scale)
            

        if lr_regressor > 0 and self.module.regressor is not None:
            param_dict_ls.append(
                {
                    "params": self.module.regressor.parameters(),
                    "lr": lr_regressor,
                    "init_lr": lr_regressor,
                    "name": "regressor",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.regressor)")
            freeze_param(self.module.regressor)
        return param_dict_ls
