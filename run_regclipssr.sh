# ----------------------------------clip regression ssrnet evaluation for morph dataset (few shot training)-------------------------------


# CUDA_VISIBLE_DEVICES='7' python scripts/run.py \
# --config configs/default_test.yaml \
# --config configs/base_cfgs/data_cfg/datasets/morph/morph.yaml \
# --config configs/base_cfgs/runner_cfg/model/image_encoder/clip-vitb16.yaml \
# --config configs/base_cfgs/runner_cfg/model/text_encoder/clip-vitb16-cntprt.yaml \
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_context/init-context-morph_class.yaml \
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/num_ranks_5.yaml \
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/plain-prompt-learner.yaml \
# --config configs/base_cfgs/runner_cfg/optim_sched/prompt_learner/tune-rank.yaml \
# --config configs/base_cfgs/runner_cfg/optim_sched/image_encoder/tune-image.yaml \
# --config configs/base_cfgs/runner_cfg/model/regclipssr.yaml \
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_rank/init-rank-morph_class.yaml
# --config configs/base_cfgs/runner_cfg/init_weight.yaml
# --config configs/base_cfgs/data_cfg/few_shots/num-shots-1.yaml
# --config configs/base_cfgs/data_cfg/label_distribution_shift/num_topk_scaled_class/num-topk-scaled-class-40.yaml \
# --config configs/base_cfgs/data_cfg/label_distribution_shift/scale_factor/scale-factor-01.yaml







# ----------------------------------clip regression ssrnet evaluation for aesthetics dataset(few shot training)-------------------------------
# CUDA_VISIBLE_DEVICES='7' python scripts/run.py \
# --config configs/default_aesthetics.yaml \
# --config configs/base_cfgs/data_cfg/datasets/aesthetics/aesthetics.yaml \
# --config configs/base_cfgs/runner_cfg/model/image_encoder/clip-vitb16.yaml \
# --config configs/base_cfgs/runner_cfg/model/text_encoder/clip-vitb16-cntprt.yaml \
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/num_ranks_5.yaml \
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/plain-prompt-learner.yaml \
# --config configs/base_cfgs/runner_cfg/optim_sched/prompt_learner/tune-ctx-rank.yaml \
# --config configs/base_cfgs/runner_cfg/optim_sched/image_encoder/tune-image.yaml \
# --config configs/base_cfgs/runner_cfg/model/regclipssr.yaml \
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_context/init-context-aesthetics_class.yaml \
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_rank/init-rank-aesthetics_class.yaml

# --config configs/base_cfgs/runner_cfg/init_weight.yaml
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_rank/init-rank-aesthetics.yaml
# --config configs/base_cfgs/runner_cfg/init_weight.yaml
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_rank/init-rank-aesthetics_class.yaml
# --config configs/base_cfgs/data_cfg/few_shots/num-shots-1.yaml






# ----------------------------------clip regression ssrnet evaluation for historical dataset(few shot training)-------------------------------
# CUDA_VISIBLE_DEVICES='3' python scripts/run.py \
# --config configs/default_historical.yaml \
# --config configs/base_cfgs/data_cfg/datasets/historical/historical.yaml \
# --config configs/base_cfgs/runner_cfg/model/image_encoder/clip-vitb16.yaml \
# --config configs/base_cfgs/runner_cfg/model/text_encoder/clip-vitb16-cntprt.yaml \
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_context/init-context-historical_class.yaml \
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/num_ranks_5.yaml \
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/plain-prompt-learner.yaml \
# --config configs/base_cfgs/runner_cfg/optim_sched/prompt_learner/tune-rank.yaml \
# --config configs/base_cfgs/runner_cfg/optim_sched/image_encoder/tune-image.yaml \
# --config configs/base_cfgs/runner_cfg/model/regclipssr.yaml \
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_rank/init-rank-historical_class.yaml
# --config configs/base_cfgs/runner_cfg/model/prompt_learner/init_rank/init-rank-historical.yaml
# --config configs/base_cfgs/runner_cfg/init_weight.yaml
# --config configs/base_cfgs/data_cfg/few_shots/num-shots-1.yaml