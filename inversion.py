# Bootstrapped from:
# https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

import argparse
import hashlib
import inspect
import itertools
import math
import os
import random
import re
from pathlib import Path
from typing import Optional, List, Literal

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import fire

from lora_diffusion import (
    PivotalTuningDatasetCapation,
    extract_lora_ups_down,
    inject_trainable_lora,
    inspect_lora,
    save_lora_weight,
    save_all,
    evaluate_pipe,
    prepare_clip_model_sets,
)

from transformers.optimization import get_scheduler

import wandb


def get_models(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path,
    revision,
    placeholder_tokens: List[str],
    initializer_tokens: List[str],
    device="cuda:0",
):

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )

    placeholder_token_ids = []

    for token, init_tok in zip(placeholder_tokens, initializer_tokens):
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

        placeholder_token_ids.append(placeholder_token_id)

        # Load models and create wrapper for stable diffusion

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        if init_tok.startswith("<rand"):
            # <rand-"sigma">, e.g. <rand-0.5>
            sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

            token_embeds[placeholder_token_id] = (
                torch.randn_like(token_embeds[0]) * sigma_val
            )
            print(
                f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[placeholder_token_id].mean().item():.3f} +- {token_embeds[placeholder_token_id].std().item():.3f}"
            )
        elif init_tok == "<zero>":
            token_embeds[placeholder_token_id] = torch.zeros_like(token_embeds[0])
        else:
            token_ids = tokenizer.encode(init_tok, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id = token_ids[0]
            token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_name_or_path or pretrained_model_name_or_path,
        subfolder=None if pretrained_vae_name_or_path else "vae",
        revision=None if pretrained_vae_name_or_path else revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
    )

    return (
        text_encoder.to(device),
        vae.to(device),
        unet.to(device),
        tokenizer,
        placeholder_token_ids,
    )


def text2img_dataloader(train_dataset, train_batch_size, tokenizer, vae, text_encoder):
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if examples[0].get("class_prompt_ids", None) is not None:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        if examples[0].get("mask", None) is not None:
            batch["mask"] = torch.stack([example["mask"] for example in examples])

        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_dataloader


def loss_step(
    batch,
    unet,
    vae,
    text_encoder,
    scheduler,
    t_mutliplier=1.0,
    mixed_precision=False,
):
    if mixed_precision:
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    latents = vae.encode(
        batch["pixel_values"].to(dtype=weight_dtype).to(unet.device)
    ).latent_dist.sample()
    latents = latents * 0.18215

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]

    timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps * t_mutliplier),
        (bsz,),
        device=latents.device,
    )
    timesteps = timesteps.long()

    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    if mixed_precision:
        with torch.cuda.amp.autocast():

            encoder_hidden_states = text_encoder(
                batch["input_ids"].to(text_encoder.device)
            )[0]

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    else:

        encoder_hidden_states = text_encoder(
            batch["input_ids"].to(text_encoder.device)
        )[0]

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")

    if batch.get("mask", None) is not None:

        mask = (
            batch["mask"]
            .to(model_pred.device)
            .reshape(
                model_pred.shape[0], 1, model_pred.shape[2] * 8, model_pred.shape[3] * 8
            )
        )
        # resize to match model_pred
        mask = (
            F.interpolate(
                mask.float(),
                size=model_pred.shape[-2:],
                mode="nearest",
            )
            + 0.05
        )

        mask = mask / mask.mean()

        model_pred = model_pred * mask

        target = target * mask

    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return loss


def train_inversion(
    unet,
    vae,
    text_encoder,
    tokenizer,
    dataloader,
    num_steps,
    accum_iter,
    scheduler,
    index_no_updates,
    optimizer,
    lr_scheduler,
    save_steps,
    placeholder_token_ids,
    placeholder_tokens,
    save_path,
    test_image_path,
    log_wandb,
    mixed_precision,
    clip_norm=False,
):

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    # Original Emb for TI
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    if mixed_precision:
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    if log_wandb:
        preped_clip = prepare_clip_model_sets()

    loss_sum = 0.0

    index_updates = ~index_no_updates

    for epoch in range(math.ceil(num_steps / len(dataloader))):
        unet.eval()
        text_encoder.train()
        for batch in dataloader:

            lr_scheduler.step()

            with torch.set_grad_enabled(True):
                loss = (
                    loss_step(
                        batch,
                        unet,
                        vae,
                        text_encoder,
                        scheduler,
                        mixed_precision=mixed_precision,
                    )
                    / accum_iter
                )

                loss.backward()
                loss_sum += loss.detach().item()

                if (global_step + 1) % accum_iter == 1:
                    optimizer.step()
                    optimizer.zero_grad()

                    with torch.no_grad():

                        # normalize embeddings
                        if clip_norm:
                            pre_norm = (
                                text_encoder.get_input_embeddings()
                                .weight[index_updates, :]
                                .norm(dim=-1, keepdim=True)
                            )

                            lambda_ = 0.1
                            text_encoder.get_input_embeddings().weight[
                                index_updates
                            ] = F.normalize(
                                text_encoder.get_input_embeddings().weight[
                                    index_updates, :
                                ],
                                dim=-1,
                            ) * (
                                pre_norm + lambda_ * (0.39 - pre_norm)
                            )
                            print(pre_norm)

                        current_norm = (
                            text_encoder.get_input_embeddings()
                            .weight[index_updates, :]
                            .norm(dim=-1)
                        )

                        text_encoder.get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]

                        print(f"Current Norm : {current_norm}")

                global_step += 1
                progress_bar.update(1)

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

            if global_step % save_steps == 0:
                save_all(
                    unet=unet,
                    text_encoder=text_encoder,
                    placeholder_token_ids=placeholder_token_ids,
                    placeholder_tokens=placeholder_tokens,
                    save_path=os.path.join(save_path, f"step_inv_{global_step}.pt"),
                    save_lora=False,
                )
                if log_wandb:
                    with torch.no_grad():
                        pipe = StableDiffusionPipeline(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=unet,
                            scheduler=scheduler,
                            safety_checker=None,
                            feature_extractor=None,
                        )

                        # open all images in test_image_path
                        images = []
                        for file in os.listdir(test_image_path):
                            if file.endswith(".png") or file.endswith(".jpg"):
                                images.append(
                                    Image.open(os.path.join(test_image_path, file))
                                )

                        wandb.log({"loss": loss_sum / save_steps})
                        loss_sum = 0.0
                        wandb.log(
                            evaluate_pipe(
                                pipe,
                                target_images=images,
                                class_token="person",
                                learnt_token="".join(placeholder_tokens),
                                n_test=1,
                                n_step=30,
                                clip_model_sets=preped_clip,
                            )
                        )

            if global_step >= num_steps:
                return


def train(
    instance_data_dir: str,
    pretrained_model_name_or_path: str,
    output_dir: str,
    pretrained_vae_name_or_path: str = None,
    revision: Optional[str] = None,
    class_data_dir: Optional[str] = None,
    stochastic_attribute: Optional[str] = None,
    use_template: Literal[None, "object", "style"] = None,
    placeholder_tokens: str = "<s>",
    placeholder_token_at_data: Optional[str] = None,
    initializer_tokens: str = "dog",
    class_prompt: Optional[str] = None,
    with_prior_preservation: bool = False,
    seed: int = 42,
    resolution: int = 512,
    color_jitter: bool = True,
    train_batch_size: int = 1,
    max_train_steps_ti: int = 2000,
    save_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    learning_rate_unet: float = 1e-5,
    learning_rate_text: float = 1e-5,
    learning_rate_ti: float = 5e-4,
    use_face_segmentation_condition: bool = False,
    scale_lr: bool = False,
    weight_decay_ti: float = 0.01,
    device="cuda:0",
    extra_args: Optional[dict] = None,
    project_name: str = "new_project",
    log_wandb: bool = False,
):

    print(log_wandb)
    if log_wandb:
        wandb.init(
            project=project_name,
            entity="simoryu",
            name=f"steps_{max_train_steps_ti}_lr_{learning_rate_ti}_",
            reinit=True,
            config={
                "steps": max_train_steps_ti,
                "lr": learning_rate_ti,
                "weight_decay_ti": weight_decay_ti,
                **(extra_args if extra_args is not None else {}),
            },
        )

    torch.manual_seed(seed)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    # print(placeholder_tokens, initializer_tokens)
    placeholder_tokens = placeholder_tokens.split("|")
    initializer_tokens = initializer_tokens.split("|")

    if placeholder_token_at_data is not None:
        tok, pat = placeholder_token_at_data.split("|")
        token_map = {tok: pat}

    else:
        token_map = None
    print("Placeholder Tokens", placeholder_tokens)
    print("Initializer Tokens", initializer_tokens)

    # get the models
    text_encoder, vae, unet, tokenizer, placeholder_token_ids = get_models(
        pretrained_model_name_or_path,
        pretrained_vae_name_or_path,
        revision,
        placeholder_tokens,
        initializer_tokens,
        device=device,
    )

    noise_scheduler = DDPMScheduler.from_config(
        pretrained_model_name_or_path, subfolder="scheduler"
    )

    if gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.gradient_checkpointing_enable()

    if scale_lr:
        unet_lr = learning_rate_unet * gradient_accumulation_steps * train_batch_size
        text_encoder_lr = (
            learning_rate_text * gradient_accumulation_steps * train_batch_size
        )
        ti_lr = learning_rate_ti * gradient_accumulation_steps * train_batch_size
    else:
        unet_lr = learning_rate_unet
        text_encoder_lr = learning_rate_text
        ti_lr = learning_rate_ti

    train_dataset = PivotalTuningDatasetCapation(
        instance_data_root=instance_data_dir,
        stochastic_attribute=stochastic_attribute,
        token_map=token_map,
        use_template=use_template,
        class_data_root=class_data_dir if with_prior_preservation else None,
        class_prompt=class_prompt,
        tokenizer=tokenizer,
        size=resolution,
        color_jitter=color_jitter,
        use_face_segmentation_condition=use_face_segmentation_condition,
    )

    train_dataset.blur_amount = 200

    train_dataloader = text2img_dataloader(
        train_dataset, train_batch_size, tokenizer, vae, text_encoder
    )

    index_no_updates = torch.arange(len(tokenizer)) != placeholder_token_ids[0]

    for tok_id in placeholder_token_ids:
        index_no_updates[tok_id] = False

    unet.requires_grad_(False)
    vae.requires_grad_(False)

    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    ti_optimizer = optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=ti_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay_ti,
    )

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=ti_optimizer,
        num_warmup_steps=20,
        num_training_steps=max_train_steps_ti,
    )

    train_inversion(
        unet,
        vae,
        text_encoder,
        tokenizer,
        train_dataloader,
        max_train_steps_ti,
        accum_iter=gradient_accumulation_steps,
        scheduler=noise_scheduler,
        index_no_updates=index_no_updates,
        optimizer=ti_optimizer,
        lr_scheduler=lr_scheduler,
        save_steps=save_steps,
        placeholder_tokens=placeholder_tokens,
        placeholder_token_ids=placeholder_token_ids,
        save_path=output_dir,
        test_image_path=instance_data_dir,
        log_wandb=log_wandb,
        mixed_precision=False,
    )


if __name__ == "__main__":
    fire.Fire(train)
