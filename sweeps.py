from inversion import train

from typing import Optional, Literal


def sweep():
    for lr in [1e-3]:
        for idx in range(5):
            for weight_decay in [1e-2]:
                for init_type in ["emma|watson", "<zero>|<zero>"]:

                    train(
                        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
                        instance_data_dir="./data_example_captioned_small",
                        output_dir=f"./exps_em/{lr}_{weight_decay}_{init_type.replace('|', '_')}",
                        resolution=512,
                        train_batch_size=1,
                        gradient_accumulation_steps=1,
                        learning_rate_ti=lr,
                        color_jitter=True,
                        placeholder_tokens="<s1>|<s2>",
                        placeholder_token_at_data="<krk>|<s1><s2>",
                        initializer_tokens=init_type,
                        save_steps=100,
                        max_train_steps_ti=2000,
                        use_template=None,
                        weight_decay_ti=weight_decay,
                        device="cuda:0",
                        use_face_segmentation_condition=True,
                        extra_args={
                            "init_type": init_type,
                        },
                        project_name="invpp_krk",
                    )


if __name__ == "__main__":
    sweep()
