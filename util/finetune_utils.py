import torch
from tqdm import tqdm


def finetune_decoder(config, model, warp_output, inpaint_output):
    params = [{"params": model.vae.decoder.parameters(), "lr": config["decoder_learning_rate"]}]
    optimizer = torch.optim.Adam(params)
    for _ in tqdm(range(config["num_finetune_decoder_steps"]), leave=False):
        optimizer.zero_grad()
        loss = model.finetune_decoder_step(
            inpaint_output["inpainted_image"].detach(),
            inpaint_output["latent"].detach(),
            warp_output["warped_image"].detach(),
            warp_output["inpaint_mask"].detach(),
        )
        loss.backward()
        optimizer.step()

    del optimizer


def finetune_depth_model(config, model, warp_output, epoch, scaler):
    params = [{"params": model.depth_model.parameters(), "lr": config["depth_model_learning_rate"]}]
    optimizer = torch.optim.Adam(params)

    mask = warp_output["warped_depth"] != -1

    for _ in tqdm(range(config["num_finetune_depth_model_steps"]), leave=False):
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=config["device"], enabled=config["enable_mix_precision"]):
            loss = model.finetune_depth_model_step(
                warp_output["warped_depth"],
                model.images[epoch],
                mask,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
