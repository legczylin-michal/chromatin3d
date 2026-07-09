import logging

import pandas as pd
import torch
from tqdm import tqdm

from chromatin3d.utils.conversions import pos_2_hic, normalise
from chromatin3d.utils.losses import data_loss_insulation_score as data_loss
from chromatin3d.utils.losses import loops_loss, rouse_loss, smoothness_loss
from chromatin3d.utils.metrics import linker_length_metric, persistence_length_metric, heatmap_correlation_metric

logger = logging.getLogger("chromatin3d")

torch.autograd.set_detect_anomaly(True)


def train(model, optimizer, train_loader, valid_loader, epochs, repeats, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    history_losses = {"data_losses": [], "loops_losses": [], "rouse_losses": [], "smoothness_losses": [], "total_losses": []}
    history_metrics = {"heatmap_correlation_metrics": [], "linker_length_mean_metrics": [], "linker_length_std_metrics": [], "persistence_length_metrics": []}

    patience_counter = 0
    for epoch in tqdm(range(epochs)):
        if patience_counter >= patience:
            logger.debug("Patience triggered. End of learning")
            break

        ## training phase
        model.train()

        # accumulated losses in one epoch
        data_a_l = 0
        loops_a_l = 0
        rouse_a_l = 0
        smoothness_a_l = 0
        total_a_l = 0

        # iterate over batches
        for inputs in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            # losses in one batch
            loops_l = 0
            rouse_l = 0
            smoothness_l = 0

            # generate several 3D structures
            hic = []
            for _ in range(repeats):
                outputs = model(inputs)

                logger.debug(f"outputs: {outputs}")
                logger.debug(f"are there nans in outputs?: {outputs.isnan().any()}")

                hic.append(pos_2_hic(outputs))

                loops_l += loops_loss(inputs, outputs)
                rouse_l += rouse_loss(outputs)
                smoothness_l += smoothness_loss(outputs)
            hic = torch.stack(hic).mean(dim=0)
            logger.debug(f"hic: {hic}")
            logger.debug(f"are there any nans in hic?: {hic.isnan().any()}")

            # averaged losses in one batch
            data_l = data_loss(normalise(inputs), normalise(hic)).mean()
            loops_l = (loops_l / repeats).mean()
            rouse_l = (rouse_l / repeats).mean()
            smoothness_l = (smoothness_l / repeats).mean()

            # total loss to be optimised, convex combination
            coef = torch.Tensor([1, 10, 100, 20])
            coef /= coef.norm()

            total_l = coef[0] * data_l + coef[1] * loops_l + coef[2] * rouse_l + coef[3] * smoothness_l

            logger.debug([data_l, loops_l, rouse_l, smoothness_l, total_l])
            logger.debug([i for i in model.parameters()])

            logger.debug(f"loss:{total_l.item()}")
            logger.debug(f"loss finite:{torch.isfinite(total_l)}")

            total_l.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        logger.debug(f"bad grad:{name}")
                        break

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2

            total_norm = total_norm ** 0.5
            logger.debug(f"grad norm:{total_norm}")
            optimizer.step()
            logger.debug([i for i in model.parameters()])

            # update accumulated losses
            data_a_l += data_l.item()
            loops_a_l += loops_l.item()
            rouse_a_l += rouse_l.item()
            smoothness_a_l += smoothness_l.item()
            total_a_l += total_l.item()

        # update history of losses
        history_losses["data_losses"].append(data_a_l / len(train_loader))
        history_losses["loops_losses"].append(loops_a_l / len(train_loader))
        history_losses["rouse_losses"].append(rouse_a_l / len(train_loader))
        history_losses["smoothness_losses"].append(smoothness_a_l / len(train_loader))
        history_losses["total_losses"].append(total_a_l / len(train_loader))

        ## validation phase
        model.eval()

        # accumulated metrics in one epoch
        heatmap_correlation_a_m = 0
        linker_length_mean_a_m = 0
        linker_length_std_a_m = 0
        persistence_length_a_m = 0

        # no learning occurs
        with torch.no_grad():
            # iterate over batches
            for inputs in valid_loader:
                inputs = inputs.to(device)

                # metrics in one batch
                linker_length_mean_m = 0
                linker_length_std_m = 0
                persistence_length_m = 0

                # generate several 3D structures
                hic = []
                for _ in range(repeats):
                    outputs = model(inputs)

                    hic.append(pos_2_hic(outputs))

                    mean, std = linker_length_metric(outputs)
                    linker_length_mean_m += mean
                    linker_length_std_m += std
                    persistence_length_m += persistence_length_metric(outputs)
                hic = torch.stack(hic).mean(dim=0)

                # averaged metrics in one batch
                heatmap_correlation_m = heatmap_correlation_metric(inputs, hic).mean()
                linker_length_mean_m = (linker_length_mean_m / repeats).mean()
                linker_length_std_m = (linker_length_std_m / repeats).mean()
                persistence_length_m = (persistence_length_m / repeats).mean()

                # update accumulated metrics
                heatmap_correlation_a_m += heatmap_correlation_m.item()
                linker_length_mean_a_m += linker_length_mean_m.item()
                linker_length_std_a_m += linker_length_std_m.item()
                persistence_length_a_m += persistence_length_m.item()

        # update history of metrics
        history_metrics["heatmap_correlation_metrics"].append(heatmap_correlation_a_m / len(valid_loader))
        history_metrics["linker_length_mean_metrics"].append(linker_length_mean_a_m / len(valid_loader))
        history_metrics["linker_length_std_metrics"].append(linker_length_std_a_m / len(valid_loader))
        history_metrics["persistence_length_metrics"].append(persistence_length_a_m / len(valid_loader))

    return pd.DataFrame(history_losses), pd.DataFrame(history_metrics)
