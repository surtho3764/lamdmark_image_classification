import torch
import numpy as np
from tqdm import tqdm as tqdm
from lib.pytorch_lib.evaluations.global_average_precision import global_average_precision_score



# val_epoch function
def val_epoch_fun(model, valid_loader, criterion, device, USE_AMP, GET_VAL_OUTPUT=False):
    #####################
    # eval the model
    #################

    model.eval()
    val_running_loss = []
    PRODS_M = []
    PREDS_M = []
    TARGETS = []

    bar = tqdm(valid_loader)

    with torch.no_grad():
        for (image, label) in bar:
            image, label = image.to(device), label.to(device)

            logits_m = model(image)

            lmax_m = logits_m.max(1)
            probs_m = lmax_m.values
            preds_m = lmax_m.indices

            # loss
            loss = criterion.loss(logits_m, label)

            if torch.cuda.is_available():
                probs_m_detach_cpu = probs_m.detach().cpu()
                preds_m_detach_cpu = preds_m.detach().cpu()
                label_detach_cpu = label.detach().cpu()
                loss_detach_cpu = loss.detach().cpu()
                loss_detach_cpu_np = loss_detach_cpu.numpy()
            else:
                probs_m_detach_cpu = probs_m.detach()
                preds_m_detach_cpu = preds_m.detach()
                label_detach_cpu = label.detach()
                loss_detach_cpu = loss.detach()
                loss_detach_cpu_np = loss_detach_cpu.numpy()

            PRODS_M.append(probs_m_detach_cpu)
            PREDS_M.append(preds_m_detach_cpu)
            TARGETS.append(label_detach_cpu)
            val_running_loss.append(loss_detach_cpu_np)

        val_running_loss = np.mean(val_running_loss)

        # torch.cat
        PRODS_M = torch.cat(PRODS_M).numpy()
        PREDS_M = torch.cat(PREDS_M).numpy()
        TARGETS = torch.cat(TARGETS)

    if GET_VAL_OUTPUT:
        return logits_m
    else:
        acc_m = (PREDS_M == TARGETS.numpy()).mean() * 100
        y_true = {idx: label if label >= 0 else None for idx, label in enumerate(TARGETS)}
        y_pred_m = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(PREDS_M, PRODS_M))}
        gap_m = global_average_precision_score(y_true, y_pred_m)
        return val_running_loss, acc_m, gap_m

