import torch
import amp
from tqdm import tqdm as tqdm



# train_epoch_fun
def train_epoch_fun(model, train_loader, optimizer, criterion, device, USE_AMP=False):
    #####################
    # train the model
    #################

    model.train()
    train_running_loss = []
    bar = tqdm(train_loader)
    for (image, label) in bar:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()

        if not USE_AMP:
            logits_m = model(image)  # output size: BATCH_SIZE * out_dim*1
            loss = criterion.loss(logits_m, label)
            loss.backward()
            optimizer.step()
        else:
            logits_m = model(image)
            loss = criterion.loss(logits_m, label)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.stpe()

        if torch.cuda.is_available():
            loss_detach_cpu = loss.detach().cpu()
            loss_detach_cpu_np = loss_detach_cpu.numpy()
        else:
            loss_detach_cpu = loss.detach()
            loss_detach_cpu_np = loss_detach_cpu.numpy()

        train_running_loss.append(loss_detach_cpu_np)

        smooth_loss = sum(train_running_loss[-100:]) / min(len(train_running_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_detach_cpu_np, smooth_loss))
    return train_running_loss


