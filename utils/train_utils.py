import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LinearLR, SequentialLR
from utils.common import evaluate


def log_msg(msg, logger=None, level="info"):
    if logger is None:
        print(msg)
    else:
        getattr(logger, level, logger.info)(msg)


def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Returns cutmixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size, C, H, W = x.size()
    index = torch.randperm(batch_size, device=x.device)

    # compute dimensions
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # CREATE A CLONE FIRST
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # recompute lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam  # Return the cloned version

def get_optimizer(model, config):
    """
    Create optimized optimizer based on configuration.
    """
    tr_cfg = config.get('training', {})
    optimizer_name = tr_cfg.get('optimizer', 'sgd').lower()
    lr = float(tr_cfg.get('learning_rate', 0.1))  # Ensure float
    weight_decay = float(tr_cfg.get('weight_decay', 5e-4))  # Ensure float
    
    if optimizer_name == 'adamw':
        # AdamW - Best for finetuning pretrained models
        betas = tr_cfg.get('betas', [0.9, 0.999])
        eps = float(tr_cfg.get('eps', 1e-8))  # Ensure eps is float
        
        # Separate weight decay for different parameter groups
        no_decay = ['bias', 'LayerNorm.weight', 'BatchNorm.weight', 'bn']
        param_groups = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps
        )
        
    elif optimizer_name == 'adam':
        # Adam - Alternative for finetuning
        betas = tr_cfg.get('betas', [0.9, 0.999])
        eps = float(tr_cfg.get('eps', 1e-8))  # Ensure eps is float
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        
    elif optimizer_name == 'sgd':
        # SGD - Original approach
        momentum = float(tr_cfg.get('momentum', 0.9))  # Ensure float
        
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'bias' not in n],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if 'bias' in n],
             'weight_decay': 0}
        ]
        
        optimizer = optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            nesterov=True
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def get_scheduler(optimizer, config, steps_per_epoch):
    """
    Create learning rate scheduler based on configuration.
    """
    tr_cfg = config.get('training', {})
    scheduler_type = tr_cfg.get('lr_scheduler', 'cosine')
    num_epochs = int(tr_cfg.get('epochs', 10))  # Ensure int
    warmup_epochs = float(tr_cfg.get('warmup_epochs', 0.0))  # Ensure float
    
    if scheduler_type == 'cosine':
        if warmup_epochs > 0:
            # Cosine with warmup - Best for finetuning
            warmup_steps = int(warmup_epochs * steps_per_epoch)
            total_steps = num_epochs * steps_per_epoch
            
            warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=0.1, 
                end_factor=1.0, 
                total_iters=warmup_steps
            )
            
            cosine_scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=total_steps - warmup_steps,
                eta_min=1e-6
            )
            
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
            return scheduler, True  # step_per_batch=True
        else:
            # Standard cosine
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
            return scheduler, False  # step_per_batch=False
            
    elif scheduler_type == 'onecycle':
        # OneCycle - Very fast convergence
        max_lr = float(tr_cfg.get('learning_rate', 0.1)) * 3  # Peak LR, ensure float
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        return scheduler, True  # step_per_batch=True
    else:
        return None, False


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    num_epochs,
    scheduler=None,
    step_per_batch=False,
    aug_list=None,
    log_interval=10,
    use_amp=True,
    logger=None
):
    """
    Train the model for one epoch with scheduler support.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    scaler = GradScaler('cuda') if use_amp else None

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch+1:3d}/{num_epochs}",
        unit="batch",
        disable=logger is not None
    )

    for batch_idx, (inputs, targets) in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # MixUp/CutMix selection
        use_mixup = aug_list and 'mixup' in aug_list
        use_cutmix = aug_list and 'cutmix' in aug_list

        if use_mixup or use_cutmix:
            if use_mixup and use_cutmix:
                aug = 'mixup' if random.random() < 0.5 else 'cutmix'
            elif use_mixup:
                aug = 'mixup'
            else:
                aug = 'cutmix'

            if aug == 'mixup':
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha=1.0)

            def compute_loss(outputs):
                return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

        # Forward pass and loss computation
        if use_amp:
            with autocast('cuda'):
                outputs = model(inputs)
                if use_mixup or use_cutmix:
                    loss = compute_loss(outputs)
                else:
                    loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            if use_mixup or use_cutmix:
                loss = compute_loss(outputs)
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Step scheduler if per-batch stepping (AFTER optimizer.step())
        if scheduler and step_per_batch:
            scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        if logger is None and ((batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader)):
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def train_model(
    model,
    train_loader,
    test_loader,
    train_eval_loader,
    config,
    device,
    save_dir=None,
    writer=None,
    logger=None
):
    """
    Train a model with updated optimizer and scheduler support.
    """
    aug_list = config.get('train_data_augmentation', [])
    tr_cfg = config.get('training', {})
    num_epochs = tr_cfg.get('epochs', 100)
    save_step = tr_cfg.get('save_step', 20)
    save_models = tr_cfg.get('save_models', False)
    early_stop = tr_cfg.get('early_stopping', False)
    patience = tr_cfg.get('patience', 10)
    resume = tr_cfg.get('resume', False)
    use_amp = tr_cfg.get('use_amp', torch.cuda.is_available())

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    # get optimizer using new function
    optimizer = get_optimizer(model, config)
    
    # get scheduler using new function
    scheduler, step_per_batch = get_scheduler(optimizer, config, len(train_loader))

    best_acc = 0.0
    epochs_no_improve = 0
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    start_epoch = 0

    # Resume logic (unchanged)
    if resume and save_dir:
        ckpt_paths = list(save_dir.glob('checkpoint_epoch*.pth'))
        if ckpt_paths:
            def _get_epoch(path):
                m = re.search(r'checkpoint_epoch(\d+)\.pth$', str(path))
                return int(m.group(1)) if m else -1
            latest_path = max(ckpt_paths, key=_get_epoch)
            checkpoint = torch.load(latest_path, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            if start_epoch >= num_epochs-1:
                log_msg(f"[Resume] Already trained to epoch {num_epochs-1}, skipping.", logger)
                return model
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            best_acc = checkpoint.get('best_acc', best_acc)
            train_losses = checkpoint.get('train_losses', [])
            test_losses = checkpoint.get('test_losses', [])
            train_accuracies = checkpoint.get('train_accuracies', [])
            test_accuracies = checkpoint.get('test_accuracies', [])
            log_msg(f"[Resume] Resumed from {latest_path} at epoch {start_epoch}", logger)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, num_epochs, 
            scheduler=scheduler,
            step_per_batch=step_per_batch,
            aug_list=aug_list,
            use_amp=use_amp, 
            logger=logger
        )

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, use_amp=use_amp,
            logger=logger, epoch=epoch, num_epochs=num_epochs
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # Step scheduler if per-epoch stepping (no epoch parameter)
        if scheduler and not step_per_batch:
            scheduler.step()

        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch+1)
            writer.add_scalar('Acc/train', train_acc, epoch+1)
            writer.add_scalar('Loss/test', test_loss, epoch+1)
            writer.add_scalar('Acc/test', test_acc, epoch+1)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch+1)

        # Save checkpoints
        if save_dir:
            if save_models and (epoch+1) % save_step == 0:
                ckpt_file = save_dir / f'checkpoint_epoch{epoch+1}.pth'
                torch.save({
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'best_acc': best_acc,
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_accuracies': train_accuracies,
                    'test_accuracies': test_accuracies
                }, ckpt_file)
                log_msg(f"[Save ] Saved checkpoint: {ckpt_file}", logger)

            if test_acc > best_acc:
                best_acc = test_acc
                epochs_no_improve = 0
                best_file = save_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_accuracies': train_accuracies,
                    'test_accuracies': test_accuracies
                }, best_file)
                log_msg(f"[Save ] Best model @ epoch {epoch+1:3d} with acc {test_acc:.2f}%", logger)
            else:
                epochs_no_improve += 1

        # Early stopping conditions
              
        if early_stop and epochs_no_improve >= patience:
            log_msg(f"[EarlyStop] Stopped after {epoch+1} epochs (no improvement in {patience})", logger)
            break

        log_msg(
            f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.5f}",
            logger
        )

    # Load best model if available
    best_model_file = save_dir / 'best_model.pth' if save_dir else None
    if best_model_file and best_model_file.exists():
        best_ckpt = torch.load(best_model_file, map_location=device)
        model.load_state_dict(best_ckpt['state_dict'])
        train_loss, train_acc = evaluate(model, train_eval_loader, criterion, device, use_amp=use_amp)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, use_amp=use_amp)
        log_msg(
            f"[Loaded] Best model from {best_model_file} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%",
            logger
        )

    return model


@torch.no_grad()
def get_model_predictions(model, data_loader, device, use_amp=True):
    """Get model predictions and confidences."""
    model.eval()
    preds, targets, confs = [], [], []

    for inputs, labs in data_loader:
        inputs, labs = inputs.to(device, non_blocking=True), labs.to(device, non_blocking=True)
        if use_amp:
            with autocast('cuda'):
                out = model(inputs)
        else:
            out = model(inputs)
        prob = F.softmax(out, dim=1)
        _, p = prob.max(1)
        preds.append(p.cpu().numpy())
        targets.append(labs.cpu().numpy())
        confs.append(prob.cpu().numpy())

    return (
        np.concatenate(preds),
        np.concatenate(targets),
        np.concatenate(confs)
    )


@torch.no_grad()
def compute_loss_values(model, data_loader, device, use_amp=True):
    """Compute per-sample loss values."""
    model.eval()
    losses, targets = [], []

    for inputs, labs in data_loader:
        inputs, labs = inputs.to(device, non_blocking=True), labs.to(device, non_blocking=True)
        if use_amp:
            with autocast('cuda'):
                out = model(inputs)
                per_example = F.cross_entropy(out, labs, reduction='none')
        else:
            out = model(inputs)
            per_example = F.cross_entropy(out, labs, reduction='none')

        losses.append(per_example.cpu().numpy())
        targets.append(labs.cpu().numpy())

    return np.concatenate(losses), np.concatenate(targets)