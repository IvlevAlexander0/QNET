import sys
import os
from io import StringIO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary

from config import Config
from data import load_severstal_data, SeverstalDataset, get_transforms
from models import create_model
from training import train_one_epoch, evaluate, FocalLoss
from utils import *


def main(config_path='config.json'):
    config = Config.from_json(config_path)
    original_resume_from = config.resume_from  # <-- сохраняем

    if original_resume_from and os.path.exists(original_resume_from):
        run_dir = original_resume_from
        resume_config_path = os.path.join(original_resume_from, 'config.json')
        if os.path.exists(resume_config_path):
            resume_config = Config.from_json(resume_config_path)
            for key, value in vars(resume_config).items():
                setattr(config, key, value)
            # Восстанавливаем resume_from, чтобы он не был затерт
            config.resume_from = original_resume_from
            run_dir, logger = setup_logging(config, run_dir=run_dir, config_path=resume_config_path)
            logger.info(f"Resuming from {config.resume_from}")
        else:
            raise FileNotFoundError(f"Config not found in resume directory: {resume_config_path}")
    else:
        run_dir, logger = setup_logging(config, config_path=config_path)

    if config.enable_tensorboard and run_dir is not None:
        writer = get_tensorboard_writer(run_dir)
    else:
        writer = None

    set_seed(config.seed, config.det)

    # Загрузка данных
    logger.info("Loading data...")
    (image_paths, image_labels,
     train_paths, train_labels,
     val_paths, val_labels,
     test_paths, test_labels,
     class_to_idx, num_classes) = load_severstal_data(
         config.dataset_path, config.train_part, config.val_part
     )
    logger.info(f"Data loaded: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

    # Статистика
    class_names = [None] * num_classes
    for name, idx in class_to_idx.items():
        class_names[idx] = name

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    print_dataset_stats(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names)
    stats_str = sys.stdout.getvalue()
    sys.stdout = old_stdout
    logger.info(f"Dataset stats:\n{stats_str}")

    # Трансформации
    mean, std = torch.tensor(config.transform_mean), torch.tensor(config.transform_std)
    noise_config = getattr(config, 'noise_config', None)
    data_transform = get_transforms(config.in_height, config.in_width, mean, std, noise_config=noise_config)

    train_dataset = SeverstalDataset(train_paths, train_labels, data_transform['train'])
    val_dataset = SeverstalDataset(val_paths, val_labels, data_transform['val'])
    test_dataset = SeverstalDataset(test_paths, test_labels, data_transform['test'])

    num_workers = getattr(config, 'num_workers', 4)
    prefetch_factor = getattr(config, 'prefetch_factor', 2)

    trainloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=SeverstalDataset.collate_fn,
    )
    validloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=SeverstalDataset.collate_fn,
    )
    testloader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=SeverstalDataset.collate_fn,
    )

    # Модель
    logger.info(f"Creating model: {config.model_name} (mode: {config.model_mode})")
    model, model_info = create_model(config, num_classes, (config.in_height, config.in_width))
    model = model.to(config.device)
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Загрузка весов
    if config.C_load_params and config.Q_load_params:
        # Полная загрузка
        if not os.path.exists(config.h_load_path):
            raise FileNotFoundError(f"Full model weights not found: {config.h_load_path}")
        state_dict = torch.load(config.h_load_path, map_location=config.device)
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded full model from {config.h_load_path}")

    elif config.C_load_params:
        # Загрузка только классической части
        if not os.path.exists(config.c_load_path):
            raise FileNotFoundError(f"Classical weights not found: {config.c_load_path}")
        state_dict = torch.load(config.c_load_path, map_location=config.device)
        prefixes = model_info.get('classical_prefix', [])
        if not prefixes:
            logger.warning("No classical prefix defined, loading whole state_dict (strict=False)")
            model.load_state_dict(state_dict, strict=False)
        else:
            load_partial_weights(model, state_dict, prefixes, logger=logger)
        logger.info(f"Loaded classical part from {config.c_load_path}")

    elif config.Q_load_params:
        # Загрузка только квантовой части
        if not os.path.exists(config.q_load_path):
            raise FileNotFoundError(f"Quantum weights not found: {config.q_load_path}")
        state_dict = torch.load(config.q_load_path, map_location=config.device)
        prefixes = model_info.get('quantum_prefix', [])
        if not prefixes:
            logger.warning("No quantum prefix defined, loading whole state_dict (strict=False)")
            model.load_state_dict(state_dict, strict=False)
        else:
            load_partial_weights(model, state_dict, prefixes, logger=logger)
        logger.info(f"Loaded quantum part from {config.q_load_path}")

    if not config.C_train_flag:
        for param in model_info.get('classical_params', []):
            param.requires_grad = False
        logger.info("Classical part frozen")
    if not config.Q_train_flag:
        for param in model_info.get('quantum_params', []):
            param.requires_grad = False
        logger.info("Quantum part frozen")
    model_summary = str(summary(model, input_size=(1, 1, config.in_height, config.in_width), verbose=0, device=config.device))
    logger.info(f"Model summary:\n{model_summary}")
    
    # Потери, оптимизатор

    if hasattr(config, 'class_weights') and len(config.class_weights) == num_classes:
        class_weights = torch.tensor(config.class_weights)
    else:
        class_weights = torch.ones(num_classes)
        logger.warning("Class weights not provided or mismatch – using equal weights.")
    loss_fn = nn.BCELoss(weight=class_weights.to(config.device))
    #loss_fn = FocalLoss(class_weights.to(config.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.initial_lr, weight_decay=config.w_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # Обучение
    start_epoch = 0
    best_val_acc, best_val_f1 = 0.0, 0.0
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
               'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    if config.resume_from:
        try:
            start_epoch, best_val_acc, best_val_f1, history = load_checkpoint(
                run_dir, model, optimizer, scheduler
            )
            # Принудительно перемещаем модель на device (если загружали на cpu)
            model = model.to(config.device)
        except FileNotFoundError as e:
            logger.warning(f"Checkpoint not found: {e}. Starting from scratch.")
            start_epoch = 0

    model_dir = os.path.join(run_dir, 'models') if run_dir else '.'
    os.makedirs(model_dir, exist_ok=True)

    best_acc_path = os.path.join(model_dir, 'best_acc_model.pth')
    best_f1_path = os.path.join(model_dir, 'best_f1_model.pth')

    logger.info(f"Start train on {config.device}")
    for epoch in range(start_epoch, config.epochs):
        start_time = time.time()

        train_loss, train_acc, train_prec, train_rec, train_f1, train_tp, train_fp, train_fn = train_one_epoch(
            model, trainloader, loss_fn, optimizer, config.device, num_classes,
            config.threshold, config.eps
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, val_tp, val_fp, val_fn = evaluate(
            model, validloader, loss_fn, config.device, num_classes,
            config.threshold, config.eps
        )
        scheduler.step()

        epoch_time = time.time() - start_time

        # Сохраняем историю
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1.mean().item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1.mean().item())

        # Логируем общие метрики
        logger.info(
            f"Epoch {epoch+1}/{config.epochs} [{epoch_time:.1f}s] "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} MacroF1={train_f1.mean():.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} MacroF1={val_f1.mean():.4f}"
        )

        # Логируем поклассные метрики (precision, recall, f1)
        # Вычисляем per-class метрики из суммарных tp/fp/fn
        eps = config.eps
        train_prec_cls = train_tp / (train_tp + train_fp + eps)
        train_rec_cls = train_tp / (train_tp + train_fn + eps)
        train_f1_cls = 2 * train_prec_cls * train_rec_cls / (train_prec_cls + train_rec_cls + eps)

        val_prec_cls = val_tp / (val_tp + val_fp + eps)
        val_rec_cls = val_tp / (val_tp + val_fn + eps)
        val_f1_cls = 2 * val_prec_cls * val_rec_cls / (val_prec_cls + val_rec_cls + eps)

        # Формируем строки с метриками по классам (используем class_names, если они доступны)
        # В main у нас есть class_names
        train_cls_str = ", ".join([f"{class_names[i]}:P={train_prec_cls[i]:.3f} R={train_rec_cls[i]:.3f} F1={train_f1_cls[i]:.3f}" for i in range(num_classes)])
        val_cls_str = ", ".join([f"{class_names[i]}:P={val_prec_cls[i]:.3f} R={val_rec_cls[i]:.3f} F1={val_f1_cls[i]:.3f}" for i in range(num_classes)])
        logger.info(f"Train per-class: {train_cls_str}")
        logger.info(f"Val   per-class: {val_cls_str}")

        save_checkpoint(run_dir, model, optimizer, scheduler, epoch, best_val_acc, best_val_f1, history, config)

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('F1_macro/train', train_f1.mean().item(), epoch)
            writer.add_scalar('F1_macro/val', val_f1.mean().item(), epoch)
            # per‑class F1
            for i, cls in enumerate(class_names):
                writer.add_scalar(f'F1_per_class/{cls}/train', train_f1_cls[i].item(), epoch)
                writer.add_scalar(f'F1_per_class/{cls}/val', val_f1_cls[i].item(), epoch)

        # Сохраняем лучшие модели в папку с логами
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_acc_path)
            logger.info(f"New best accuracy: {val_acc:.4f} (saved to {best_acc_path})")
        val_macro_f1 = val_f1.mean().item()
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), best_f1_path)
            logger.info(f"New best F1: {best_val_f1:.4f} (saved to {best_f1_path})")

    # Финальная оценка
    # 1. Оценка модели с наилучшей точностью (best acc)
    logger.info("Loading best model (by accuracy) for final evaluation")
    model.load_state_dict(torch.load(best_acc_path))

    logger.info("Evaluating on validation set (best acc model)")
    val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _ = evaluate(
        model, validloader, loss_fn, config.device, num_classes,
        config.threshold, config.eps
    )
    logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val MacroF1={val_f1.mean():.4f}")
    for i, cls in enumerate(class_names):
        logger.info(f"  Class {cls}: Precision={val_prec[i]:.4f}, Recall={val_rec[i]:.4f}, F1={val_f1[i]:.4f}")

    logger.info("Evaluating on test set (best acc model)")
    test_loss, test_acc, test_prec, test_rec, test_f1, _, _, _ = evaluate(
        model, testloader, loss_fn, config.device, num_classes,
        config.threshold, config.eps
    )
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test MacroF1={test_f1.mean():.4f}")
    for i, cls in enumerate(class_names):
        logger.info(f"  Class {cls}: Precision={test_prec[i]:.4f}, Recall={test_rec[i]:.4f}, F1={test_f1[i]:.4f}")

    # 2. Оценка модели с наилучшим F1 (best f1)
    logger.info("Loading best model (by F1) for final evaluation")
    model.load_state_dict(torch.load(best_f1_path))

    logger.info("Evaluating on validation set (best F1 model)")
    val_loss_f1, val_acc_f1, val_prec_f1, val_rec_f1, val_f1_metrics, _, _, _ = evaluate(
        model, validloader, loss_fn, config.device, num_classes,
        config.threshold, config.eps
    )
    logger.info(f"Val Loss (F1 model): {val_loss_f1:.4f}, Val Acc (F1 model): {val_acc_f1:.4f}, Val MacroF1 (F1 model)={val_f1_metrics.mean():.4f}")
    for i, cls in enumerate(class_names):
        logger.info(f"  Class {cls}: Precision={val_prec_f1[i]:.4f}, Recall={val_rec_f1[i]:.4f}, F1={val_f1_metrics[i]:.4f}")

    logger.info("Evaluating on test set (best F1 model)")
    test_loss_f1, test_acc_f1, test_prec_f1, test_rec_f1, test_f1_metrics, _, _, _ = evaluate(
        model, testloader, loss_fn, config.device, num_classes,
        config.threshold, config.eps
    )
    logger.info(f"Test Loss (F1 model): {test_loss_f1:.4f}, Test Acc (F1 model): {test_acc_f1:.4f}, Test MacroF1 (F1 model)={test_f1_metrics.mean():.4f}")
    for i, cls in enumerate(class_names):
        logger.info(f"  Class {cls}: Precision={test_prec_f1[i]:.4f}, Recall={test_rec_f1[i]:.4f}, F1={test_f1_metrics[i]:.4f}")


    # Сохранение визуализаций (если включено)
    if config.save_visualizations and run_dir:
        vis_dir = os.path.join(run_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Графики обучения
        save_training_plots(history, save_dir=vis_dir)
        logger.info(f"Training plots saved to {vis_dir}/training_metrics.png")

        # Предсказания на тесте
        save_predictions(model, test_paths, test_labels, class_names, config.device,
                        transform=data_transform['test'], save_dir=vis_dir)
        logger.info(f"Predictions saved to {vis_dir}")

    close_tensorboard_writer()

    logger.info("=== Run finished ===")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    main(config_path)