import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil

from pytorch3dunet.datasets.utils import get_train_loaders
from pytorch3dunet.unet3d.config import TorchDevice
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model, is_model_2d
from pytorch3dunet.unet3d.utils import (
    RunningAverage,
    TensorboardFormatter,
    create_lr_scheduler,
    create_optimizer,
    get_logger,
    get_number_of_learnable_parameters,
    load_checkpoint,
    save_checkpoint,
)

logger = get_logger("UNetTrainer")


def create_trainer(config: dict) -> "UNetTrainer":
    # Create the model
    model = get_model(config["model"])

    device = config.get("device", None)
    assert device, "Device not specified in the config file and could not be inferred automatically"
    logger.info(f"Using device: {device}")
    model.to(device)

    # Log the number of learnable parameters
    logger.info(f"Number of learnable params {get_number_of_learnable_parameters(model)}")

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_train_loaders(config)

    # Create the optimizer
    optimizer = create_optimizer(config["optimizer"], model)

    # Create learning rate adjustment strategy
    lr_scheduler = create_lr_scheduler(config.get("lr_scheduler", None), optimizer)

    trainer_config = config["trainer"]
    # Create tensorboard formatter
    tensorboard_formatter_config = trainer_config.pop("tensorboard_formatter", {})
    tensorboard_formatter = TensorboardFormatter(**tensorboard_formatter_config)
    # Create trainer
    resume = trainer_config.pop("resume", None)
    pre_trained = trainer_config.pop("pre_trained", None)

    return UNetTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_criterion=loss_criterion,
        eval_criterion=eval_criterion,
        loaders=loaders,
        tensorboard_formatter=tensorboard_formatter,
        resume=resume,
        pre_trained=pre_trained,
        device=device,
        **trainer_config,
    )


def _split_and_move_to_device(t: Any, device: TorchDevice) -> tuple[torch.Tensor, torch.Tensor]:
    def _move_to_device(input, device):
        if isinstance(input, tuple | list):
            return tuple([_move_to_device(x, device) for x in input])
        else:
            # send batch to device, see: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
            return input.to(device, non_blocking=True)

    input, target = _move_to_device(t, device)
    return input, target


class UNetTrainer:
    """UNet trainer.

    Args:
        model: UNet 3D model to be trained.
        optimizer: Optimizer used for training.
        lr_scheduler: Learning rate scheduler. Note that lr_scheduler.step() is invoked after every validation
            step (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion: Loss function.
        eval_criterion: Used to compute training/validation metric (such as Dice, IoU, AP or Rand score).
            Saving the best checkpoint is based on the result of this function on the validation set.
        loaders: Dictionary with 'train' and 'val' data loaders.
        checkpoint_dir: Directory for saving checkpoints and tensorboard logs.
        max_num_epochs: Maximum number of epochs.
        max_num_iterations: Maximum number of iterations.
        validate_after_iters: Validate after that many iterations. Default: 200.
        log_after_iters: Number of iterations before logging to tensorboard. Default: 100.
        validate_iters: Number of validation iterations. If None validate on the whole validation set. Default: None.
        num_iterations: Useful when loading the model from the checkpoint. Default: 1.
        num_epoch: Useful when loading the model from the checkpoint. Default: 0.
        eval_score_higher_is_better: If True higher eval scores are considered better. Default: True.
        tensorboard_formatter: Converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard. Default: None.
        skip_train_validation: If True eval_criterion is not evaluated on the training set (used when
            evaluation is expensive). Default: False.
        resume: Path to the checkpoint to be resumed. Default: None.
        pre_trained: Path to the pre-trained model. Default: None.
        max_val_images: Maximum number of images to log during validation. Default: 100.
        device: Device to use for training (CPU, CUDA, MPS). Default: None.
    """

    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        loss_criterion,
        eval_criterion,
        loaders,
        checkpoint_dir,
        max_num_epochs,
        max_num_iterations,
        validate_after_iters=200,
        log_after_iters=100,
        validate_iters=None,
        num_iterations=1,
        num_epoch=0,
        eval_score_higher_is_better=True,
        tensorboard_formatter=None,
        skip_train_validation=False,
        resume=None,
        pre_trained=None,
        max_val_images=4,
        device: TorchDevice | None = None,
    ):
        self.max_val_images = max_val_images
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.loaders = loaders
        self.iters_per_epoch = len(self.loaders["train"])
        self.checkpoint_dir = checkpoint_dir
    

        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = max(1, self.iters_per_epoch // 10)
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        assert device, "Device must be specified"
        self.device = device

        logger.info(model)
        logger.info(f"eval_score_higher_is_better: {eval_score_higher_is_better}")
        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float("-inf")
        else:
            self.best_eval_score = float("+inf")

        self.writer = SummaryWriter(
            log_dir=os.path.join(checkpoint_dir, "logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        )
        self.tb_drive_writer = None

        drive_log_dir = "/content/drive/MyDrive/AI_Oil_Gas/runs"

        if os.path.exists("/content/drive/MyDrive"):

            os.makedirs(drive_log_dir, exist_ok=True)

            self.tb_drive_writer = SummaryWriter(drive_log_dir)

            logger.info(f"Google Drive TensorBoard: {drive_log_dir}")

        self.drive_model_dir = None
        drive_model_dir = "/content/drive/MyDrive/AI_Oil_Gas/models"
        if os.path.exists("/content/drive/MyDrive"):
            os.makedirs(drive_model_dir, exist_ok=True)

            self.drive_model_dir = drive_model_dir

            logger.info(f"Google Drive models: {drive_model_dir}")

        assert tensorboard_formatter is not None, "TensorboardFormatter must be provided"
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = load_checkpoint(resume, self.model, self.optimizer)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state["best_eval_score"]
            self.num_iterations = state["num_iterations"]
            self.num_epochs = state["num_epochs"]
            self.checkpoint_dir = os.path.split(resume)[0]
        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            load_checkpoint(pre_trained, self.model, None)
            if not self.checkpoint_dir:
                self.checkpoint_dir = os.path.split(pre_trained)[0]

        # use DataParallel if more than 1 GPU available
        if device == TorchDevice.CUDA and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training")


    def fit(self):
        """ for _ in range(self.num_epochs, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info("Stopping criterion is satisfied. Finishing training")
                return

            self.num_epochs += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...") """
        for _ in range(self.num_epochs, self.max_num_epochs):
            self.train()
            logger.info("=" * 80)
            logger.info(f"Epoch {self.num_epochs + 1} finished")
            logger.info("Running validation...")

            self.model.eval()
            eval_score = self.validate()
            self.model.train()

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(eval_score)
            elif self.scheduler is not None:
                self.scheduler.step()

            self._log_lr()

            is_best = self._is_best_eval_score(eval_score)
            self._save_checkpoint(is_best)

            self.num_epochs += 1

        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}")
        if self.tb_drive_writer is not None:
            self.tb_drive_writer.close()

        self.writer.close()

    def train(self):
        """Train one epoch."""

        train_losses = RunningAverage()

        self.eval_criterion.reset()  # обнуляем накопитель метрик в начале эпохи

       
        self.model.train()

        batch_idx = 1

        for t in self.loaders["train"]:

            input, target = _split_and_move_to_device(t, self.device)

            output, loss = self._forward_pass(input, target)

            train_losses.update(loss.item(), self._batch_size(input))

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # вычисляем метрики
            metrics = self.eval_criterion(output, target)   

            if batch_idx % self.log_after_iters == 0 or batch_idx == self.iters_per_epoch:

                pixel_acc = metrics["pixel_acc"]
                mean_class_acc = metrics["mean_class_acc"]
                mean_iou = metrics["mean_iou"]
                mean_iou_all = metrics["mean_iou_all"]

                class_acc = np.array(metrics["class_acc"])
                iou = np.array(metrics["iou"])

                progress = 100.0 * batch_idx / self.iters_per_epoch

                logger.info("=" * 70)
                logger.info(
                    f"Epoch [{self.num_epochs + 1}/{self.max_num_epochs}] "
                    f"Batch [{batch_idx}/{self.iters_per_epoch}] "
                    f"({progress:.1f}%)"
                )
                logger.info(f"Loss                : {train_losses.avg:.5f}")
                logger.info(f"Pixel Accuracy      : {pixel_acc:.5f}")
                logger.info(f"Mean Class Accuracy : {mean_class_acc:.5f}")
                logger.info(f"Mean IoU            : {mean_iou:.5f}")
                logger.info(f"Mean IoU (all)      : {mean_iou_all:.5f}")
                logger.info(f"Class Accuracy      : {np.round(class_acc, 4)}")
                logger.info(f"IoU                 : {np.round(iou, 4)}")
                logger.info("=" * 70)

                

            self.num_iterations += 1
            batch_idx += 1
        pixel_acc = metrics["pixel_acc"]
        mean_class_acc = metrics["mean_class_acc"]
        mean_iou = metrics["mean_iou"]

        class_acc = np.array(metrics["class_acc"])
        iou = np.array(metrics["iou"])
        self._log_stats(
                        "train",
                        train_losses.avg,
                        mean_iou
                        )
        self._log_segmentation_metrics(
                        phase="Train",
                        loss=train_losses.avg,
                        pixel_acc=pixel_acc,
                        mean_class_acc=mean_class_acc,
                        mean_iou=mean_iou,
                        class_acc=class_acc,
                        iou=iou,
                        step=self.num_epochs+1,
                    )
                       
        return False

    def should_stop(self):
        """Check if training should be terminated.

        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case).

        Returns:
            True if training should stop, False otherwise.
        """
        """   if self.max_num_iterations < self.num_iterations:
            logger.info(f"Maximum number of iterations {self.max_num_iterations} exceeded.")
            return True
        """
        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]["lr"]
        if lr < min_lr:
            logger.info(f"Learning rate below the minimum {min_lr}.")
            return True

        return False

    def validate(self):
        logger.info("Validating...")

        val_losses = RunningAverage()

        self.eval_criterion.reset()
        
        with torch.no_grad():
            # select indices of validation samples to log
            rs = np.random.RandomState(42)
            if len(self.loaders["val"]) <= self.max_val_images:
                indices = list(range(len(self.loaders["val"])))
            else:
                indices = rs.choice(len(self.loaders["val"]), size=self.max_val_images, replace=False)

            images_for_logging = []
            for i, t in enumerate(tqdm(self.loaders["val"])):
                input, target = _split_and_move_to_device(t, self.device)

                output, loss = self._forward_pass(input, target)
                val_losses.update(loss.item(), self._batch_size(input))
                metrics = self.eval_criterion(output, target)   # накопленные метрики с начала валидации
                # save val images for logging
                if i in indices and len(images_for_logging) < self.max_val_images:
                    imgs = (
                        input.cpu().numpy(),
                        target.cpu().numpy(),
                        output.cpu().numpy()
                    )
                    images_for_logging.append(imgs + (i,))

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            # log images in a separate thread
            with ThreadPoolExecutor() as executor:
                for input, target, output, i in images_for_logging:
                    executor.submit(self._log_images, input, target, output, f"val_{i}_")

            """ logger.info(f"Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}")"""
            pixel_acc = metrics["pixel_acc"]
            mean_class_acc = metrics["mean_class_acc"]
            mean_iou = metrics["mean_iou"]
            mean_iou_all = metrics["mean_iou_all"]

            class_acc = np.array(metrics["class_acc"])
            iou = np.array(metrics["iou"])

            logger.info("")
            logger.info("=" * 70)
            logger.info(f"Validation Epoch {self.num_epochs + 1}")
            logger.info(f"Loss                : {val_losses.avg:.5f}")
            logger.info(f"Pixel Accuracy      : {pixel_acc:.5f}")
            logger.info(f"Mean Class Accuracy : {mean_class_acc:.5f}")
            logger.info(f"Mean IoU            : {mean_iou:.5f}")
            logger.info(f"Mean IoU (all)      : {mean_iou_all:.5f}")
            logger.info(f"Class Accuracy      : {np.round(class_acc,4)}")
            logger.info(f"IoU                 : {np.round(iou,4)}")
            logger.info("=" * 70)

            self._log_stats("val", val_losses.avg, mean_iou)
            self._log_segmentation_metrics(
                phase="Validation",
                loss=val_losses.avg,
                pixel_acc=pixel_acc,
                mean_class_acc=mean_class_acc,
                mean_iou=mean_iou,
                class_acc=class_acc,
                iou=iou,
                step=self.num_epochs + 1,
            )

            return mean_iou

    def _forward_pass(self, inp: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if is_model_2d(self.model):
            # remove the singleton z-dimension from the input
            inp = torch.squeeze(inp, dim=-3)
            # forward pass
            output, logits = self.model(inp, return_logits=True)
            # add the singleton z-dimension to the output
            output = torch.unsqueeze(output, dim=-3)
            logits = torch.unsqueeze(logits, dim=-3)
        else:
            # forward pass
            output, logits = self.model(inp, return_logits=True)

        # always compute the loss using logits
        loss = self.loss_criterion(logits, target)

        # return probabilities and loss
        return output, loss

    def _is_best_eval_score(self, eval_score: float) -> bool:
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f"Saving new best evaluation metric: {eval_score}")
            self.best_eval_score = eval_score

        return is_best
    def _save_checkpoint(self, is_best: bool):
        # remove `module` prefix from layer names when using nn.DataParallel
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        last_file_path = os.path.join(
            self.checkpoint_dir,
            "last_checkpoint.pytorch"
        )

        logger.info(f"Saving checkpoint to '{last_file_path}'")

        save_checkpoint(
            {
                "num_epochs": self.num_epochs + 1,
                "num_iterations": self.num_iterations,
                "model_state_dict": state_dict,
                "best_eval_score": self.best_eval_score,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            is_best,
            checkpoint_dir=self.checkpoint_dir,
        )

        # -------------------------------------------------------
        # Copy checkpoints to Google Drive if it is available
        # -------------------------------------------------------
        if self.drive_model_dir is not None:

            try:
                shutil.copy2(
                    os.path.join(self.checkpoint_dir, "last_checkpoint.pytorch"),
                    os.path.join(self.drive_model_dir, "last_checkpoint.pytorch")
                )

                if is_best:
                    shutil.copy2(
                        os.path.join(self.checkpoint_dir, "best_checkpoint.pytorch"),
                        os.path.join(self.drive_model_dir, "best_checkpoint.pytorch")
                    )

                logger.info(
                    f"Checkpoint copied to Google Drive: {self.drive_model_dir}"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to copy checkpoint to Google Drive: {e}"
                )

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("learning_rate", lr, self.num_iterations)

    def _add_scalar(self, tag, value, step):
        """
        Записывает scalar одновременно
        в локальный TensorBoard и,
        если подключен Google Drive,
        то ещё и туда.
        """

        self.writer.add_scalar(tag, value, step)

        if self.tb_drive_writer is not None:
            self.tb_drive_writer.add_scalar(tag, value, step)
    def _log_segmentation_metrics(
        self,
        phase,
        loss,
        pixel_acc,
        mean_class_acc,
        mean_iou,
        class_acc,
        iou,
        step,
    ):
        """
        Полностью логирует все метрики сегментации.
        """

        self._add_scalar(
            f"{phase}/Loss",
            loss,
            step
        )

        self._add_scalar(
            f"{phase}/PixelAccuracy",
            pixel_acc,
            step
        )

        self._add_scalar(
            f"{phase}/MeanClassAccuracy",
            mean_class_acc,
            step
        )

        self._add_scalar(
            f"{phase}/MeanIoU",
            mean_iou,
            step
        )

        for c, value in enumerate(class_acc):

            self._add_scalar(
                f"{phase}/ClassAccuracy/{c}",
                float(value),
                step
            )

        for c, value in enumerate(iou):

            self._add_scalar(
                f"{phase}/IoU/{c}",
                float(value),
                step
            )

    def _log_stats(self, phase: str, loss_avg: float, eval_score_avg: float):
        tag_value = {f"{phase}_loss_avg": loss_avg, f"{phase}_eval_score_avg": eval_score_avg}

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info("Logging model parameters and gradients")
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + "/grad", value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input: np.ndarray, target: np.ndarray, prediction: np.ndarray, prefix: str):
        inputs_map = {"inputs": input, "targets": target, "predictions": prediction}
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list | tuple):
                for i, b in enumerate(batch):
                    img_sources[f"{name}{i}"] = b
            else:
                img_sources[name] = batch

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations)

    @staticmethod
    def _batch_size(input: torch.Tensor) -> int:
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
