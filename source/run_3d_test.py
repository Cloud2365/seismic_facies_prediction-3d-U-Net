import argparse
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Добавляем путь к библиотеке
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pytorch-3dunet-master')))

from pytorch3dunet.datasets.hdf5 import StandardHDF5Dataset
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.utils import get_logger, TensorboardFormatter
from sklearn.metrics import confusion_matrix
from PIL import Image

logger = get_logger("TestEvaluator")

# ---------------------------------------------------------
# Цветовая палитра как в trainer
# ---------------------------------------------------------

SEISMIC_COLORS = np.asarray([
    [0, 0, 0],
    [69, 117, 180],
    [145, 191, 219],
    [224, 243, 248],
    [254, 224, 144],
    [252, 141, 89],
    [215, 48, 39],
    [100, 100, 100],
    [200, 200, 200],
    [255, 255, 255]
], dtype=np.uint8)


def colorize_mask(mask):
    """
    mask -> RGB
    """
    rgb = SEISMIC_COLORS[mask.astype(np.uint8)]
    return Image.fromarray(rgb)


def normalize_seismic(img):
    """
    Приводим сейсмику к диапазону 0..255
    """
    img = img.astype(np.float32)

    img -= img.min()

    if img.max() > 0:
        img /= img.max()

    img *= 255

    return Image.fromarray(img.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description='Тестирование модели с вычислением всех метрик и визуализацией')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Путь к обученной модели (best_checkpoint.pytorch)')
    parser.add_argument('--test_h5', type=str, default='E:/Projects/AI_Oil_Gas/collab_project/source/temp_splits/test.h5',
                        help='Путь к test.h5 файлу')
    parser.add_argument('--arch', type=str, default='UNet3D',
                        choices=['UNet3D', 'ResidualUNet3D', 'ResidualUNetSE3D'])
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--f_maps', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patch_shape', type=int, nargs=3, default=[16, 128, 128])
    parser.add_argument('--stride_shape', type=int, nargs=3, default=[8, 64, 64])
    parser.add_argument('--log_dir', type=str, default='./test_logs',
                        help='Директория для TensorBoard логов теста')
    parser.add_argument('--max_vis_samples', type=int, default=5,
                        help='Максимальное количество батчей для визуализации')
    args = parser.parse_args()

    # Проверяем, что файлы существуют
    if not os.path.exists(args.model_path):
        print(f"Ошибка: модель не найдена по пути {args.model_path}")
        sys.exit(1)
    if not os.path.exists(args.test_h5):
        print(f"Ошибка: файл с тестовыми данными не найден {args.test_h5}")
        sys.exit(1)

    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Загружаем модель
    model_config = {
        'name': args.arch,
        'in_channels': 1,
        'out_channels': args.n_classes,
        'f_maps': args.f_maps,
        'layer_order': 'gcr',
        'num_groups': 8,
        'is_segmentation': True
    }
    model = get_model(model_config)
    state = torch.load(args.model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in state:
        state_dict = state['model_state_dict']
    else:
        state_dict = state
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print(f"Модель загружена: {args.arch}, параметров: {sum(p.numel() for p in model.parameters())}")

    # Создаём тестовый датасет
    test_dataset = StandardHDF5Dataset(
        file_path=args.test_h5,
        phase='val',
        slice_builder_config={
            'name': 'SliceBuilder',
            'patch_shape': args.patch_shape,
            'stride_shape': args.stride_shape
        },
        transformer_config={
            'raw': [{'name': 'Normalize'}, {'name': 'ToTensor', 'expand_dims': True}],
            'label': [{'name': 'ToTensor', 'expand_dims': False, 'dtype': 'long'}]
        },
        raw_internal_path='raw',
        label_internal_path='label',
        global_normalization=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Создаём объект для вычисления метрик
    eval_metric_config = {
        'name': 'SegmentationMetrics',
        'n_classes': args.n_classes,
        'ignore_background': True
    }
    eval_criterion = get_evaluation_metric({'eval_metric': eval_metric_config})

    # Создаём TensorboardFormatter (как в trainer.py)
    tensorboard_formatter = TensorboardFormatter()

    # Накапливаем метрики
    # Накопитель метрик (глобальный, по всему тесту)
    eval_criterion.reset()
    all_preds = []
    all_targets = []

    # Для визуализации сохраняем несколько батчей
    saved_slices = []

    logger.info("Начинаем тестирование...")
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tqdm(test_loader)):
            input = input.to(device)
            target = target.to(device)
            output, logits = model(input, return_logits=True)
            pred = torch.argmax(output, dim=1)
            all_preds.append(pred.cpu().numpy().flatten())
            all_targets.append(target.cpu().numpy().flatten())
            metrics = eval_criterion(output, target)   # накопленные метрики по всем виденным батчам
            # Сохраняем несколько батчей для визуализации
            if len(saved_slices) < args.max_vis_samples:

                prediction = torch.argmax(output, dim=1)

                saved_slices.append({

                    "raw": input[0].cpu().numpy(),

                    "gt": target[0].cpu().numpy(),

                    "pred": prediction[0].cpu().numpy()

                })

    # Получаем метрики
    pixel_acc = metrics["pixel_acc"]
    mean_class_acc = metrics["mean_class_acc"]
    mean_iou = metrics["mean_iou"]
    class_acc = np.array(metrics["class_acc"])
    iou = np.array(metrics["iou"])
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(args.n_classes)))
    precisions = []
    recalls = []
    f1_scores = []
    for c in range(args.n_classes):
        tp = cm[c,c]
        fp = cm[:,c].sum() - tp
        fn = cm[c,:].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
    mean_precision = np.mean(precisions)
    mean_f1_score = np.mean(f1_scores)

    # Вывод в консоль
    print("\n" + "=" * 70)
    print("Результаты на тестовой выборке:")
    print(f"Pixel Accuracy      : {pixel_acc:.5f}")
    print(f"Mean Class Accuracy : {mean_class_acc:.5f}")
    print(f"Mean IoU            : {mean_iou:.5f}")
    print(f"Mean Precision      : {mean_precision:.5f}")
    print(f"Mean-F1-Score       : {mean_f1_score:.5f}")
    print(f"Class Accuracy      : {np.round(class_acc, 4)}")
    print(f"IoU         : {np.round(iou, 4)}")
    print(f"Precision   : {np.round(precisions, 4)}")
    print(f"Recall      : {np.round(recalls, 4)}")
    print(f"F1-Score    : {np.round(f1_scores, 4)}")
    print("=" * 70 + "\n")

    # Запись метрик в TensorBoard
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    writer.add_scalar('Test/PixelAccuracy', pixel_acc, 0)
    writer.add_scalar('Test/MeanClassAccuracy', mean_class_acc, 0)
    writer.add_scalar('Test/MeanIoU', mean_iou, 0)
    for c, val in enumerate(class_acc):
        writer.add_scalar(f'Test/ClassAccuracy/class_{c}', val, 0)
    for c, val in enumerate(iou):
        writer.add_scalar(f'Test/IoU/class_{c}', val, 0)
    for c, val in enumerate(precisions):
        writer.add_scalar(f'Test/Precision/class_{c}', val, 0)
    for c, val in enumerate(recalls):
        writer.add_scalar(f'Test/Recall/class_{c}', val, 0)
    for c, val in enumerate(f1_scores):
        writer.add_scalar(f'Test/F1/class_{c}', val, 0)

    writer.close()

    #############################################################
    # СОХРАНЕНИЕ КАРТИНОК
    #############################################################

    image_root = os.path.join(args.log_dir, "Test")

    os.makedirs(image_root, exist_ok=True)

    print("Сохраняем изображения...")

    for idx, sample in enumerate(saved_slices):

        folder = os.path.join(image_root, f"slice_{idx:03d}")

        os.makedirs(folder, exist_ok=True)

        raw = sample["raw"][0]

        gt = sample["gt"]

        pred = sample["pred"]

        #####################################################
        # берём центральный слой патча
        #####################################################

        center = raw.shape[0] // 2

        raw = raw[center]

        gt = gt[center]

        pred = pred[center]

        #####################################################

        normalize_seismic(raw).save(

            os.path.join(folder, "seismic.png")

        )

        colorize_mask(gt).save(

            os.path.join(folder, "ground_truth.png")

        )

        colorize_mask(pred).save(

            os.path.join(folder, "prediction.png")

        )

    print("Изображения сохранены.")

if __name__ == '__main__':
    main()