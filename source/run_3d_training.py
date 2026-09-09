import argparse
import os
import yaml
import subprocess
import numpy as np
import h5py


def compute_class_weights(labels, n_classes):
    class_counts = np.bincount(labels.flatten(), minlength=n_classes)
    total = np.sum(class_counts)
    weights = [total / (c + 1e-6) for c in class_counts]
    weights = np.array(weights) / np.sum(weights) * n_classes
    return weights.tolist()
    

def generate_config(args, train_path, val_path, class_weights=None):
    config = {
        'loaders': {
            'train': {
                'file_paths': [train_path],
                'slice_builder': {
                    'ndim': 3,  
                    'name': 'SliceBuilder',
                    'patch_shape': args.patch_shape,
                    'stride_shape': args.stride_shape
                },
               'transformer': {
                    'raw': [
                        {'name': 'Normalize'}
                    ],
                    'label': []
                }
            },
            'val': {
                'file_paths': [val_path],
                'slice_builder': {
                    'ndim': 3, 
                    'name': 'SliceBuilder',
                    'patch_shape': args.patch_shape,
                    'stride_shape': args.stride_shape
                },
                'transformer': {
                     'raw': [
                        {'name': 'Normalize'},
                        {
                            'name': 'ToTensor',
                            'expand_dims': True
                        }
                    ],
                    'label': [
                        {
                            'name': 'ToTensor',
                            'expand_dims': False,
                            'dtype': 'long'
                        }
                    ] 
                }
            },
            'num_workers': args.num_workers,
            'batch_size': args.batch_size
        },
        'model': {
            'name': args.arch,
            'in_channels': 1,
            'out_channels': args.n_classes,
            'f_maps': args.f_maps,
            'layer_order': 'gcr',
            'num_groups': 8,
            'is_segmentation': True
        },
        'loss': {
            'name': args.loss
        },
        'optimizer': {
            'name': 'Adam',
            'learning_rate': args.lr,
            'weight_decay': 0.00001
        },
        'trainer': {
            'checkpoint_dir': args.checkpoint_dir,
            'max_num_iterations': None,
            'max_num_epochs': args.n_epoch, 
            'resume': args.resume,
        },
        'eval_metric': {
            'name': 'SegmentationMetrics',
            'n_classes': args.n_classes
            },
        'logging': {
            'log_dir': args.log_dir,
            'tensorboard': True
        }
    }

    if args.loss == 'CrossEntropyLoss' and args.class_weights and class_weights is not None:
        config['loss']['weight'] = class_weights
    
    # =========================================================
    # AUGMENTATION
    # =========================================================

    if args.aug:

        raw_aug_list = [
            {
                'name': 'RandomRotate',
                'angle_spectrum': args.rot_angle,
                'order': 1,
                'axes': [[1, 2]]
            },
            {
                'name': 'RandomFlip'
            }
        ]

        label_aug_list = [
            {
                'name': 'RandomRotate',
                'angle_spectrum': args.rot_angle,
                'order': 0,
                'axes': [[1, 2]]
            },
            {
                'name': 'RandomFlip'
            }
        ]

        config['loaders']['train']['transformer']['raw'].extend(
            raw_aug_list
        )

        config['loaders']['train']['transformer']['label'].extend(
            label_aug_list
        )

    # ToTensor должен выполняться ПОСЛЕ augmentation
    config['loaders']['train']['transformer']['raw'].append(
        {
            'name': 'ToTensor',
            'expand_dims': True
        }
    )

    config['loaders']['train']['transformer']['label'].append(
        {
            'name': 'ToTensor',
            'expand_dims': False,
            'dtype': 'long'
        }
    )

    
    os.makedirs(args.config_dir, exist_ok=True)
    config_path = os.path.join(args.config_dir, 'generated_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path

def main():
    parser = argparse.ArgumentParser(description='3D U-Net training with dynamic split')
    parser.add_argument('--data_h5', type=str,
                        default='/content/collab_project/pytorch-3dunet-master/my_data/all_data.h5',
                        help='Path to all_data.h5')
    parser.add_argument('--per_val', type=float, default=0.15,
                        help='Fraction for validation (from remaining after test)')
    parser.add_argument('--per_test', type=float, default=0.15,
                        help='Fraction for test (held out)')
    parser.add_argument('--arch', type=str, default='UNet3D',
                        choices=['UNet3D', 'ResidualUNet3D', 'ResidualUNetSE3D'])
    parser.add_argument(
    '--split',
    type=str,
    default='spatial',
    choices=['spatial', 'random'],
    help='Dataset split method'
    )
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--f_maps', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patch_shape', type=int, nargs=3, default=[16, 128, 128])
    parser.add_argument('--stride_shape', type=int, nargs=3, default=[8, 64, 64])
    parser.add_argument('--aug', action='store_true', help='Enable augmentation')
    parser.add_argument('--rot_angle', type=float, default=10.0)
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss',
                        choices=['GeneralizedDiceLoss', 'CrossEntropyLoss'])
    parser.add_argument('--class_weights', action='store_true',
                        help='Compute and use class weights (only for CrossEntropyLoss)')
                        
    parser.add_argument('--checkpoint_dir', type=str, default='/content/checkpoints')
    parser.add_argument('--log_dir', type=str, default='/content/logs')
    parser.add_argument('--config_dir', type=str, default='/content/my_configs')
    parser.add_argument('--temp_dir', type=str, default='/content/temp_splits')
    parser.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    # Создаём папки
    for d in [args.temp_dir, args.checkpoint_dir, args.log_dir, args.config_dir]:
        os.makedirs(d, exist_ok=True)

    # Загружаем данные
    print(f"Загрузка {args.data_h5} ...")
    with h5py.File(args.data_h5, 'r') as f:
        raw = f['raw'][:]
        label = f['label'][:]
    n = raw.shape[0]
    print(f"Всего срезов: {n}")

    # =========================================================
    # SPLIT DATASET
    # =========================================================

    test_size = int(n * args.per_test)
    val_size = int(n * args.per_val)
    train_size = n - test_size - val_size


    if args.split == 'spatial':

        # -----------------------------------------------------
        # SPATIAL SPLIT
        # -----------------------------------------------------
        # Срезы идут последовательно.
        # Соседние срезы остаются в одном split.

        train_idx = np.arange(
            0,
            train_size
        )

        val_idx = np.arange(
            train_size,
            train_size + val_size
        )

        test_idx = np.arange(
            train_size + val_size,
            n
        )

        print()
        print("=" * 70)
        print("SPATIAL SPLIT")
        print("=" * 70)

        print(
            f"Train: {len(train_idx)} "
            f"[0:{train_size}]"
        )

        print(
            f"Val:   {len(val_idx)} "
            f"[{train_size}:{train_size + val_size}]"
        )

        print(
            f"Test:  {len(test_idx)} "
            f"[{train_size + val_size}:{n}]"
        )


    elif args.split == 'random':

        # -----------------------------------------------------
        # RANDOM SPLIT
        # -----------------------------------------------------
        # Все срезы перемешиваются.
        # Затем случайно выбираются train / val / test.

        rng = np.random.default_rng(42)

        indices = np.arange(n)

        rng.shuffle(indices)

        train_idx = indices[
            :train_size
        ]

        val_idx = indices[
            train_size:
            train_size + val_size
        ]

        test_idx = indices[
            train_size + val_size:
        ]

        print()
        print("=" * 70)
        print("RANDOM SPLIT")
        print("=" * 70)

        print(
            f"Train: {len(train_idx)}"
        )

        print(
            f"Val:   {len(val_idx)}"
        )

        print(
            f"Test:  {len(test_idx)}"
        )

        # -----------------------------------------------------
        # Проверяем пересечения
        # -----------------------------------------------------

        train_set = set(
            train_idx.tolist()
        )

        val_set = set(
            val_idx.tolist()
        )

        test_set = set(
            test_idx.tolist()
        )

        print()
        print("Checking intersections:")

        print(
            "Train ∩ Val:",
            len(train_set & val_set)
        )

        print(
            "Train ∩ Test:",
            len(train_set & test_set)
        )

        print(
            "Val ∩ Test:",
            len(val_set & test_set)
        )
    if args.split == 'random':
        args.temp_dir = os.path.join(
            args.temp_dir,
            'random'
            )
    else:
        args.temp_dir = os.path.join(
             args.temp_dir,
            'spatial'
            )
    # Сохраняем временные train и val
    train_path = os.path.join(args.temp_dir, 'train.h5')
    val_path = os.path.join(args.temp_dir, 'val.h5')
    test_path = os.path.join(args.temp_dir, 'test.h5')
    with h5py.File(train_path, 'w') as f:
        f.create_dataset('raw', data=raw[train_idx], compression='gzip')
        f.create_dataset('label', data=label[train_idx], compression='gzip')
    with h5py.File(val_path, 'w') as f:
        f.create_dataset('raw', data=raw[val_idx], compression='gzip')
        f.create_dataset('label', data=label[val_idx], compression='gzip')
    
    with h5py.File(test_path, 'w') as f:
        f.create_dataset('raw', data=raw[test_idx], compression='gzip')
        f.create_dataset('label', data=label[test_idx], compression='gzip')


    # Веса классов
    class_weights_list = None
    if args.class_weights and args.loss == 'CrossEntropyLoss':
        print("Вычисление весов классов...")
        with h5py.File(train_path, 'r') as f:
            train_labels = f['label'][:]
        class_weights_list = compute_class_weights(train_labels, args.n_classes)
        print("Веса:", class_weights_list)

    # Генерация конфига и запуск
        # Генерация конфига и запуск
    config_path = generate_config(args, train_path, val_path, class_weights_list)
    print(f"Конфиг сохранён: {config_path}")
    
    # Запуск через Python-модуль
    import sys
    cmd = [sys.executable, '-m', 'pytorch3dunet.train', '--config', config_path]
    
    # Если указан чекпоинт для возобновления
    if args.resume:
        print(f"Возобновление с чекпоинта: {args.resume}")
    
    print("Запуск:", ' '.join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == '__main__': 
    main()