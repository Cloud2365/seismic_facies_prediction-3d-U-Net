import argparse
import os
import numpy as np
import h5py
from PIL import Image
import re

# Пути задаются аргументами командной строки (см. parse_args); значения ниже
# заполняются в __main__.
CROSSLINE_DIR = None
MASK_DIR = None


def parse_args():
    parser = argparse.ArgumentParser(description="Сборка all_data.h5 из crossline TIFF и масок PNG")
    parser.add_argument("--data_dir", required=True, help="Папка с подпапками crosslines/ и masks/")
    parser.add_argument("--output_dir", required=True, help="Куда записать all_data.h5")
    return parser.parse_args()

def get_number_from_filename(filename, pattern):
    """Извлекает номер из имени файла по шаблону."""
    match = re.search(pattern, filename)
    return int(match.group(1)) if match else None

def is_valid_image_file(filename):
    """Проверяет, что файл не является служебным и имеет нужное расширение."""
    if filename.startswith('._') or filename.startswith('~'):
        return False
    if filename.startswith('_'):
        return False
    return True

def load_crossline_data():
    all_crossline_files = [f for f in os.listdir(CROSSLINE_DIR) if is_valid_image_file(f)]
    crossline_files = [
        f for f in all_crossline_files 
        if f.endswith('.tiff') and get_number_from_filename(f, r"crossline_(\d+)") is not None
    ]
    crossline_files.sort(key=lambda x: get_number_from_filename(x, r"crossline_(\d+)"))
    
    all_mask_files = [f for f in os.listdir(MASK_DIR) if is_valid_image_file(f)]
    mask_files = [
        f for f in all_mask_files 
        if f.endswith('.png') and get_number_from_filename(f, r"crossline_(\d+)") is not None
    ]
    mask_files.sort(key=lambda x: get_number_from_filename(x, r"crossline_(\d+)"))
    
    print(f"Найдено кросслайнов: {len(crossline_files)}")
    print(f"Найдено масок для кросслайнов: {len(mask_files)}")
    
    if not crossline_files or not mask_files:
        print("Ошибка: не найдены кросслайны или маски к ним!")
        return np.array([]), np.array([])
    
    crossline_nums = [get_number_from_filename(f, r"crossline_(\d+)") for f in crossline_files]
    if len(set(crossline_nums)) != len(crossline_nums):
        raise ValueError("Повторяющиеся номера crossline в именах файлов")
    mask_nums = [get_number_from_filename(f, r"crossline_(\d+)") for f in mask_files]
    
    common_nums = sorted(set(crossline_nums) & set(mask_nums))
    unpaired = sorted(set(crossline_nums) ^ set(mask_nums))
    if unpaired:
        print(f"ВНИМАНИЕ: {len(unpaired)} crossline без пары (сейсмика или маска), например: {unpaired[:10]}")
    if common_nums and common_nums[-1] - common_nums[0] + 1 != len(common_nums):
        print("ВНИМАНИЕ: номера crossline идут с пропусками; индексы split в run_3d_training.py "
              "будут относиться к позициям в кубе, а не к номерам crossline")
    print(f"Совпадающих номеров: {len(common_nums)}")
    
    if len(common_nums) == 0:
        print("Примеры номеров из кросслайнов:", crossline_nums[:10])
        print("Примеры номеров из масок:", mask_nums[:10])
        raise ValueError("Нет ни одного совпадающего кросслайна!")
    
    # --- Загружаем данные ---
    seismic_stack = []
    mask_stack = []
    
    for i, num in enumerate(common_nums):
        crossline_file = [f for f in crossline_files if get_number_from_filename(f, r"crossline_(\d+)") == num][0]
        mask_file = [f for f in mask_files if get_number_from_filename(f, r"crossline_(\d+)") == num][0]
        
        # Загружаем сейсмику
        crossline_path = os.path.join(CROSSLINE_DIR, crossline_file)
        try:
            seismic = np.array(Image.open(crossline_path), dtype=np.float32)
        except Exception as e:
            # Пропуск среза молча сдвинул бы все индексы split ([0:619], [809:951] ...)
            raise RuntimeError(f"Ошибка при загрузке {crossline_path}: {e}") from e

        if seismic.ndim == 3:
            seismic = seismic[:, :, 0]

        # Загружаем маску
        mask_path = os.path.join(MASK_DIR, mask_file)
        try:
            mask = np.array(Image.open(mask_path), dtype=np.int64)
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке {mask_path}: {e}") from e

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Проверяем соответствие размеров
        if seismic.shape != mask.shape:
            raise ValueError(
                f"Размеры seismic и mask не совпадают для crossline {num}: "
                f"seismic={seismic.shape}, mask={mask.shape}"
            )

        # Добавляем только полностью согласованную пару
        seismic_stack.append(seismic)
        mask_stack.append(mask)
        if (i + 1) % 50 == 0:
            print(f"Загружено {i+1}/{len(common_nums)}")
    
    if len(seismic_stack) == 0:
        print("Ошибка: не загружено ни одного файла!")
        return np.array([]), np.array([])
    
    seismic_volume = np.stack(seismic_stack, axis=0)
    mask_volume = np.stack(mask_stack, axis=0)
    if seismic_volume.shape != mask_volume.shape:
        raise ValueError(
            f"Итоговые размеры seismic и mask не совпадают: "
            f"{seismic_volume.shape} vs {mask_volume.shape}"
        )
    print(f"Сейсмика: {seismic_volume.shape}, Маски: {mask_volume.shape}")
    return seismic_volume, mask_volume

if __name__ == "__main__":
    args = parse_args()
    CROSSLINE_DIR = os.path.join(args.data_dir, "crosslines")
    MASK_DIR = os.path.join(args.data_dir, "masks")
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Начинаю загрузку данных (кросслайны)...")
    seismic, masks = load_crossline_data()
    
    if seismic.size == 0:
        print("Ошибка: данные не загружены. Завершаю.")
        exit(1)
    
    all_path = os.path.join(OUTPUT_DIR, 'all_data.h5')
    with h5py.File(all_path, 'w') as f:
        f.create_dataset('raw', data=seismic, compression='gzip')
        f.create_dataset('label', data=masks, compression='gzip')
    
    print(f"Сохранён {all_path}")
    print("ГОТОВО!")