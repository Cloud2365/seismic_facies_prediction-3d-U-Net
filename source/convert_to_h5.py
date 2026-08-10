import os
import numpy as np
import h5py
from PIL import Image
import re

# ===== НАСТРОЙКИ =====
DATA_DIR = r"E:\Projects\AI_Oil_Gas\seismic_facies_prediction\data\row"
CROSSLINE_DIR = os.path.join(DATA_DIR, "crosslines")
MASK_DIR = os.path.join(DATA_DIR, "masks")
OUTPUT_DIR = r"E:\Projects\AI_Oil_Gas\seismic_facies_prediction\pytorch-3dunet-master\my_data"
# =======================

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
    mask_nums = [get_number_from_filename(f, r"crossline_(\d+)") for f in mask_files]
    
    common_nums = sorted(set(crossline_nums) & set(mask_nums))
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
            print(f"Ошибка при загрузке {crossline_path}: {e}")
            continue
        
        if seismic.ndim == 3:
            seismic = seismic[:, :, 0]
        seismic_stack.append(seismic)
        
        # Загружаем маску
        mask_path = os.path.join(MASK_DIR, mask_file)
        try:
            mask = np.array(Image.open(mask_path), dtype=np.int64)
        except Exception as e:
            print(f"Ошибка при загрузке {mask_path}: {e}")
            continue
        
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_stack.append(mask)
        
        if (i + 1) % 50 == 0:
            print(f"Загружено {i+1}/{len(common_nums)}")
    
    if len(seismic_stack) == 0:
        print("Ошибка: не загружено ни одного файла!")
        return np.array([]), np.array([])
    
    seismic_volume = np.stack(seismic_stack, axis=0)
    mask_volume = np.stack(mask_stack, axis=0)
    print(f"Сейсмика: {seismic_volume.shape}, Маски: {mask_volume.shape}")
    return seismic_volume, mask_volume

if __name__ == "__main__":
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