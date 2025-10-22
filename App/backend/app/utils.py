from PIL import Image
import torch
import numpy as np
import pandas as pd
from orix.quaternion import Orientation, symmetry
from orix.plot import IPFColorKeyTSL
from orix.vector import Vector3d

#---------------------------------Функция построения IPF карт------------------------------------
async def process_IPF(in_path, out_path, type):
    try:
        df, size_x, size_y, step_x, step_y = load_EBSD_file(in_path)

        create_IPF_image(df, type, out_path, size_x, size_y)
    except Exception as e:
        raise Exception(f"IPF failed: {str(e)}")
#------------------------------------------------------------------------------------------------

#---------------------------------Функция построения простого изображения по углам эйлера------------------------------------
async def process_eulers(in_path, out_path):
    df, size_x, size_y, step_x, step_y = load_EBSD_file(in_path)

    create_euler_image(df, out_path, size_x, size_y)
    return step_x, step_y
#------------------------------------------------------------------------------------------------

#---------------------------------Функция реконструкции углов эйлера------------------------------------
async def process_euler_reconstraction(in_path, out_path, model_path, window_padding, y_step = 4, x_step = 4, resized = False):

    generator, device = initialize_model(model_path)
    
    # load file and convert to tensor
    if (resized):
        df, size_x, size_y, step_x, step_y = get_full_interlaced(in_path, y_step, x_step)
    else:
        df, size_x, size_y, step_x, step_y = load_EBSD_file(in_path)
    image, mask = convert_EBSD_to_tensor(df, size_x, size_y)

    print("Функция реконструкции углов запущена...",size_x,size_y,step_x,step_y)

    _, h, w = image.shape
    image = (image*2 - 1.).to(device)
    image_inpainted = image.clone()
    mask = mask.to(device)
    mask_new = torch.zeros_like(mask)

    window_size_mask = 32

    for i in range(0, h, window_size_mask):
        for j in range(0, w, window_size_mask):
            # Get current window coordinates
            i_end = i + window_size_mask
            j_end = j + window_size_mask

            if i_end > h:
                i_end = h
                i = h - window_size_mask
            if j_end > w:
                j_end = w
                j = w - window_size_mask
    
            mask_window = mask[:, i:i_end, j:j_end]
            num = mask_window.sum()
            if num >= window_size_mask * window_size_mask - 10:
                mask_new[:, i:i_end, j:j_end] = mask_window

    padding = 16
    #mask_new[:, :padding, :] = 0          # Верхняя граница
    #mask_new[:, h-padding:, :] = 0        # Нижняя граница
    #mask_new[:, :, :padding] = 0          # Левая граница
    #mask_new[:, :, w-padding:] = 0        # Правая граница

    # Window parameters
    window_size = 256
    feather = 16
    stride = window_size - window_padding

    generate(h,w,stride,feather,window_size,generator,device,image_inpainted,mask)
    generate(h,w,stride,feather,window_size,generator,device,image_inpainted,mask_new)

    # Normalize and save
    img_out = (image_inpainted.permute(1, 2, 0) + 1) * 180
    img_out = img_out.cpu().numpy() 
    img_np = np.array(img_out)
    flat_data = img_np.reshape(-1, 3)
    df = pd.DataFrame(flat_data, columns=["Euler1", "Euler2", "Euler3"])

    with open(out_path, 'w', newline='') as f:
        f.write(f"FRAGMENT\n{size_x}\n")
        f.write(f"{size_y}\n")
        df.to_csv(f, sep='\t', index=False, lineterminator='\n')
#------------------------------------------------------------------------------------------------------------

#-----------------------------------ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ---------------------------------------
def initialize_model(path_to_model):
    from .models import Generator

    generator_state_dict = torch.load(path_to_model, weights_only=False)['G']
    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available()
                          and use_cuda_if_available else 'cpu')

    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)

    generator.load_state_dict(generator_state_dict, strict=True)
    return generator, device

def generate(h,w,stride,feather,window_size,generator,device, image_inpainted,mask):
    # Определяем целевой размер (256x256)
    target_size = 256
    
    # Если изображение меньше целевого размера, добавляем паддинг
    if h < target_size or w < target_size:
        # Вычисляем необходимое дополнение
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)
        
        # Создаем тензоры с паддингом
        image_padded = torch.zeros(3, h + pad_h, w + pad_w, device=device)
        mask_padded = torch.zeros(1, h + pad_h, w + pad_w, device=device)
        
        # Копируем исходные данные в левый верхний угол
        image_padded[:, :h, :w] = image_inpainted
        mask_padded[:, :h, :w] = mask
        
        # Обновляем переменные для работы с дополненным изображением
        image_inpainted = image_padded
        mask = mask_padded
        h_padded, w_padded = h + pad_h, w + pad_w
    else:
        h_padded, w_padded = h, w
    
    # Используем дополненные размеры в цикле обработки
    for i in range(0, h_padded, stride):
        for j in range(0, w_padded, stride):
            # Get current window coordinates
            i_end = i + window_size
            j_end = j + window_size

            if i_end > h_padded:
                i_end = h_padded
                i = h_padded - window_size
            if j_end > w_padded:
                j_end = w_padded
                j = w_padded - window_size

            # Extract window
            image_window = image_inpainted[:, i:i_end, j:j_end]
            mask_window = mask[:, i:i_end, j:j_end]
            
            # Prepare input
            image_masked = image_window * (1. - mask_window)
            ones_x = torch.ones(1, *image_window.shape[1:], device=device)
            x = torch.cat([
                image_masked.unsqueeze(0),
                ones_x.unsqueeze(0),
                (ones_x * mask_window).unsqueeze(0)
            ], dim=1)
            
            # Process window
            with torch.inference_mode():
                _, x_stage2 = generator(x, mask_window.unsqueeze(0))
            
            # Blend result
            window_result = image_window.unsqueeze(0) * (1. - mask_window.unsqueeze(0)) + x_stage2 * mask_window.unsqueeze(0)
            
            # Create weight matrix
            weight = torch.ones(1, 1, *image_window.shape[1:], device=device)

            # Add to output
            h_start, h_end = i, min(i + window_size, h_padded)
            w_start, w_end = j, min(j + window_size, w_padded)
            
            image_inpainted[:, h_start:h_end, w_start:w_end] = (
                image_inpainted[:, h_start:h_end, w_start:w_end] * (1 - mask_window) + 
                window_result[0, :, :h_end-h_start, :w_end-w_start] * mask_window
            )
            mask[:, i:i_end, j:j_end] = 0
    
    # Обрезаем результат до исходного размера
    if h < target_size or w < target_size:
        image_inpainted = image_inpainted[:, :h, :w]
        mask = mask[:, :h, :w]
    
    return image_inpainted, mask

def load_EBSD_file(path_to_file):
    size_x = 0
    size_y = 0
    step_x = 0
    step_y = 0

    # Определяем тип файла по расширению
    try:
        file_extension = path_to_file.lower().split('.')[-1]
    except AttributeError:
        # Если path_to_file не строка или нет метода lower()
        file_extension = str(path_to_file).split('.')[-1].lower()
    except IndexError:
        # Если нет расширения (файл без точки)
        print(f"Предупреждение: не удалось определить расширение файла {path_to_file}")
        file_extension = 'txt'  # значение по умолчанию
    
    with open(path_to_file, 'r',encoding="cp1251") as f:
        first_line = f.readline().strip()
        isFragment = (first_line == "FRAGMENT")
    
    if isFragment: # Фрагменты EBSD (из датасета)
        if file_extension in ['ctf', 'txt']:
            df = pd.read_csv(path_to_file, sep='\t', skiprows=3)
        elif file_extension == 'xlsx':
            df = pd.read_excel(path_to_file, skiprows=3)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_extension}")
            
        with open(path_to_file, 'r') as f:
            f.readline()
            size_x = int(f.readline())
            size_y = int(f.readline())
    else: # Файлы EBSD
        if file_extension in ['ctf', 'txt']:
            with open(path_to_file, 'r') as f:
                line_number = 0
                for line in f:
                    line_number += 1
                    if line.strip().startswith("XCells"):
                        size_x = int(line.strip().split()[1])
                    if line.strip().startswith("YCells"):
                        size_y = int(line.strip().split()[1])
                    if all(col in line for col in ["Euler1", "Euler2", "Euler3"]):
                        header_line_number = line_number
                        break
            df = pd.read_csv(path_to_file, sep='\t', skiprows=header_line_number-1)
        elif file_extension == 'xlsx':
            # Для xlsx файлов ищем заголовок и метаданные по-другому
            temp_df = pd.read_excel(path_to_file, header=None)
            
            # Ищем строки с XCells и YCells
            for idx, row in temp_df.iterrows():
                if row.astype(str).str.contains('XCells').any():
                    cell_value = row.astype(str)
                    xcells_line = cell_value[cell_value.str.contains('XCells')].iloc[0]
                    size_x = int(xcells_line.split()[1])
                if row.astype(str).str.contains('YCells').any():
                    cell_value = row.astype(str)
                    ycells_line = cell_value[cell_value.str.contains('YCells')].iloc[0]
                    size_y = int(ycells_line.split()[1])
                # Ищем строку с заголовком
                if all(col in row.astype(str).values for col in ["Euler1", "Euler2", "Euler3"]):
                    header_line_number = idx
                    break
            
            df = pd.read_excel(path_to_file, skiprows=header_line_number)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_extension}")

    # Переименование столбцов если такие имеются
    column_rename_map = {}
    if 'Xpos' in df.columns:
        column_rename_map['Xpos'] = 'X'
    if 'Ypos' in df.columns:
        column_rename_map['Ypos'] = 'Y'
    if 'Euler1(°)' in df.columns:
        column_rename_map['Euler1(°)'] = 'Euler1'
    if 'Euler2(°)' in df.columns:
        column_rename_map['Euler2(°)'] = 'Euler2'
    if 'Euler3(°)' in df.columns:
        column_rename_map['Euler3(°)'] = 'Euler3'
    
    if column_rename_map:
        df = df.rename(columns=column_rename_map)
        print(f"Переименованы столбцы: {column_rename_map}")

    if 'X' in df.columns and 'Y' in df.columns:
        if size_x == 0 or size_y == 0:  
            # Рассчитываем size_x
            last_xpos = int(df['X'].iloc[-1])
            prev_xpos = int(df['X'].iloc[-2])
            size_x = int(last_xpos / (last_xpos - prev_xpos)) + 1
            # Рассчитываем size_y
            last_ypos = int(df['Y'].iloc[-1])
            size_y = int(last_ypos / (last_xpos - prev_xpos)) + 1
        # Рассчитываем шаг по x и y
        step_x = df["X"][1] - df["X"][0]
        step_y = df["Y"][size_x] - df["Y"][0]
    else:
        print(f"{path_to_file}: X или Y координаты отсутствуют")
 
    try:
        df['Euler1'] = pd.to_numeric(df['Euler1'].str.replace(',', '.'), errors='coerce')
        df['Euler2'] = pd.to_numeric(df['Euler2'].str.replace(',', '.'), errors='coerce')
        df['Euler3'] = pd.to_numeric(df['Euler3'].str.replace(',', '.'), errors='coerce')
    except AttributeError:
        print(f"{path_to_file}: Замена запятых на точки не требуется")
    
    return df, size_x, size_y, step_x, step_y

def convert_EBSD_to_tensor(df, size_x, size_y):
    df = df[['Euler1', 'Euler2', 'Euler3']]
    data = df.values.astype(np.float32)
    image_data = data.reshape(size_y, size_x, 3)
    image = torch.from_numpy(image_data).permute(2, 0, 1) / 360.0  # [3, H, W]
    mask_data = np.all(image_data == 0, axis=2).astype(np.float32)
    mask = torch.from_numpy(mask_data).unsqueeze(0)  # [1, H, W]
    return image, mask

def create_euler_image(df, out_path, size_x, size_y):

    euler_angles = df[["Euler1", "Euler2", "Euler3"]].values

    try:
        image = Image.fromarray((euler_angles.reshape(size_y, size_x, 3) / 360 * 255).astype('uint8'))
        image.save(out_path)
        print(f"Изображение успешно сохранено!\n")
    except Exception as e:
        print(f"Ошибка при сохранении изображения: {e}\n")

def create_IPF_image(df, type, out_path, size_x, size_y):

    euler_angles = torch.tensor(df[["Euler1", "Euler2", "Euler3"]].values, device='cuda')

    if type == 0:
        colors = get_colors_IPF_X(euler_angles)
    elif type == 1:
        colors = get_colors_IPF_Y(euler_angles)
    elif type == 2:
        colors = get_colors_IPF_Z(euler_angles)

    colors_cpu = colors.cpu().numpy()

    try:
        image = Image.fromarray((colors_cpu.reshape(size_y, size_x, 3) * 255).astype('uint8'))
        image.save(out_path)
        print(f"Изображение успешно сохранено!\n")
    except Exception as e:
        print(f"Ошибка при сохранении изображения: {e}\n")

def get_colors_IPF_X(euler_angles):
    solution_vectors = Orientation.from_euler(euler_angles.cpu().numpy(), degrees=True)
    ipfkey_x = IPFColorKeyTSL(symmetry.Oh, direction=Vector3d.xvector())
    rgb_x = ipfkey_x.orientation2color(solution_vectors)
    return torch.tensor(rgb_x, device='cuda')

def get_colors_IPF_Y(euler_angles):
    solution_vectors = Orientation.from_euler(euler_angles.cpu().numpy(), degrees=True)
    ipfkey_y = IPFColorKeyTSL(symmetry.Oh, direction=Vector3d.yvector())
    rgb_y = ipfkey_y.orientation2color(solution_vectors)
    return torch.tensor(rgb_y, device='cuda')

def get_colors_IPF_Z(euler_angles):
    solution_vectors = Orientation.from_euler(euler_angles.cpu().numpy(), degrees=True)
    ipfkey = IPFColorKeyTSL(symmetry.Oh)
    rgb_z = ipfkey.orientation2color(solution_vectors)
    return torch.tensor(rgb_z, device='cuda')

#---------------------------------Функция подготовки чересстрочных пропусков------------------------------------
def get_full_interlaced(in_path, y, x):
    df, size_x, size_y, step_x, step_y = load_EBSD_file(in_path)

    df = df[['Euler1', 'Euler2', 'Euler3']]
    data = df.values.astype(np.float32)
    image_data = data.reshape(size_y, size_x, 3)

    # Увеличиваем размеры для создания пропусков
    size_y = size_y * y
    size_x = size_x * x

    # Создаем массив с пропусками по обеим осям
    restored = np.zeros((size_y, size_x, 3), dtype=np.float32)
    restored[::y, ::x, :] = image_data

    print(restored.shape)

    flat_data = restored.reshape(-1, 3)
    df = pd.DataFrame(flat_data, columns=["Euler1", "Euler2", "Euler3"])
    #create_euler_image(df, out_path, size_x, size_y)
    return df, size_x, size_y, step_x, step_y
#------------------------------------------------------------------------------------------------