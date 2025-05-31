import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QWidget, QFileDialog, QLabel, 
                            QSpacerItem, QSizePolicy, QMessageBox)
from PyQt5.QtCore import QFileInfo
from PyQt5 import QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtCore import QTimer

from scipy.interpolate import griddata

from math import sin, cos, radians

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self._setup_constants()
        self._init_ui()
        self._init_data()

        self.folder_import = None
        self.folder_export = None

    def _setup_constants(self):
        self.MIN_WINDOW_SIZE = (1200, 600)
        self.CANVAS_MARGINS = {"left": 0.0, "bottom": 0.0, "right": 1.00, "top": 1.00}
        self.FILE_TYPES = "CTF Files (*.ctf);;Excel Files (*.xlsx)"
        
    def _init_data(self):
        self.STEP = 6
        self.data = None
        self.data_target = None
        self.filename = None
        self.image_data = None
        self.fragment_data = None
        self.x_coords = None
        self.y_coords = None
        self.image_size = (0, 0)
        self.selection_coords = [0, 0, 0, 0]
        self.drawing = False
        self.current_rect = None
        self.start_point = None
        self.last_mouse_pos = [0, 0]
        self.file_type = "EBSD"

    def _init_ui(self):
        self.setWindowTitle('EBSD Euler Editor')
        self.setGeometry(100, 100, *self.MIN_WINDOW_SIZE)

        self.setStyleSheet("background-color: #2E4E5E; color: white;")
        self.button_style = "background-color: #1E80CC; color: white; font-size: 14px;"

        self.font_header = QtGui.QFont()
        self.font_header.setFamily("Arial")
        self.font_header.setPointSize(18)
        self.font_header.setBold(True)

        self.font_content = QtGui.QFont()
        self.font_content.setFamily("Arial")
        self.font_content.setPointSize(10)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        self._create_left_panel(main_layout)
        self._create_right_panel(main_layout)
        self._connect_event_handlers()

    def _create_left_panel(self, parent_layout):
        left_layout = QVBoxLayout()
        parent_layout.addLayout(left_layout)
        
        control_layout = QHBoxLayout()
        self.load_button = QPushButton('Загрузить EBSD')
        self.load_button.setStyleSheet(self.button_style)
        self.main_size_label = QLabel('Main size: ')
        self.main_size_label.setFont(self.font_content)

        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.main_size_label)
        left_layout.addLayout(control_layout)

        # Папка импорта
        layout_import_folder = QHBoxLayout()
        self.button_folder_import = QPushButton('Папка импорта')
        self.button_folder_import.setStyleSheet(self.button_style)
        self.label_folder_import = QLabel('No folder selected')
        self.label_folder_import.setFont(self.font_content)
        layout_import_folder.addWidget(self.button_folder_import)
        layout_import_folder.addWidget(self.label_folder_import)

        # Папка экспорта
        layout_export_folder = QHBoxLayout()
        self.button_folder_export = QPushButton('Папка экспорта')
        self.button_folder_export.setStyleSheet(self.button_style)
        self.label_folder_export = QLabel('No folder selected')
        self.label_folder_export.setFont(self.font_content)
        layout_export_folder.addWidget(self.button_folder_export)
        layout_export_folder.addWidget(self.label_folder_export)
        
        # Панель выбора операции
        layout_operations = QHBoxLayout()
        self.button_export_cur = QPushButton('Экспорт текущего')
        self.button_export_cur.setStyleSheet(self.button_style)
        self.button_export_all = QPushButton('Экспорт всех из папки')
        self.button_export_all.setStyleSheet(self.button_style)
        layout_operations.addWidget(self.button_export_cur)
        layout_operations.addWidget(self.button_export_all)
        
        self.main_figure, self.main_ax = plt.subplots()
        self.main_figure.subplots_adjust(**self.CANVAS_MARGINS)
        self.main_canvas = FigureCanvasQTAgg(self.main_figure)
        self.main_figure.patch.set_facecolor('#2E2E2E')
        self.main_ax.set_facecolor('#2E2E2E')

        left_layout.addLayout(layout_import_folder)
        left_layout.addLayout(layout_export_folder)
        left_layout.addLayout(layout_operations)
        left_layout.addWidget(self.main_canvas)
        left_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def _create_right_panel(self, parent_layout):
        right_layout = QVBoxLayout()
        parent_layout.addLayout(right_layout)
        
        control_layout = QHBoxLayout()
        self.export_button = QPushButton('Сохранить фрагмент в файл')
        self.export_button.setStyleSheet(self.button_style)
        self.size_label = QLabel('Select size: ')
        self.size_label.setFont(self.font_content)
        control_layout.addWidget(self.export_button)
        control_layout.addWidget(self.size_label)
        right_layout.addLayout(control_layout)
        
        self.fragment_figure, self.fragment_ax = plt.subplots()
        self.fragment_figure.subplots_adjust(**self.CANVAS_MARGINS)
        self.fragment_canvas = FigureCanvasQTAgg(self.fragment_figure)
        self.fragment_figure.patch.set_facecolor('#2E2E2E')
        self.fragment_ax.set_facecolor('#2E2E2E')
        right_layout.addWidget(self.fragment_canvas)
        
        right_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def _connect_event_handlers(self):
        self.load_button.clicked.connect(self.open_file)
        self.export_button.clicked.connect(self.export_fragment)
        self.button_folder_import.clicked.connect(self.select_folder_import)
        self.button_folder_export.clicked.connect(self.select_folder_export)
        self.button_export_cur.clicked.connect(lambda: self.export_cur(show_msg=True))
        self.button_export_all.clicked.connect(self.export_all)
        self.main_canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.main_canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.main_canvas.mpl_connect('button_release_event', self.on_mouse_release)

    def select_folder_import(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.folder_import = folder_path
            self.label_folder_import.setText(f"{folder_path}")
    
    def select_folder_export(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.folder_export = folder_path
            self.label_folder_export.setText(f"{folder_path}")

    def export_cur(self, show_msg = True):
        if not self.folder_export:
            QMessageBox.warning(self, 'Warning', 'Please select an export folder first.')
            return
        if self.data is None:
            QMessageBox.warning(self, 'Warning', 'No data loaded.')
            return
        
        euler_angles = self.data[["Euler1", "Euler2", "Euler3"]].values
        data = euler_angles.reshape(*self.image_size[::-1], 3)

        window_size = 256
        width, height = self.image_size
        
        if width < window_size or height < window_size:
            if show_msg:
                QMessageBox.warning(self, 'Warning', 'The image is smaller than 256x256. Exporting the entire image.')
            return
        
        total = 0
        try:
            # Перебираем изображение с шагом 256, корректируя координаты у краев
            for y_start in range(0, height, window_size):
                for x_start in range(0, width, window_size):
                    # Корректируем координаты, если окно выходит за границы
                    x_end = min(x_start + window_size, width)
                    y_end = min(y_start + window_size, height)
                    
                    # Если окно вышло за границы, сдвигаем его назад
                    if x_end - x_start < window_size:
                        x_start = max(0, x_end - window_size)
                    if y_end - y_start < window_size:
                        y_start = max(0, y_end - window_size)
                    
                    # Обновляем конечные координаты после сдвига
                    x_end = min(x_start + window_size, width)
                    y_end = min(y_start + window_size, height)

                    self.selection_coords[2] = x_end
                    self.selection_coords[0] = x_start
                    self.selection_coords[3] = y_end
                    self.selection_coords[1] = y_start
                    
                    fragment_data = data[y_start:y_end, x_start:x_end, :].copy()
                    print(y_start,y_end,x_start,x_end)
                    
                    img_np = np.array(fragment_data)
                    flat_data = img_np.reshape(-1, 3)
                    df = pd.DataFrame(flat_data, columns=["Euler1", "Euler2", "Euler3"])
                    # Сохраняем фрагмент
                    filename = f"{self.filename}_{x_start}-{y_start}.ctf"
                    file_path = os.path.join(self.folder_export, filename)
                    self._save_fragment_file(file_path, df)
                    total += 1
            
            if show_msg:
                QMessageBox.information(self, 'Success', f'Exported {total} fragments.')
        
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Auto export failed: {str(e)}')

    def export_all(self):
        if not self.folder_import:
            QMessageBox.warning(self, 'Warning', 'Please select an import folder first.')
            return
        if not self.folder_export:
            QMessageBox.warning(self, 'Warning', 'Please select an export folder first.')
            return
        # Проходим по всем файлам в папке
        for root, dirs, files in os.walk(self.folder_import):
            for file in files:
                if file.endswith(".ctf"):
                    file_path = os.path.join(root, file)
                    self.load_file(file_path)
                    self.export_cur(show_msg=False)
                    print(file_path)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", self.FILE_TYPES
        )
        if not file_path:
            return

        self.load_file(file_path)


    def load_file(self, file_path):
        self._init_data() # reset data

        fileinfo = QFileInfo(file_path)
        self.filename = fileinfo.baseName()

        try:
            self._process_ctf_file(file_path)    
            self._process_image_data()
            self._display_main_image()
            self.main_size_label.setText(f'{self.filename} | Main size: {str(self.image_size[0])} x {str(self.image_size[1])}')
              
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load file: {str(e)}')

    def _process_ctf_file(self, file_path):
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            
        if first_line == "FRAGMENT":
            self.file_type = "FRAGMENT"
            self._process_fragment_ctf(file_path)
        else:
            self.file_type = "EBSD"
            self._process_ebsd_ctf(file_path)

    def _process_fragment_ctf(self, file_path):
        with open(file_path, 'r') as f:
            f.readline() # Skip type line
            self.image_size = (int(f.readline()), int(f.readline()))
            
        self.data = pd.read_csv(file_path, sep='\t', skiprows=3)

    def _process_ebsd_ctf(self, file_path):
        with open(file_path, 'r') as f:
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

        self.data = pd.read_csv(file_path, engine='c', low_memory=False, sep='\t', skiprows=header_line_number-1)

        try:
            self.data['Euler1'] = pd.to_numeric(self.data['Euler1'].str.replace(',', '.'), errors='coerce')
            self.data['Euler2'] = pd.to_numeric(self.data['Euler2'].str.replace(',', '.'), errors='coerce')
            self.data['Euler3'] = pd.to_numeric(self.data['Euler3'].str.replace(',', '.'), errors='coerce')
        except AttributeError:
            print(f"Замена запятых на точки не требуется для файла: {file_path}")
            pass

        self._normalize_coordinates()

    def _normalize_coordinates(self):
        self.STEP = float(self.data['X'].iloc[-1]) - float(self.data['X'].iloc[-2])
        self.x_coords = self.data['X'] / self.STEP
        self.y_coords = self.data['Y'] / self.STEP

        self.image_size = (
            int(np.ceil(self.x_coords.max())) + 1,
            int(np.ceil(self.y_coords.max())) + 1
        )
        print("image_size:", self.image_size)
        print("step_x:", self.STEP)

    def _process_image_data(self):
        if self.file_type == "FRAGMENT":
            x = np.arange(len(self.data)) % self.image_size[0]
            y = np.arange(len(self.data)) // self.image_size[0]
            self.x_coords = x
            self.y_coords = y
 
        euler_angles = self.data[["Euler1", "Euler2", "Euler3"]].values
        self.image_data = euler_angles.reshape(*self.image_size[::-1], 3) / 360

    def _display_main_image(self):
        self.main_ax.clear()
        self.main_ax.imshow(self.image_data)
        self.main_ax.axis('off')
        self.main_canvas.draw_idle() # draw

    def on_mouse_press(self, event):
        if event.xdata and event.ydata:
            self.drawing = True
            self.start_point = (event.xdata, event.ydata)

    def on_mouse_move(self, event):
        if self.drawing:
            if event.xdata:
                self.last_mouse_pos[0] = event.xdata
            if event.ydata:
                self.last_mouse_pos[1] = event.ydata
            self._update_rectangle()
        
    def on_mouse_release(self, event):
        self.drawing = False
        self._update_selection_coords()
        self._display_fragment()

    def _update_rectangle(self):
        if self.current_rect:
            self.current_rect.remove()

        x1, y1 = self.start_point
        x2, y2 = self.last_mouse_pos

        width = x2 - x1
        height = y2 - y1
        
        self.current_rect = plt.Rectangle(
            (x1, y1), width, height, 
            edgecolor='white', facecolor='none', linewidth=1
        )
        self.main_ax.add_patch(self.current_rect)
        self.main_canvas.draw()

    def _update_selection_coords(self):
        if self.start_point and self.last_mouse_pos:
            x1, y1 = self.start_point
            x2, y2 = self.last_mouse_pos
            
            if abs(x2 - x1) > 1 and abs(y2 - y1) > 1:
                self.selection_coords = [
                    int(min(x1, x2)), 
                    int(min(y1, y2)), 
                    int(max(x1, x2)), 
                    int(max(y1, y2))
                ]

    def _display_fragment(self):
        if self.data is not None and self.selection_coords[0] > 1:
            x1, y1, x2, y2 = self.selection_coords
            fragment = self.image_data[y1:y2, x1:x2]
            
            self.fragment_ax.clear()
            self.fragment_ax.imshow(fragment)
            self.fragment_ax.axis('off')
            self.fragment_canvas.draw()
            
            size_info = f"Select size: {x2-x1} x {y2-y1} | Coords: {x1}, {y1}, {x2}, {y2}"
            self.size_label.setText(size_info)

    def export_fragment(self):
        if not self.selection_coords:
            QMessageBox.warning(self, 'Warning', 'No fragment selected')
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Fragment", "", self.FILE_TYPES
        )
        if not file_path:
            return

        try:
            fragment = self._extract_fragment_data()
            self._save_fragment_file(file_path, fragment)
            QMessageBox.information(self, 'Success', 'Fragment saved successfully')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Export failed: {str(e)}', '')

    def _extract_fragment_data(self):
        x1, y1, x2, y2 = self.selection_coords

        mask = (
            (self.x_coords >= x1) & (self.x_coords < x2) &
            (self.y_coords >= y1) & (self.y_coords < y2)
        )
        fragment = self.data[mask].copy()
        
        return fragment[['Euler1', 'Euler2', 'Euler3']]

    def _save_fragment_file(self, path, data):
        with open(path, 'w', newline='') as f:
            f.write(f"FRAGMENT\n{self.selection_coords[2]-self.selection_coords[0]}\n")
            f.write(f"{self.selection_coords[3]-self.selection_coords[1]}\n")
            data.to_csv(f, sep='\t', index=False, lineterminator='\n')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())