import sys
import os
import traceback
import re
from pathlib import Path

import numpy as np
from PIL.Image import open as PILopen
from PIL import Image, ImageQt

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QStatusBar, QProgressDialog,
    QSplitter, QMessageBox, QToolButton, QMenu, QSlider, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtGui import QPixmap, QImage, QImageReader, QColor
from PySide6.QtCore import Qt, QCoreApplication

from inference.classifier import return_classifier
from inference.segmenter import return_segmenter
from config import Config


class ImageViewer(QMainWindow):
    def __init__(self, classes, classification, segmentation):
        super().__init__()
        self.setWindowTitle("Просмотр изображений (Multi‑label)")
        self.setGeometry(100, 100, 1200, 700)

        self.image_paths = []
        self.labels = []          # каждый элемент – список активных классов (строки)
        self.probs_list = []      # каждый элемент – массив вероятностей (np.array)
        self.current_index = -1
        self.save_dir = None
        self.classes = classes
        self.threshold = 0.5

        # Задаём цвета для классов (можно взять из конфига или сгенерировать)
        self.class_colors = [
            '#FF4136', '#2ECC40', '#0074D9', '#FF851B',
            '#FFD700', '#8A2BE2', '#FF69B4', '#00CED1'
        ]
        if len(self.class_colors) < len(self.classes):
            import colorsys
            for i in range(len(self.class_colors), len(self.classes)):
                hue = i / len(self.classes)
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
                self.class_colors.append('#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
                ))

        self.segmented_paths = {}

        self.classification = classification
        self.segmentation = segmentation

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # --- Основной вертикальный сплиттер: изображения + таблица ---
        self.main_splitter = QSplitter(Qt.Vertical)

        self.image_splitter = QSplitter(Qt.Horizontal)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.setText("Оригинальное изображение")
        self.image_label.setStyleSheet("background-color: #f0f0f0;")

        self.image_segm = QLabel()
        self.image_segm.setAlignment(Qt.AlignCenter)
        self.image_segm.setMinimumSize(200, 200)
        self.image_segm.setText("Сегментированное изображение")
        self.image_segm.setStyleSheet("background-color: #f0f0f0;")

        self.image_splitter.addWidget(self.image_label)
        self.image_splitter.addWidget(self.image_segm)
        self.image_splitter.setSizes([500, 500])

        self.main_splitter.addWidget(self.image_splitter)

        # Таблица предсказаний
        self.prediction_table = QTableWidget()
        self.prediction_table.setColumnCount(4)
        self.prediction_table.setHorizontalHeaderLabels(["Класс", "Цвет", "Вероятность", "Обнаружен"])
        self.prediction_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.prediction_table.verticalHeader().setVisible(False)
        self.prediction_table.setMinimumHeight(150)
        self.main_splitter.addWidget(self.prediction_table)

        self.main_splitter.setSizes([500, 200])

        # --- Остальные элементы управления ---
        self.caption_label = QLabel()
        self.caption_label.setAlignment(Qt.AlignCenter)
        self.caption_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.open_button = QToolButton()
        self.open_button.setText("Открыть")
        self.open_button.setPopupMode(QToolButton.InstantPopup)

        self.open_menu = QMenu()
        self.open_menu.addAction("Папку с изображениями", self.open_folder)
        self.open_menu.addAction("Одно изображение", self.open_single_image)
        self.open_button.setMenu(self.open_menu)

        self.select_save_button = QPushButton("Выбрать папку для сохранения")
        self.save_button = QPushButton("Сохранить сегментированные изображения")
        self.prev_button = QPushButton("Назад")
        self.next_button = QPushButton("Вперед")

        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.save_button.setEnabled(False)

        self.select_save_button.clicked.connect(self.select_save_directory)
        self.save_button.clicked.connect(self.save_images)
        self.prev_button.clicked.connect(self.show_prev_image)
        self.next_button.clicked.connect(self.show_next_image)

        # Элементы управления порогом
        self.threshold_label = QLabel(f"Порог: {self.threshold:.2f}")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(int(self.threshold * 100))
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)

        self.threshold_edit = QLineEdit()
        self.threshold_edit.setFixedWidth(50)
        self.threshold_edit.setText(f"{self.threshold:.2f}")
        self.threshold_edit.returnPressed.connect(self.on_threshold_entered)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Порог:"))
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_edit)
        threshold_layout.addStretch()

        button_layout1 = QHBoxLayout()
        button_layout1.addWidget(self.open_button)
        button_layout1.addWidget(self.select_save_button)
        button_layout1.addWidget(self.save_button)

        button_layout2 = QHBoxLayout()
        button_layout2.addWidget(self.prev_button)
        button_layout2.addWidget(self.next_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.main_splitter, 1)
        main_layout.addWidget(self.caption_label)
        main_layout.addLayout(threshold_layout)
        main_layout.addLayout(button_layout1)
        main_layout.addLayout(button_layout2)

        self.central_widget.setLayout(main_layout)

    # ----------------------------------------------------------------------
    # Вспомогательный метод для получения PIL.Image из сегментации
    # ----------------------------------------------------------------------
    def _get_segmented_image(self, img):
        """Вызывает сегментатор и преобразует результат в PIL.Image."""
        result = self.segmentation(img)
        if isinstance(result, np.ndarray):
            # Предполагаем, что массив в диапазоне [0, 1] (float) или [0, 255] (uint8)
            if result.dtype in (np.float32, np.float64):
                arr = (result * 255).astype(np.uint8)
            elif result.dtype == np.uint8:
                arr = result
            else:
                arr = result.astype(np.uint8)
            # Если массив одноканальный, конвертируем в RGB (для совместимости)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.concatenate([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] != 3:
                raise ValueError(f"Unexpected channel dimension: {arr.shape[2]}")
            return Image.fromarray(arr)
        elif isinstance(result, Image.Image):
            return result
        else:
            raise TypeError("Segmentation result must be numpy.ndarray or PIL.Image")

    # ----------------------------------------------------------------------
    # Обработка классификации
    # ----------------------------------------------------------------------
    def get_probs_and_active(self, img):
        result = self.classification(img)
        if isinstance(result, tuple) and len(result) == 2:
            probs, labels = result
        else:
            probs = result
            labels = None

        if labels is None:
            labels = (np.array(probs) >= self.threshold).astype(int)

        active = [self.classes[i] for i, val in enumerate(labels) if val == 1]
        return np.array(probs), active

    # ----------------------------------------------------------------------
    # Обработка и сохранение одного изображения
    # ----------------------------------------------------------------------
    def process_and_save_image(self, img_path, active_classes, save_dir):
        try:
            invalid_chars = re.compile(r'[<>:"/\\|?*]')
            base_name = os.path.basename(img_path)
            file_name, file_ext = os.path.splitext(base_name)

            if active_classes:
                clean_classes = [invalid_chars.sub('_', c) for c in active_classes]
                suffix = "_" + "_".join(clean_classes)
            else:
                suffix = "_none"

            new_file_name = f"{file_name}{suffix}{file_ext}"
            save_path = os.path.join(save_dir, new_file_name)

            if not os.path.exists(save_path):
                image = Image.open(img_path)
                segm_image = self._get_segmented_image(image)
                qimage = ImageQt.ImageQt(segm_image)
                qimage.save(save_path)

            return save_path
        except Exception as e:
            print(f"Ошибка обработки изображения {img_path}: {e}")
            return None

    # ----------------------------------------------------------------------
    # Открытие папки и одного изображения (без изменений, но с использованием _get_segmented_image)
    # ----------------------------------------------------------------------
    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с изображениями",
            str(Path.home()),
            QFileDialog.ShowDirsOnly
        )

        if not folder_path:
            return

        supported = [f.data().decode().lower() for f in QImageReader.supportedImageFormats()]

        image_paths = []
        for root, dirs, files in os.walk(folder_path):
            for fname in files:
                ext = os.path.splitext(fname)[-1].lower()[1:]
                if ext in supported:
                    image_paths.append(os.path.join(root, fname))

        total_images = len(image_paths)
        if total_images == 0:
            self.status_bar.showMessage("В папке не найдено изображений")
            return

        progress = QProgressDialog("Обработка изображений...", "Отмена", 0, total_images, self)
        progress.setWindowTitle("Обработка изображений")
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)

        labels = []
        probs_list = []
        processed = 0
        for fpath in image_paths:
            try:
                img = PILopen(fpath)
                probs, active = self.get_probs_and_active(img)
                labels.append(active)
                probs_list.append(probs)

                if self.save_dir:
                    segm_path = self.process_and_save_image(fpath, active, self.save_dir)
                    if segm_path:
                        self.segmented_paths[fpath] = segm_path
            except Exception as e:
                print(f"Ошибка обработки файла {fpath}: {e}")
                labels.append([])
                probs_list.append(np.zeros(len(self.classes)))

            processed += 1
            progress.setValue(processed)
            QCoreApplication.processEvents()
            if progress.wasCanceled():
                image_paths = image_paths[:processed]
                labels = labels[:processed]
                probs_list = probs_list[:processed]
                break

        progress.close()

        if not image_paths:
            self.status_bar.showMessage("Не загружено изображений")
            return

        self.set_image_list(image_paths, labels, probs_list)

    def open_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_path:
            return

        img = PILopen(file_path)
        probs, active = self.get_probs_and_active(img)

        if self.save_dir:
            segm_path = self.process_and_save_image(file_path, active, self.save_dir)
            if segm_path:
                self.segmented_paths[file_path] = segm_path

        self.set_image_list([file_path], [active], [probs])

    def set_image_list(self, image_paths, labels, probs_list):
        self.image_paths = image_paths
        self.labels = labels
        self.probs_list = probs_list
        self.current_index = 0

        self.status_bar.showMessage(f"Загружено изображений: {len(self.image_paths)}")

        self.prev_button.setEnabled(len(self.image_paths) > 1)
        self.next_button.setEnabled(len(self.image_paths) > 1)

        self.show_image()

    # ----------------------------------------------------------------------
    # Выбор папки для сохранения и обработка существующих
    # ----------------------------------------------------------------------
    def select_save_directory(self):
        save_dir = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку для сохранения",
            str(Path.home()),
            QFileDialog.ShowDirsOnly
        )

        if save_dir:
            self.save_dir = save_dir
            self.status_bar.showMessage(f"Папка для сохранения: {save_dir}")
            self.save_button.setEnabled(True)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if self.image_paths:
                self.process_existing_images()

    def process_existing_images(self):
        if not self.save_dir or not self.image_paths:
            return

        total = len(self.image_paths)
        progress = QProgressDialog("Обработка существующих изображений...", "Отмена", 0, total, self)
        progress.setWindowTitle("Обработка изображений")
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)

        for i, img_path in enumerate(self.image_paths):
            if progress.wasCanceled():
                break
            if img_path not in self.segmented_paths:
                active = self.labels[i]
                segm_path = self.process_and_save_image(img_path, active, self.save_dir)
                if segm_path:
                    self.segmented_paths[img_path] = segm_path
            progress.setValue(i + 1)
            QCoreApplication.processEvents()

        progress.close()
        self.status_bar.showMessage(f"Обработано изображений: {len(self.segmented_paths)}")

    # ----------------------------------------------------------------------
    # Сохранение всех изображений
    # ----------------------------------------------------------------------
    def save_images(self):
        if not self.save_dir:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите папку для сохранения!")
            return
        if not self.image_paths:
            QMessageBox.warning(self, "Ошибка", "Нет изображений для сохранения!")
            return

        total = len(self.image_paths)
        progress = QProgressDialog("Сохранение изображений...", "Отмена", 0, total, self)
        progress.setWindowTitle("Сохранение изображений")
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)

        list_file_path = os.path.join(self.save_dir, "image_list.txt")
        file_mode = "a" if os.path.exists(list_file_path) else "w"
        invalid_chars = re.compile(r'[<>:"/\\|?*]')

        saved_count = 0
        skipped_count = 0

        with open(list_file_path, file_mode, encoding="utf-8") as list_file:
            if file_mode == "a":
                list_file.write("\n" + "-" * 40 + "\n")

            for i, img_path in enumerate(self.image_paths):
                if progress.wasCanceled():
                    break

                progress.setValue(i)
                progress.setLabelText(f"Обработка {i+1}/{total}")
                QCoreApplication.processEvents()

                active = self.labels[i]
                base_name = os.path.basename(img_path)
                file_name, file_ext = os.path.splitext(base_name)

                if active:
                    clean_classes = [invalid_chars.sub('_', c) for c in active]
                    suffix = "_" + "_".join(clean_classes)
                else:
                    suffix = "_none"

                new_file_name = f"{file_name}{suffix}{file_ext}"
                save_path = os.path.join(self.save_dir, new_file_name)

                if os.path.exists(save_path):
                    skipped_count += 1
                    self.segmented_paths[img_path] = save_path
                    continue

                list_file.write(new_file_name + "\n")

                try:
                    image = Image.open(img_path)
                    segm_image = self._get_segmented_image(image)
                    qimage = ImageQt.ImageQt(segm_image)
                    if qimage.save(save_path):
                        saved_count += 1
                        self.segmented_paths[img_path] = save_path
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(e)
                    skipped_count += 1

                QCoreApplication.processEvents()

            if not progress.wasCanceled():
                progress.setValue(total)

        progress.close()

        message = f"Сохранено: {saved_count}"
        if skipped_count > 0:
            message += f"\nПропущено: {skipped_count} (уже существуют или ошибка)"
        QMessageBox.information(
            self,
            "Сохранение завершено",
            f"{message}\nФайл списка: image_list.txt\nРежим: {'Дозапись' if file_mode == 'a' else 'Новый файл'}"
        )

    # ----------------------------------------------------------------------
    # Отображение текущего изображения
    # ----------------------------------------------------------------------
    def show_image(self):
        if not (0 <= self.current_index < len(self.image_paths)):
            self.image_label.setText("Изображение не доступно")
            self.image_segm.setText("Изображение не доступно")
            self.caption_label.setText("")
            self.prediction_table.clearContents()
            return

        img_path = self.image_paths[self.current_index]
        active_classes = self.labels[self.current_index] if self.current_index < len(self.labels) else []
        probs = self.probs_list[self.current_index] if self.current_index < len(self.probs_list) else None

        # Оригинал
        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            orig_scaled = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(orig_scaled)
        else:
            self.image_label.setText("Ошибка загрузки")

        # Сегментация
        if img_path in self.segmented_paths and os.path.exists(self.segmented_paths[img_path]):
            segm_pixmap = QPixmap(self.segmented_paths[img_path])
        else:
            try:
                img = PILopen(img_path)
                segm_image = self._get_segmented_image(img)
                segm_pixmap = QPixmap.fromImage(ImageQt.ImageQt(segm_image))
                if self.save_dir:
                    segm_path = self.process_and_save_image(img_path, active_classes, self.save_dir)
                    if segm_path:
                        self.segmented_paths[img_path] = segm_path
            except Exception as e:
                print(f"Ошибка сегментации: {e}")
                segm_pixmap = QPixmap()

        if not segm_pixmap.isNull():
            segm_scaled = segm_pixmap.scaled(
                self.image_segm.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_segm.setPixmap(segm_scaled)
        else:
            self.image_segm.setText("Ошибка сегментации")

        # Подпись
        if active_classes:
            caption = "Классы: " + ", ".join(active_classes)
        else:
            caption = "Нет дефектов"
        self.caption_label.setText(caption)

        # Таблица
        self.update_prediction_table(probs)

        self.setWindowTitle(f"Просмотр изображений - [{self.current_index+1}/{len(self.image_paths)}]")

    # ----------------------------------------------------------------------
    # Обновление таблицы
    # ----------------------------------------------------------------------
    def update_prediction_table(self, probs):
        if probs is None or len(probs) != len(self.classes):
            self.prediction_table.setRowCount(0)
            return

        self.prediction_table.setRowCount(len(self.classes))
        for i, cls_name in enumerate(self.classes):
            prob = probs[i]
            detected = prob >= self.threshold

            item_class = QTableWidgetItem(cls_name)
            item_class.setTextAlignment(Qt.AlignCenter)
            self.prediction_table.setItem(i, 0, item_class)

            color_hex = self.class_colors[i % len(self.class_colors)]
            color_widget = QLabel()
            color_widget.setStyleSheet(f"background-color: {color_hex}; border-radius: 4px;")
            color_widget.setAlignment(Qt.AlignCenter)
            color_widget.setFixedSize(24, 24)
            self.prediction_table.setCellWidget(i, 1, color_widget)

            prob_str = f"{prob:.2f}"
            item_prob = QTableWidgetItem(prob_str)
            item_prob.setTextAlignment(Qt.AlignCenter)
            self.prediction_table.setItem(i, 2, item_prob)

            status = "✅ Да" if detected else "❌ Нет"
            item_status = QTableWidgetItem(status)
            item_status.setTextAlignment(Qt.AlignCenter)
            if detected:
                item_status.setForeground(QColor(0, 128, 0))
            else:
                item_status.setForeground(QColor(128, 128, 128))
            self.prediction_table.setItem(i, 3, item_status)

        self.prediction_table.resizeRowsToContents()

    # ----------------------------------------------------------------------
    # Навигация
    # ----------------------------------------------------------------------
    def show_prev_image(self):
        if self.image_paths:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self.show_image()

    def show_next_image(self):
        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.show_image()

    # ----------------------------------------------------------------------
    # Изменение порога
    # ----------------------------------------------------------------------
    def on_threshold_changed(self, value):
        self.threshold = value / 100.0
        self.threshold_label.setText(f"Порог: {self.threshold:.2f}")
        self.threshold_edit.setText(f"{self.threshold:.2f}")
        if self.image_paths:
            self.reclassify_all()

    def on_threshold_entered(self):
        try:
            val = float(self.threshold_edit.text())
            if 0.0 <= val <= 1.0:
                self.threshold = val
                self.threshold_slider.setValue(int(val * 100))
                self.threshold_label.setText(f"Порог: {val:.2f}")
                if self.image_paths:
                    self.reclassify_all()
            else:
                self.threshold_edit.setText(f"{self.threshold:.2f}")
        except ValueError:
            self.threshold_edit.setText(f"{self.threshold:.2f}")

    def reclassify_all(self):
        if not self.image_paths:
            return

        new_labels = []
        for probs in self.probs_list:
            labels = (probs >= self.threshold).astype(int)
            active = [self.classes[i] for i, val in enumerate(labels) if val == 1]
            new_labels.append(active)

        self.labels = new_labels
        self.show_image()
        self.status_bar.showMessage("Метки обновлены с новым порогом")

    # ----------------------------------------------------------------------
    # Обработка изменения размера окна
    # ----------------------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_paths and 0 <= self.current_index < len(self.image_paths):
            self.show_image()


# ----------------------------------------------------------------------
# Функция запуска
# ----------------------------------------------------------------------
def app_run(classes, classification, segmentation):
    app = QApplication(sys.argv)

    try:
        window = ImageViewer(classes, classification, segmentation)
        window.show()
        return app.exec()
    except Exception as e:
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Critical)
        error_msg.setWindowTitle("Ошибка запуска")
        error_msg.setText("Не удалось запустить приложение")
        error_details = f"Произошла ошибка при инициализации:\n\n{str(e)}"
        error_details += f"\n\nТип ошибки: {type(e).__name__}"
        error_details += f"\n\nTraceback:\n{traceback.format_exc()}"
        error_msg.setDetailedText(error_details)
        error_msg.exec()
        return 1


if __name__ == "__main__":
    config = Config.from_json('config.json')

    classifier_config = {
        'device': config.device,
        'num_classes': config.num_classes,
        'in_height': config.in_height,
        'in_width': config.in_width,
        'model_name': config.model_name,
        'model_mode': config.model_mode,
        'h_load_path': config.h_load_path,
        'transform_mean': config.transform_mean,
        'transform_std': config.transform_std,
        'threshold': config.threshold,
        'class_names': config.class_names,
    }
    classifier_func, class_names = return_classifier(classifier_config)
    segmenter_func = return_segmenter()   # возвращает numpy-массив

    exit_code = app_run(class_names, classifier_func, segmenter_func)
    sys.exit(exit_code)