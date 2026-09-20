import sys
import os
import csv
import numpy as np
import mat73
from datetime import datetime, timedelta

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

def savgol_filter_numpy(x, window_length, polyorder=2):
    """Быстрый фильтр Савицкого — Голея на чистом NumPy"""
    window_length = int(window_length)
    if window_length % 2 == 0:
        window_length += 1
    
    half_window = window_length // 2
    idx = np.arange(-half_window, half_window + 1)
    X = np.vander(idx, polyorder + 1, increasing=True)
    
    try:
        coefs = np.linalg.pinv(X)[0]
    except np.linalg.LinAlgError:
        return x.copy()
        
    return np.convolve(x, coefs[::-1], mode='same')


class DateAxis(pg.AxisItem):
    """Кастомный класс оси для адаптивного отображения времени и даты снизу"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_stamps = None
        self.enableAutoSIPrefix(False)
        self.setTextPen(QtGui.QColor(0, 0, 0))
        self.setPen(QtGui.QColor(0, 0, 0))

    def set_time_stamps(self, time_stamps):
        self.time_stamps = time_stamps
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        strings = []
        if self.time_stamps is None or len(self.time_stamps) == 0:
            return [f"{int(v)}" for v in values]

        if len(values) > 1:
            span_indices = values[-1] - values[0]
            dt_span = span_indices * (self.time_stamps[-1] - self.time_stamps[0]) / max(1, len(self.time_stamps))
            avg_interval_sec = abs(dt_span / len(values))
        else:
            avg_interval_sec = 86400

        for v in values:
            try:
                idx = int(round(v))
                idx = max(0, min(idx, len(self.time_stamps) - 1))
                ts = self.time_stamps[idx]
                dt = datetime.fromtimestamp(ts)
                
                if avg_interval_sec < 3600 * 6:
                    strings.append(dt.strftime('%d.%m.%Y\n%H:%M'))
                elif avg_interval_sec < 86400 * 5:
                    strings.append(dt.strftime('%d.%m.%Y\n%H:00'))
                else:
                    strings.append(dt.strftime('%d.%m.%Y'))
            except (ValueError, OverflowError):
                strings.append('')
        return strings


class NumericAxis(pg.AxisItem):
    """Ось для отображения порядковых номеров (индексов) сверху"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enableAutoSIPrefix(False)
        self.setTextPen(QtGui.QColor(0, 0, 0))
        self.setPen(QtGui.QColor(0, 0, 0))

    def tickStrings(self, values, scale, spacing):
        return [f"{int(v):,}" if v.is_integer() else f"{v:.1f}" for v in values]


class DonskoiApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Анализ нагонов и сгонов — Донской 1001 (POSIX)")
        self.resize(1300, 900)
        
        self.file_path = "U_Donskoi_1001_Baza_очищенный_pos.mat"
        self.u_level = None
        self.time_stamps = None
        self.trend1 = None
        self.trend2 = None
        self.nagon_mask = None
        self.sgon_mask = None
        self.current_start = 0
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        left_layout = QtWidgets.QVBoxLayout()
        
        # Панель выбора файла
        file_layout = QtWidgets.QHBoxLayout()
        file_layout.addWidget(QtWidgets.QLabel("Файл данных (.mat):"))
        self.input_file_path = QtWidgets.QLineEdit(self.file_path)
        file_layout.addWidget(self.input_file_path)
        self.btn_browse = QtWidgets.QPushButton("Обзор...")
        file_layout.addWidget(self.btn_browse)
        self.btn_load_file = QtWidgets.QPushButton("Загрузить файл")
        self.btn_load_file.setStyleSheet("background-color: #d0e8ff; font-weight: bold;")
        file_layout.addWidget(self.btn_load_file)
        left_layout.addLayout(file_layout)
        
        self.status_label = QtWidgets.QLabel("Ожидание выбора и загрузки файла данных...")
        left_layout.addWidget(self.status_label)
        
        # Создаем оси: СНИЗУ — время (DateAxis), СВЕРХУ — порядковые номера (NumericAxis)
        self.bottom_axis = DateAxis(orientation='bottom')
        self.top_axis = NumericAxis(orientation='top')
        
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': self.bottom_axis, 'top': self.top_axis})
        self.plot_widget.setLabel('left', 'Уровень воды', units='м')
        self.plot_widget.setLabel('bottom', 'Дата и время')
        self.plot_widget.setLabel('top', 'Порядковый номер точки (индекс)')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setBackground('w')
        
        # Настройка левой оси
        self.plot_widget.getAxis('left').setTextPen(QtGui.QColor(0, 0, 0))
        self.plot_widget.getAxis('left').setPen(QtGui.QColor(0, 0, 0))
        
        p_item = self.plot_widget.getPlotItem()
        p_item.showAxis('top')
        p_item.layout.setContentsMargins(10, 40, 10, 10)
        
        # Привязываем обе оси к ViewBox
        self.bottom_axis.linkToView(p_item.vb)
        self.top_axis.linkToView(p_item.vb)
        
        # Настраиваем легенду: задаем черный цвет подписей (labelTextColor='k') и полупрозрачный белый фон
        self.legend = self.plot_widget.addLegend(offset=(30, 30), labelTextColor='k')
        self.legend.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 220)))
        self.legend.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200)))
        
        left_layout.addWidget(self.plot_widget)
        main_layout.addLayout(left_layout, stretch=4)
        
        # Правая панель управления
        right_panel = QtWidgets.QVBoxLayout()
        group_box = QtWidgets.QGroupBox("Параметры отображения и расчетов")
        group_layout = QtWidgets.QVBoxLayout(group_box)
        
        group_layout.addWidget(QtWidgets.QLabel("Окно сглаживания Тренд 1:"))
        self.input_w1 = QtWidgets.QLineEdit("200")
        group_layout.addWidget(self.input_w1)
        
        group_layout.addWidget(QtWidgets.QLabel("Окно сглаживания Тренд 2:"))
        self.input_w2 = QtWidgets.QLineEdit("1700")
        group_layout.addWidget(self.input_w2)
        
        group_layout.addWidget(QtWidgets.QLabel("Длина отображаемого участка (точек):"))
        self.input_length = QtWidgets.QLineEdit("2000")
        group_layout.addWidget(self.input_length)
        
        group_layout.addWidget(QtWidgets.QLabel("Стартовый индекс:"))
        self.input_start = QtWidgets.QLineEdit("0")
        group_layout.addWidget(self.input_start)
        
        group_layout.addSpacing(10)
        self.btn_apply = QtWidgets.QPushButton("Пересчитать и обновить")
        group_layout.addWidget(self.btn_apply)
        
        group_layout.addSpacing(15)
        nav_layout = QtWidgets.QHBoxLayout()
        self.btn_back = QtWidgets.QPushButton("НАЗАД")
        self.btn_forward = QtWidgets.QPushButton("ВПЕРЕД")
        nav_layout.addWidget(self.btn_back)
        nav_layout.addWidget(self.btn_forward)
        group_layout.addLayout(nav_layout)
        
        group_layout.addSpacing(15)
        self.btn_save_png = QtWidgets.QPushButton("СОХРАНИТЬ PNG")
        self.btn_save_png.setStyleSheet("background-color: #e0e0e0; font-weight: bold;")
        group_layout.addWidget(self.btn_save_png)
        
        group_layout.addSpacing(15)
        self.btn_save_nagons = QtWidgets.QPushButton("СОХРАНИТЬ 25 НАГОНОВ")
        self.btn_save_nagons.setStyleSheet("background-color: #d4edda; font-weight: bold; color: #155724;")
        group_layout.addWidget(self.btn_save_nagons)
        
        self.btn_save_sgons = QtWidgets.QPushButton("СОХРАНИТЬ 25 СГОНОВ")
        self.btn_save_sgons.setStyleSheet("background-color: #fff3cd; font-weight: bold; color: #856404;")
        group_layout.addWidget(self.btn_save_sgons)
        
        group_layout.addStretch()
        right_panel.addWidget(group_box)
        main_layout.addLayout(right_panel, stretch=1)
        
        # Кривые на графике (линия нулевого уровня черная, толщина 3)
        self.curve_zero = self.plot_widget.plot(pen=pg.mkPen(color=(0, 0, 0), width=3, style=QtCore.Qt.PenStyle.DashLine), name="Нулевой уровень")
        self.curve_t1 = self.plot_widget.plot(pen=pg.mkPen(color=(255, 0, 0), width=3.5), name="Тренд 1")
        self.curve_t2 = self.plot_widget.plot(pen=pg.mkPen(color=(0, 0, 255), width=3.5), name="Тренд 2")
        self.curve_sgon = self.plot_widget.plot(pen=pg.mkPen(color=(255, 127, 0), width=3.5), name="Сгон")
        self.curve_nagon = self.plot_widget.plot(pen=pg.mkPen(color=(0, 180, 0), width=3.5), name="Нагон")

        # Сигналы
        self.btn_browse.clicked.connect(self.browse_file)
        self.btn_load_file.clicked.connect(self.load_data)
        self.btn_apply.clicked.connect(self.recalculate_and_plot)
        self.btn_back.clicked.connect(self.on_back_clicked)
        self.btn_forward.clicked.connect(self.on_forward_clicked)
        self.btn_save_png.clicked.connect(self.on_save_png_clicked)
        self.btn_save_nagons.clicked.connect(lambda: self.save_top_events(is_nagon=True))
        self.btn_save_sgons.clicked.connect(lambda: self.save_top_events(is_nagon=False))
        
        p_item.vb.sigRangeChanged.connect(self.update_axes)

    def browse_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл MAT", "", "MAT Files (*.mat);;All Files (*.*)")
        if filename:
            self.file_path = filename
            self.input_file_path.setText(filename)

    def load_data(self):
        self.file_path = self.input_file_path.text().strip()
        if not os.path.exists(self.file_path):
            self.status_label.setText(f"Ошибка: файл не найден -> {self.file_path}")
            return

        try:
            mat_data = mat73.loadmat(self.file_path)
            
            # Читаем уровень воды (переводим из сантиметров в метры)
            self.u_level = np.array(mat_data['U_Donskoi_1001_Baza']).flatten() / 100.0
            
            # Прямое считывание готовой POSIX-шкалы времени
            if 'T_pos' in mat_data:
                self.time_stamps = np.array(mat_data['T_pos']).flatten()
            else:
                raise ValueError("В файле отсутствует переменная 'T_pos'!")

            min_len = min(len(self.u_level), len(self.time_stamps))
            self.u_level = self.u_level[:min_len]
            self.time_stamps = self.time_stamps[:min_len]

            self.bottom_axis.set_time_stamps(self.time_stamps)
            self.status_label.setText(f"Данные успешно загружены. Всего точек: {len(self.u_level):,}")
            self.recalculate_and_plot()
            
        except Exception as e:
            self.status_label.setText(f"Ошибка чтения файла: {str(e)}")

    def recalculate_and_plot(self):
        if self.u_level is None:
            return
        try:
            w1 = int(self.input_w1.text())
            w2 = int(self.input_w2.text())
            self.current_start = int(self.input_start.text())
        except ValueError:
            return
            
        self.trend1 = savgol_filter_numpy(self.u_level, window_length=w1, polyorder=2)
        self.trend2 = savgol_filter_numpy(self.u_level, window_length=w2, polyorder=2)
        self.nagon_mask = self.trend1 > self.trend2
        self.sgon_mask = self.trend1 < self.trend2

        self.update_plot()

    def update_plot(self):
        if self.u_level is None:
            return
            
        try:
            display_len = int(self.input_length.text())
        except ValueError:
            display_len = 2000

        total_points = len(self.u_level)
        self.current_start = max(0, min(self.current_start, total_points - 1))
        end_idx = min(self.current_start + display_len, total_points)
        self.input_start.setText(str(self.current_start))
        
        x = np.arange(self.current_start, end_idx, dtype=float)
        y_orig = self.u_level[self.current_start:end_idx]
        y_t1 = self.trend1[self.current_start:end_idx]
        y_t2 = self.trend2[self.current_start:end_idx]
        
        y_nagon = np.where(self.nagon_mask[self.current_start:end_idx], y_orig, np.nan)
        y_sgon = np.where(self.sgon_mask[self.current_start:end_idx], y_orig, np.nan)
        y_zero = np.zeros_like(x)

        # Обновляем кривые
        self.curve_zero.setData(x, y_zero)
        self.curve_t1.setData(x, y_t1)
        self.curve_t2.setData(x, y_t2)
        self.curve_sgon.setData(x, y_sgon)
        self.curve_nagon.setData(x, y_nagon)

        self.plot_widget.setXRange(self.current_start, end_idx - 1, padding=0.01)
        self.update_axes()

    def update_axes(self):
        vb = self.plot_widget.getViewBox()
        self.bottom_axis.linkedViewChanged(vb)
        self.top_axis.linkedViewChanged(vb)

    def on_forward_clicked(self):
        if self.u_level is None:
            return
        try:
            display_len = int(self.input_length.text())
        except ValueError:
            display_len = 2000
        step = max(500, display_len // 2)
        self.current_start = min(self.current_start + step, len(self.u_level) - 1)
        self.update_plot()

    def on_back_clicked(self):
        if self.u_level is None:
            return
        try:
            display_len = int(self.input_length.text())
        except ValueError:
            display_len = 2000
        step = max(500, display_len // 2)
        self.current_start = max(0, self.current_start - step)
        self.update_plot()

    def on_save_png_clicked(self):
        if self.u_level is None:
            QtWidgets.QMessageBox.warning(self, "Внимание", "Нет данных для сохранения!")
            return
        try:
            display_len = int(self.input_length.text())
            w1 = self.input_w1.text()
            w2 = self.input_w2.text()
        except ValueError:
            display_len = 2000
            w1 = "200"
            w2 = "1700"

        end_idx = min(self.current_start + display_len, len(self.u_level))
        default_filename = f"ДОНСКОЙ_{self.current_start}_{end_idx}_w1_{w1}_w2_{w2}.png"
        
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить график как PNG", default_filename, "PNG Images (*.png)")
        if file_name:
            try:
                pixmap = self.plot_widget.grab()
                pixmap.save(file_name, "PNG")
                self.status_label.setText(f"График успешно сохранен в: {file_name}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить файл: {str(e)}")

    def save_top_events(self, is_nagon=True):
        if self.u_level is None or self.trend1 is None or self.trend2 is None:
            QtWidgets.QMessageBox.warning(self, "Внимание", "Сначала загрузите данные и выполните расчет!")
            return

        w1 = self.input_w1.text().strip()
        w2 = self.input_w2.text().strip()
        event_type_str = "нагоны" if is_nagon else "сгоны"
        
        default_filename = f"ДОНСКОЙ_{event_type_str}_w1_{w1}_w2_{w2}_наибольшие_25_.csv"
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, f"Сохранить 25 {event_type_str}", default_filename, "CSV Files (*.csv)")
        
        if not file_name:
            return

        try:
            mask = self.nagon_mask if is_nagon else self.sgon_mask
            dt_sec = np.mean(np.diff(self.time_stamps)) if len(self.time_stamps) > 1 else 600.0
            
            events = []
            n = len(mask)
            in_event = False
            start_idx = 0
            
            for i in range(n):
                if mask[i] and not in_event:
                    in_event = True
                    start_idx = i
                elif not mask[i] and in_event:
                    in_event = False
                    end_idx = i - 1
                    ev = self._evaluate_event(start_idx, end_idx, is_nagon, dt_sec)
                    if ev:
                        events.append(ev)
            if in_event:
                end_idx = n - 1
                ev = self._evaluate_event(start_idx, end_idx, is_nagon, dt_sec)
                if ev:
                    events.append(ev)

            if not events:
                QtWidgets.QMessageBox.information(self, "Информация", f"Не найдено ни одного события ({event_type_str}).")
                return

            if is_nagon:
                top_height = sorted(events, key=lambda x: x['extreme_val'], reverse=True)[:25]
            else:
                top_height = sorted(events, key=lambda x: x['extreme_val'], reverse=False)[:25]

            top_volume = sorted(events, key=lambda x: x['volume'], reverse=True)[:25]

            with open(file_name, mode='w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(["Признак", "Дата начала", "Дата окончания", "Максимальная высота (м) / Объем (м·сут)", "Продолжительность (дни)"])
                
                for ev in top_height:
                    writer.writerow([
                        "ВЫСОТА",
                        ev['start_time'].strftime('%d.%m.%Y %H:%M'),
                        ev['end_time'].strftime('%d.%m.%Y %H:%M'),
                        f"{ev['extreme_val']:.3f}",
                        f"{ev['duration_days']:.2f}"
                    ])
                
                for ev in top_volume:
                    writer.writerow([
                        "ОБЬЕМ",
                        ev['start_time'].strftime('%d.%m.%Y %H:%M'),
                        ev['end_time'].strftime('%d.%m.%Y %H:%M'),
                        f"{ev['volume']:.3f}",
                        f"{ev['duration_days']:.2f}"
                    ])

            self.status_label.setText(f"Файл успешно сохранен: {os.path.basename(file_name)}")
            QtWidgets.QMessageBox.information(self, "Успех", f"Файл успешно сохранен:\n{file_name}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {str(e)}")

    def _evaluate_event(self, start_idx, end_idx, is_nagon, dt_sec):
        max_limit = len(self.time_stamps) - 1
        start_idx = max(0, min(start_idx, max_limit))
        end_idx = max(0, min(end_idx, max_limit))
        if start_idx > end_idx:
            return None
            
        start_time = datetime.fromtimestamp(self.time_stamps[start_idx])
        end_time = datetime.fromtimestamp(self.time_stamps[end_idx])
        duration_days = (self.time_stamps[end_idx] - self.time_stamps[start_idx]) / 86400.0
        duration_days = max(duration_days, 1.0 / 24.0)
        
        sub_u = self.u_level[start_idx:end_idx+1]
        sub_t1 = self.trend1[start_idx:end_idx+1]
        sub_t2 = self.trend2[start_idx:end_idx+1]
        
        if is_nagon:
            extreme_val = np.max(sub_u) if len(sub_u) > 0 else 0.0
            anomaly = sub_t1 - sub_t2
            volume = np.sum(np.maximum(0, anomaly)) * dt_sec / 86400.0
        else:
            extreme_val = np.min(sub_u) if len(sub_u) > 0 else 0.0
            anomaly = sub_t2 - sub_t1
            volume = np.sum(np.maximum(0, anomaly)) * dt_sec / 86400.0
            
        return {
            'start_time': start_time,
            'end_time': end_time,
            'extreme_val': extreme_val,
            'volume': volume,
            'duration_days': duration_days
        }

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = DonskoiApp()
    window.show()
    sys.exit(app.exec())
