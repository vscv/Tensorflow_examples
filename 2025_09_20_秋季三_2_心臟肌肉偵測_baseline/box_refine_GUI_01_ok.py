import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
from copy import deepcopy

# 右鍵選單問題
import tkinter as tk
from tkinter import Menu as TkMenu


# 存檔問題
from matplotlib.widgets import Button

# -- 中文字型問題 --
#import matplotlib
#matplotlib.rc('font', family='Microsoft JhengHei')

# --- 1. 數據讀取與解析函式 --- 重複使用

def load_yolo_submission(file_path):
    """
    讀取 YOLO 競賽提交檔，並解析成 DataFrame。
    欄位順序：[檔名, 類別, 信心指數, x1, y1, x2, y2]
    """
    try:
        # 由於數據沒有列名，我們手動指定它們
        df = pd.read_csv(
            file_path,
            sep=' ',
            header=None,
            names=['filename', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2']
        )
        return df
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}")
        return None
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return None
        
# --- 2. 數據處理與轉換函式 --- 重複使用

def preprocess_data(df):
    """
    從 DataFrame 中提取病人ID和影像Z軸編號。
    """
    if df is None:
        return None

    # 提取病人ID (patient00##) 和 Z軸編號 (####)
    # 使用正規表達式提取：
    # patient(00\d{2}) -> 擷取 00xx 部分
    # _(\d{4}) -> 擷取四位數字部分
    df['patient_id'] = df['filename'].str.extract(r'(patient\d{4})')
    df['z_index'] = df['filename'].str.extract(r'_(\d{4})').astype(int)

    # 計算預測框的 Y 軸高度（用於視覺化）
    # y_center: 預測框的中心點 Y 座標
    # y_min: 預測框的最小 Y 座標 (y1)
    # y_max: 預測框的最大 Y 座標 (y2)
    df['y_min'] = df[['y1', 'y2']].min(axis=1) # 確保 y_min 總是較小的值
    df['y_max'] = df[['y1', 'y2']].max(axis=1) # 確保 y_max 總是較大的值
    df['y_center'] = (df['y_min'] + df['y_max']) / 2
    # X_
    df['x_min'] = df[['x1', 'x2']].min(axis=1)
    df['x_max'] = df[['x1', 'x2']].max(axis=1)
    df['x_center'] = (df['x_min'] + df['x_max']) / 2
    
    return df

        
        
        
class BoxEditorGUI:
    def __init__(self, df_clean, output_dir="edited_submissions"):
        self.df = df_clean.copy()
        self.original_df = df_clean.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        

        
        
        # 提取所有病人
        self.patients = sorted(self.df['patient_id'].unique())
        self.current_patient_idx = 0
        
        # 編輯歷史 (支援 undo/redo)
        self.history = []
        self.history_idx = -1
        self.max_history = 50
        
        # 複製緩衝區
        self.clipboard = None
        
        # 圖表
        self.fig, (self.ax_y, self.ax_x) = plt.subplots(1, 2, figsize=(20, 10))
        self.fig.canvas.manager.set_window_title("Box check tool")
        self.fig.suptitle("Click red center to 'Del' | left click copy/past | Ctrl+Z undo", fontsize=12)
        
        # 事件綁定
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        #self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        # 修改 on_key
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # 右鍵選單
        self.context_menu = None
        
        self.redraw()
        plt.show()

    def get_current_patient(self):
        return self.patients[self.current_patient_idx]
    
    def get_current_data(self):
        pid = self.get_current_patient()
        return self.df[self.df['patient_id'] == pid].copy().sort_values('z_index')
    
    def save_history(self):
        """儲存當前狀態到歷史"""
        current_state = self.df.copy()
        self.history = self.history[:self.history_idx + 1]
        self.history.append(current_state)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.history_idx = len(self.history) - 1

    def undo(self):
        if self.history_idx > 0:
            self.history_idx -= 1
            self.df = self.history[self.history_idx].copy()
            self.redraw()
    
    def redo(self):
        if self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            self.df = self.history[self.history_idx].copy()
            self.redraw()

    def redraw(self):
        self.ax_y.clear()
        self.ax_x.clear()
        
        data = self.get_current_data()
        if data.empty:
            self.ax_y.text(0.5, 0.5, "無資料", transform=self.ax_y.transAxes, ha='center')
            self.ax_x.text(0.5, 0.5, "無資料", transform=self.ax_x.transAxes, ha='center')
            self.fig.canvas.draw()
            return
        
        z = data['z_index'].values
        y_min, y_max = data['y_min'].values, data['y_max'].values
        y_center = data['y_center'].values
        x_min, x_max = data['x_min'].values, data['x_max'].values
        x_center = data['x_center'].values
        conf = data['confidence'].values
        
        # === Y/Z 圖 ===
        for zi, ymin, ymax, c in zip(z, y_min, y_max, conf):
            self.ax_y.plot([zi, zi], [ymin, ymax], color='blue', alpha=c, lw=1.5)
            self.ax_y.scatter(zi, (ymin+ymax)/2, s=50*c + 10, c='red', zorder=5, picker=True)
        
        self.ax_y.set_xlabel("Z-Index (CT Slice)")
        self.ax_y.set_ylabel("Y Pixel (Top → Bottom)")
        self.ax_y.set_title(f"Y/Z Distribution - {self.get_current_patient()}")
        self.ax_y.invert_yaxis()
        self.ax_y.grid(True, alpha=0.3)
        
        # --- 【修正處】強制設定 Y 軸範圍並反轉 ---
        # 即使共享 Y 軸，也強制設定範圍並反轉，確保所有子圖統一
        y_lim_min = 150   #統一設定Y軸上下限
        y_lim_max = 450   #統一設定Y軸上下限
        self.ax_y.set_ylim(y_lim_max, y_lim_min) # 注意：這裡的 max, min 順序是反的！
                                         # (y_lim_max 在第一個參數，讓它變成 Y 軸頂部)
        
        
        
        
        # === X/Z 圖 ===
        for zi, xmin, xmax, c in zip(z, x_min, x_max, conf):
            self.ax_x.plot([zi, zi], [xmin, xmax], color='green', alpha=c, lw=1.5)
            self.ax_x.scatter(zi, (xmin+xmax)/2, s=50*c + 10, c='red', zorder=5, picker=True)
        
        self.ax_x.set_xlabel("Z-Index (CT Slice)")
        self.ax_x.set_ylabel("X Pixel (Left → Right)")
        self.ax_x.set_title(f"X/Z Distribution - {self.get_current_patient()}")
        self.ax_x.grid(True, alpha=0.3)
        
                # --- 【修正處】強制設定 Y 軸範圍並反轉 ---
        # 即使共享 Y 軸，也強制設定範圍並反轉，確保所有子圖統一
        x_lim_min = 150   #統一設定Y軸上下限
        x_lim_max = 450   #統一設定Y軸上下限
        self.ax_x.set_ylim(x_lim_max, x_lim_min) # 注意：這裡的 max, min 順序是反的！
                                         # (y_lim_max 在第一個參數，讓它變成 Y 軸頂部)
                                         
                             
        # 存檔案扭
        # 另外增加一個save按鍵
        #from matplotlib.widgets import Button
        ax_save = plt.axes([0.43, 0.01, 0.18, 0.06])
        self.btn_save = Button(ax_save, 'Save current patient\n(Ctrl+S)', color='#90EE90')
        self.btn_save.on_clicked(lambda x: self.save_current_patient())
        
        # 狀態列
        self.fig.text(0.01, 0.01, f"Patient: {self.get_current_patient()} | Nbox: {len(data)} | "
                                 f"history: {self.history_idx+1}/{len(self.history)}",
                     fontsize=10, transform=self.fig.transFigure)
        
        self.fig.canvas.draw()

#    def on_key(self, event):
#        if event.key == 'left' and self.current_patient_idx > 0:
#            self.current_patient_idx -= 1
#            self.redraw()
#        elif event.key == 'right' and self.current_patient_idx < len(self.patients)-1:
#            self.current_patient_idx += 1
#            self.redraw()
#        elif event.key in ['ctrl+z', 'cmd+z']:
#            self.undo()
#        elif event.key in ['ctrl+y', 'cmd+y']:
#            self.redo()
#        elif event.key == 's':
#            self.save_current_patient()

    def on_key(self, event):
        if event.key in ['ctrl+s', 'cmd+s']:
            self.save_current_patient()
            try:
                if hasattr(event, 'guiEvent') and event.guiEvent:
                    event.guiEvent.Skip()
            except:
                pass
            return

        if event.key == 'left' and self.current_patient_idx > 0:
            self.current_patient_idx -= 1
            self.redraw()
        elif event.key == 'right' and self.current_patient_idx < len(self.patients)-1:
            self.current_patient_idx += 1
            self.redraw()
        elif event.key in ['ctrl+z', 'cmd+z']:
            self.undo()
        elif event.key in ['ctrl+y', 'cmd+y']:
            self.redo()
            

    def on_click(self, event):
        if event.inaxes not in [self.ax_y, self.ax_x]:
            return
        if event.button == 1:  # 左鍵：刪除
            self.delete_nearest_box(event)
        elif event.button == 3:  # 右鍵：選單
            self.show_context_menu(event)

    def delete_nearest_box(self, event):
        data = self.get_current_data()
        z = data['z_index'].values
        if event.inaxes == self.ax_y:
            centers = data['y_center'].values
        else:
            centers = data['x_center'].values
        
        # 找最近的點
        dist = np.abs(z - event.xdata) + np.abs(centers - event.ydata) * 0.1
        idx = np.argmin(dist)
        global_idx = data.index[idx]
        
        self.save_history()
        self.df = self.df.drop(global_idx)
        print(f"刪除: {data.iloc[idx]['filename']} (conf={data.iloc[idx]['confidence']:.4f})")
        self.redraw()

#    def show_context_menu(self, event):
#        from matplotlib.widgets import Menu
#        data = self.get_current_data()
#        z = data['z_index'].values
#        centers = data['y_center' if event.inaxes == self.ax_y else 'x_center'].values
#        dist = np.abs(z - event.xdata) + np.abs(centers - event.ydata) * 0.1
#        idx = np.argmin(dist)
#        self.selected_row = data.iloc[idx]
#        self.selected_global_idx = data.index[idx]
#        
#        menu = Menu(self.fig, [
#            ("copy this box", self.copy_box),
#            ("past on here (replace)", lambda: self.paste_box(replace=True)),
#            ("past as new box", lambda: self.paste_box(replace=False)),
#        ])
#        menu.popup(event)

    #原因是：matplotlib.widgets.Menu 從 3.4 版開始就已經被移除（deprecated → removed），現在官方已不再提供這個類別。
#    def show_context_menu(self, event):
#        # 找出最近的 box
#        data = self.get_current_data()
#        if data.empty:
#            return
#        z = data['z_index'].values
#        centers = data['y_center' if event.inaxes == self.ax_y else 'x_center'].values
#        dist = np.abs(z - event.xdata) + np.abs(centers - event.ydata) * 0.1
#        idx = np.argmin(dist)
#        self.selected_row = data.iloc[idx].copy()
#        self.selected_global_idx = data.index[idx]
#
#        # 使用 tkinter 右鍵選單（跨平台、穩定）
#        try:
#            menu = plt.tk.Tk()
#            menu.withdraw()  # 隱藏主窗口
#            rmenu = plt.tk.Menu(menu, tearoff=0)
#            rmenu.add_command(label="複製此 box", command=self.copy_box)
#            rmenu.add_command(label="貼上並替換此位置", command=lambda: self.paste_box(replace=True))
#            rmenu.add_command(label="貼上為新 box", command=lambda: self.paste_box(replace=False))
#            rmenu.add_separator()
#            rmenu.add_command(label="刪除此 box", command=lambda: self.delete_selected_box())
#            
#            # 捕捉關閉事件，避免殘留
#            try:
#                rmenu.tk_popup(int(event.x_root), int(event.y_root))
#            finally:
#                rmenu.grab_release()
#        except Exception as e:
#            print("右鍵選單錯誤（tkinter）:", e)
#            # 後備：用 print 提示
#            print("請用左鍵刪除，右鍵功能暫不可用")
 


    def show_context_menu(self, event):
        """右鍵選單：使用純 tkinter（不依賴 matplotlib.tk）"""
        data = self.get_current_data()
        if data.empty:
            return

        # 找最近的 box
        z = data['z_index'].values
        centers = data['y_center' if event.inaxes == self.ax_y else 'x_center'].values
        dist = np.abs(z - event.xdata) + np.abs(centers - event.ydata) * 0.1
        idx = np.argmin(dist)
        self.selected_row = data.iloc[idx].copy()
        self.selected_global_idx = data.index[idx]

        # === 純 tkinter 右鍵選單（最穩定）===
        root = tk.Tk()
        root.withdraw()  # 隱藏主窗口
        menu = TkMenu(root, tearoff=0)
        menu.add_command(label="複製此 box", command=self.copy_box)
        menu.add_command(label="貼上並替換此位置", command=lambda: self.paste_box(replace=True))
        menu.add_command(label="貼上為新 box", command=lambda: self.paste_box(replace=False))
        menu.add_separator()
        menu.add_command(label="刪除此 box", command=self.delete_selected_box)

        try:
            # 這兩行是關鍵：正確轉換座標
            menu.tk_popup(int(event.x_root), int(event.y_root))
        finally:
            menu.grab_release()
            root.destroy()  # 一定要銷毀，避免殘留
            
            
 
 
    def delete_selected_box(self):
        if hasattr(self, 'selected_global_idx'):
            self.save_history()
            self.df = self.df.drop(self.selected_global_idx)
            print(f"已刪除: {self.selected_row['filename']}")
            self.redraw()

    def copy_box(self):
        self.clipboard = self.selected_row.copy()
        print(f"已複製: {self.clipboard['filename']} (conf={self.clipboard['confidence']:.4f})")

    def paste_box(self, replace=True):
        if self.clipboard is None:
            print("剪貼簿為空！")
            return
        
        data = self.get_current_data()
        z_click = int(round(self.selected_row['z_index']))  # 點擊的 z
        new_row = self.clipboard.copy()
        new_row['z_index'] = z_click
        new_row['filename'] = new_row['filename'].str.replace(r'_\d{4}', f'_{z_click:04d}', regex=True)
        
        self.save_history()
        if replace:
            # 替換該 z 的所有 box
            self.df = self.df[~((self.df['patient_id'] == self.get_current_patient()) & 
                               (self.df['z_index'] == z_click))]
        
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"past ok on z={z_click}")
        self.redraw()

    def save_current_patient(self):
        pid = self.get_current_patient()
        patient_data = self.df[self.df['patient_id'] == pid].copy()
        if patient_data.empty:
            print(f"病人 {pid} 無資料，跳過儲存。")
            return
        
        # 合併所有病人（維持原排序）
        other_patients = self.df[self.df['patient_id'] != pid]
        final_df = pd.concat([other_patients, patient_data], ignore_index=True)
        final_df = final_df.sort_values('confidence', ascending=False)
        
        # 輸出欄位
        output_cols = ['filename', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2']
        output_df = final_df[output_cols].copy()
        output_df['x1'] = output_df['x1'].astype(int)
        output_df['y1'] = output_df['y1'].astype(int)
        output_df['x2'] = output_df['x2'].astype(int)
        output_df['y2'] = output_df['y2'].astype(int)
        
        output_path = os.path.join(self.output_dir, f"edited_submission.txt")
        output_df.to_csv(output_path, sep=' ', index=False, header=False, float_format='%.4f')
        print(f"已儲存: {output_path} (含 {len(patient_data)} 個 box)")

# ======================
# 使用方式
# ======================
# 1. 載入你的 fused txt
#df_raw = load_yolo_submission("fused_predictions_v9e-W7-1112_V2_order_refine-01.txt")
df_raw = load_yolo_submission("SA_V1_1113.txt")
df_clean = preprocess_data(df_raw)

# 2. 啟動 GUI
editor = BoxEditorGUI(df_clean)

# 3. 操作：
#    ← → 切換病人
#    左鍵點擊中心點 → 刪除
#    右鍵 → 複製/貼上
#    Ctrl+S → 儲存目前病人
#    Ctrl+Z / Y → 復原/重做
