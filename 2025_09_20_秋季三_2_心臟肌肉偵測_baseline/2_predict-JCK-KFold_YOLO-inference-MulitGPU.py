"""
這是一個非常實用且穩健的解決方案，可以在使用 file_list 作為 source 時，有效控制記憶體消耗，同時避免使用 stream=True 可能導致的進度追蹤困難。

我們將在 single_gpu_predict 函式內部新增一個分批器 (Batcher) 邏輯，將分配給該 GPU 的大檔案列表 (file_list)，進一步切割成每次只包含 10 個檔案的小批次 (Sub-batches)。

修改後的 Python 程式碼：內部批次處理 (Sub-Batching)
以下是修正後的 single_gpu_predict 函式，以及相關的更新
"""


"""
od43ctv9ek3retrain-fzkg4:baseline$time python 2_predict-JCK-KFold_YOLO-inference-MulitGPU.py
Active GPUs: [0, 1, 2, 3, 4, 5, 6, 7], N_GPUS: 8, Sub-Batch Size: 10
總檔案數: 16620, 分割成 8 份。
多 GPU 進程完成數: 100%|███████████████████████████████████████████████████████████████████| 8/8 [00:41<00:00,  5.15s/進程]

開始合併 8 個臨時結果檔案...
✅ 最終結果已儲存到 ./predict_txt/V9e-K3-512-V2_epoch38.pt_512_final.txt
總計有 box 的影像張數: 2917, 總計 box 數量: 3185
所有臨時檔案已清除。

real    0m43.769s
user    4m55.561s
sys     0m32.420s
"""

import os
import glob
from ultralytics import YOLO
from multiprocessing import Pool
from tqdm import tqdm

# --- 1. 配置參數 (保持不變) ---
Fold = "V9e-K3-512-V2"
imgsz = 512
PT = 'epoch38.pt'
run_ta = False
TTA = "-ta" if run_ta else ""
PT_PATH = f'runs/detect/{Fold}/weights/{PT}'
SOURCE_DIR = "./datasets/test_all/"

AVAILABLE_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]
N_GPUS = 8
ACTIVE_DEVICES = AVAILABLE_DEVICES[:N_GPUS]

OUTPUT_DIR = f'./predict_txt/{Fold}{TTA}_{PT}_{imgsz}_parallel/'
FINAL_OUTPUT_FILE = f'./predict_txt/{Fold}{TTA}_{PT}_{imgsz}_final.txt'

# --- 新增配置：內部批次大小 ---
SUB_BATCH_SIZE = 10
print(f'Active GPUs: {ACTIVE_DEVICES}, N_GPUS: {N_GPUS}, Sub-Batch Size: {SUB_BATCH_SIZE}')
# ... (其他 print 保持不變)

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 輔助函式 (保持不變) ---
def split_data(source_dir, n_splits):
    # ... (split_data 函式內容保持不變)
    all_files = glob.glob(os.path.join(source_dir, '*.png'))
    
    if not all_files:
        print(f"錯誤: 在 {source_dir} 中找不到任何 .png 檔案。")
        return []
    
    total_files = len(all_files)
    chunk_size = total_files // n_splits
    remainder = total_files % n_splits
    
    splits = []
    current = 0
    for i in range(n_splits):
        end = current + chunk_size + (1 if i < remainder else 0)
        splits.append(all_files[current:end])
        current = end
        
    print(f"總檔案數: {total_files}, 分割成 {n_splits} 份。")
    return splits


# --- 4. 單一 GPU 推論函式 (內部批次處理修正) ---
def single_gpu_predict(args):
    """
    在單個進程中，使用指定的 GPU 對一組檔案進行推論，並進行內部批次切割。
    """
    gpu_id, total_file_list = args
    model = YOLO(PT_PATH)
    
    temp_output_path = os.path.join(OUTPUT_DIR, f'temp_gpu_{gpu_id}.txt')
    
    box_count_1 = 0
    box_count_2 = 0
    
    # --- 【關鍵修正：內部批次切割與循環】 ---
    
    # 設置 tqdm 追蹤該進程的工作進度
    total_files_in_split = len(total_file_list)
    
    # 為了避免 Too many open files 錯誤，我們使用列表收集後一次性寫入
    all_output_lines = []

    # 進行內部批次循環
    for i in tqdm(range(0, total_files_in_split, SUB_BATCH_SIZE),
                  desc=f"GPU {gpu_id} 推論進度 ({total_files_in_split} 張)",
                  leave=False):
        
        # 提取當前子批次的檔案列表
        sub_batch_list = total_file_list[i:i + SUB_BATCH_SIZE]
        
        # 執行推論 (此時 source 列表長度僅為 SUB_BATCH_SIZE)
        results = model.predict(source=sub_batch_list,
                                save=False,
                                imgsz=imgsz,
                                iou=0.75,
                                device=gpu_id,
                                augment=run_ta,
                                verbose=False
                                )
        
        # 處理結果
        for r in results: # results 現在是一個 list of Results
            # 取得圖片檔名（不含副檔名）
            filename = os.path.basename(r.path).split('.png')[0]

            boxes = r.boxes
            box_num = len(boxes.cls.tolist())

            if box_num > 0:
                box_count_1 += 1
                for j in range(box_num):
                    # 提取資訊
                    label = int(boxes.cls[j].item())
                    conf = boxes.conf[j].item()
                    x1, y1, x2, y2 = r.boxes.xyxy[j].tolist()

                    # 建立一行資料
                    line = f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                    all_output_lines.append(line)
                    
                    box_count_2 += 1
    
    # 一次性寫入結果並關閉檔案 (解決 Too many open files)
    with open(temp_output_path, 'w') as output_file:
        output_file.writelines(all_output_lines)
    
    return temp_output_path, box_count_1, box_count_2, gpu_id

# --- 5. 合併結果函式 (新增排序功能) ---
def combine_results(temp_files):
    """將所有臨時檔案合併成一個最終輸出檔案，並依檔名排序。"""
    combined_box_count_1 = 0
    combined_box_count_2 = 0
    all_lines = [] # 用於收集所有結果行
    temp_paths_to_delete = []

    print(f"\n開始合併 {len(temp_files)} 個臨時結果檔案...")
    
    # 步驟 1: 讀取所有臨時檔案並收集所有行
    for temp_path, box_c1, box_c2, _ in temp_files:
        combined_box_count_1 += box_c1
        combined_box_count_2 += box_c2
        
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as infile:
                all_lines.extend(infile.readlines())
            temp_paths_to_delete.append(temp_path) # 收集路徑以便稍後刪除

    # 步驟 2: 排序所有行
    # 排序鍵：取每行的第一個元素 (即檔名: patient00##_####)
    def sort_key(line):
        return line.split(' ')[0]

    all_lines.sort(key=sort_key)
    print(f"已收集並排序 {len(all_lines)} 條檢測結果。")
    
    # 步驟 3: 寫入最終檔案
    with open(FINAL_OUTPUT_FILE, 'w') as outfile:
        outfile.writelines(all_lines)
            
    # 步驟 4: 清理臨時檔案
    for temp_path in temp_paths_to_delete:
        os.remove(temp_path)
            
    print(f'✅ 最終結果已儲存到 {FINAL_OUTPUT_FILE} (並已排序)')
    print(f'總計有 box 的影像張數: {combined_box_count_1}, 總計 box 數量: {combined_box_count_2}')
    print(f'所有臨時檔案已清除。')


# --- 6. 執行主程序 (保持不變) ---
if __name__ == '__main__':
    
    # 1. 數據分割
    file_splits = split_data(SOURCE_DIR, N_GPUS)
    
    if not file_splits:
        print("未找到任何檔案，程序結束。")
        exit()

    # 2. 準備多進程參數
    tasks = []
    for i in range(N_GPUS):
        gpu_id = ACTIVE_DEVICES[i]
        tasks.append((gpu_id, file_splits[i]))

    # 3. 啟動進程池
    with Pool(processes=N_GPUS) as pool:
        # 這裡使用 map 並且在外部使用 tqdm 追蹤進程完成數
        # 內部 tqdm (在 single_gpu_predict 內) 追蹤每個進程的批次進度
        results_from_pool = list(tqdm(pool.imap(single_gpu_predict, tasks),
                                      total=len(tasks),
                                      desc="多 GPU 進程完成數",
                                      unit="進程"))

    # 4. 合併結果
    combine_results(results_from_pool)











"""把張數手動改小事可以跑沒有OOM的 tasks.append((gpu_id, file_splits[i][:3]))"""
##%%time
#"""
#ulimit -n 65536
#⚠️ 請注意： 這個設定只對當前終端機會話有效。您需要在啟動 Jupyter Kernel 或運行腳本的環境中執行此命令。
#"""
#
#import os
#import glob
#from ultralytics import YOLO
#from multiprocessing import Pool, cpu_count
#from tqdm import tqdm
#
## --- 1. 配置參數 ---
## 這些參數在多進程中是共享的，但每個進程會使用自己的 GPU ID
#Fold = "V9e-K3-512-V2"
#imgsz = 512
#PT = 'epoch38.pt'
#run_ta = False
#TTA = "-ta" if run_ta else ""
#PT_PATH = f'runs/detect/{Fold}/weights/{PT}'
#SOURCE_DIR = "./datasets/test_all/"
#
## --- 2. GPU 與分割設定 ---
## 可用的 GPU 列表
#AVAILABLE_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7] 
## 設置要使用的 GPU 數量 (N)
#N_GPUS = 8
#
## 實際使用的 GPU 列表
#ACTIVE_DEVICES = AVAILABLE_DEVICES[:N_GPUS]
#
#print(f'Active GPUs: {ACTIVE_DEVICES}, N_GPUS: {N_GPUS}')
#print(f'run_ta={run_ta}, out={Fold}/{TTA}, imgsz={imgsz}, Model={PT}')
#
## 確保輸出目錄存在
#OUTPUT_DIR = f'./predict_txt/{Fold}{TTA}_{PT}_{imgsz}_parallel/'
#os.makedirs(OUTPUT_DIR, exist_ok=True)
## 最終合併的檔案名稱
#FINAL_OUTPUT_FILE = f'./predict_txt/{Fold}{TTA}_{PT}_{imgsz}_final.txt'
#
#
## --- 3. 數據分割 ---
#def split_data(source_dir, n_splits):
#    """讀取所有圖片檔案路徑，並將其分割成 N 份。"""
#    # 這裡假設您的圖檔是 .png 格式
#    all_files = glob.glob(os.path.join(source_dir, '*.png'))
#    
#    if not all_files:
#        print(f"錯誤: 在 {source_dir} 中找不到任何 .png 檔案。")
#        return []
#    
#    # 計算每份大小
#    total_files = len(all_files)
#    chunk_size = total_files // n_splits
#    remainder = total_files % n_splits
#    
#    splits = []
#    current = 0
#    for i in range(n_splits):
#        end = current + chunk_size + (1 if i < remainder else 0)
#        splits.append(all_files[current:end])
#        current = end
#        
#    print(f"總檔案數: {total_files}, 分割成 {n_splits} 份。")
#    return splits
#
## --- 4. 單一 GPU 推論函式 ---
#def single_gpu_predict(args):
#    """
#    在單個進程中，使用指定的 GPU 對一組檔案進行推論。
#    """
#    gpu_id, file_list = args
#    # 每個進程創建自己的 YOLO 模型實例
#    model = YOLO(PT_PATH)
#    
#    # 創建一個臨時代理檔案來儲存此進程的推論結果
#    temp_output_path = os.path.join(OUTPUT_DIR, f'temp_gpu_{gpu_id}.txt')
#    output_file = open(temp_output_path, 'w')
#    
#    box_count_1 = 0
#    box_count_2 = 0
#
#
#    # 執行推論 - 設置 stream=True
#    # results_generator 是一個生成器
#    # 執行推論 (source 使用檔案列表)
#    results = model.predict(source=file_list,
#                            save=False,
#                            imgsz=imgsz,
#                            iou=0.75,
#                            device=gpu_id, # 關鍵：指定該進程使用的 GPU
#                            #stream=True, # 關鍵：啟用 stream 模式！
#                            augment=run_ta,
#                            verbose=False # 關閉詳細輸出
#                            )
#
#    # 寫入結果
#    # 寫入結果 (直接迭代生成器，無需 len())
#    # 迭代器中的 r 是一個單個的 Results 物件，代表一張圖片的結果
#    # 修正處: 將 for i in range(len(results)): 替換為直接迭代
#    for i in range(len(results)):
##    for i in results_generator:
#        # 取得圖片檔名（不含副檔名）
#        # path = results[i].path 應該包含完整的路徑
#        filename = os.path.basename(results[i].path).split('.png')[0]
#
#        boxes = results[i].boxes
#        box_num = len(boxes.cls.tolist())
#
#        if box_num > 0:
#            box_count_1 += 1
#            for j in range(box_num):
#                # 提取資訊
#                label = int(boxes.cls[j].item())
#                conf = boxes.conf[j].item()
#                x1, y1, x2, y2 = boxes.xyxy[j].tolist()
#
#                # 建立一行資料
#                line = f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
#                output_file.write(line)
#                
#                box_count_2 += 1
#
#    output_file.close()
#    return temp_output_path, box_count_1, box_count_2, gpu_id
#
## --- 5. 合併結果函式 ---
#def combine_results(temp_files):
#    """將所有臨時檔案合併成一個最終輸出檔案。"""
#    combined_box_count_1 = 0
#    combined_box_count_2 = 0
#    
#    print(f"\n開始合併 {len(temp_files)} 個臨時結果檔案...")
#    
#    with open(FINAL_OUTPUT_FILE, 'w') as outfile:
#        for temp_path, box_c1, box_c2, _ in temp_files:
#            combined_box_count_1 += box_c1
#            combined_box_count_2 += box_c2
#            
#            # 讀取臨時檔案內容並寫入最終檔案
#            if os.path.exists(temp_path):
#                with open(temp_path, 'r') as infile:
#                    outfile.write(infile.read())
#                # 刪除臨時檔案
#                os.remove(temp_path)
#            
#    print(f'✅ 最終結果已儲存到 {FINAL_OUTPUT_FILE}')
#    print(f'總計有 box 的影像張數: {combined_box_count_1}, 總計 box 數量: {combined_box_count_2}')
#    print(f'所有臨時檔案已清除。')
#
#
## --- 6. 執行主程序 ---
#if __name__ == '__main__':
#    
#    # 1. 數據分割
#    file_splits = split_data(SOURCE_DIR, N_GPUS)
#    
#    if not file_splits:
#        print("未找到任何檔案，程序結束。")
#        exit()
#
#    # 2. 準備多進程參數 (GPU ID 和對應的檔案列表)
#    tasks = []
#    for i in range(N_GPUS):
#        # 確保每個任務拿到自己的 GPU ID 和檔案列表
#        gpu_id = ACTIVE_DEVICES[i]
#        tasks.append((gpu_id, file_splits[i]))
#
#    # 3. 啟動進程池
#    # 使用 Pool 執行並行推論
#    with Pool(processes=N_GPUS) as pool:
#        # 使用 tqdm 顯示進度
#        results_from_pool = list(tqdm(pool.imap(single_gpu_predict, tasks), 
#                                      total=len(tasks), 
#                                      desc="多 GPU 推論進度"))
#
#    # 4. 合併結果
#    combine_results(results_from_pool)
