################################
#                              #
# All in one ToolKit for PCGA  #
# 2023.10.26                   #
#                              #
################################

import numpy as np


# PCGA tk
#reshape_FV_value_list
def reshape_FV_value_list(fv_list):
    rank = [
        4,
        6,
        8,
        8,
        8,
        8,
        6,
        4]
#     gt_ = [
#      23 21 24 18
#      24 24 25 20 23 27
#      26 27 22 25 24 26 26 23
#      29 23 27 28  0 27 26 25
#      25  0 28 28 29 29 29 28
#      30 30 30 29 26 29 28 27
#      29 27 23 27 28 27
#      25 27 24 24]

#     for i, idx in enumerate(rank):
#         print(f"rank: {i} {idx}")

#         arr_idx_1 = np.pad(sampl_arr_1[i:idx], ((8-idx, 8-idx)), 'constant',constant_values=(0))

#         print(arr_idx_1)

    tmp_arr = []
    steps = 0
    h_rows = 4
    for i, idx in enumerate(rank):
        #print(f"rank: {i} {idx}")

#         # 8-rows
#         if i == 0 and idx == 4:
#             arr_idx_1 = np.pad(fv_list[0:idx], ((2, 2)), 'constant',constant_values=(0))
#         if i == 1 and idx == 6:
#             arr_idx_1 = np.pad(fv_list[4:4+idx], ((1, 1)), 'constant',constant_values=(0))
#         if i == 2 and idx == 8:
#             arr_idx_1 = np.pad(fv_list[10:10+idx], ((8-idx, 8-idx)), 'constant',constant_values=(0))
#         if i == 3 and idx == 8:
#             arr_idx_1 = np.pad(fv_list[18:18+idx], ((8-idx, 8-idx)), 'constant',constant_values=(0))
#         if i == 4 and idx == 8:
#             arr_idx_1 = np.pad(fv_list[26:26+idx], ((8-idx, 8-idx)), 'constant',constant_values=(0))
#         if i == 5 and idx == 8:
#             arr_idx_1 = np.pad(fv_list[34:34+idx], ((8-idx, 8-idx)), 'constant',constant_values=(0))
#         if i == 6 and idx == 6:
#             arr_idx_1 = np.pad(fv_list[42:42+idx], ((1, 1)), 'constant',constant_values=(0))
#         if i == 7 and idx == 4:
#             arr_idx_1 = np.pad(fv_list[48:48+idx], ((2, 2)), 'constant',constant_values=(0))


        # 8-rows in onece, half-rows = 4
        arr_idx_1 = np.pad(fv_list[steps : steps + idx], ((h_rows - int(idx/2), h_rows - int(idx/2))), 'constant', constant_values=(40))

        steps += idx

        tmp_arr.append(arr_idx_1)

        #print(arr_idx_1)
    #print(tmp_arr)
    return np.vstack(tmp_arr)

def reshape_FV_value_to_org_list(fv_list):
    rank = [
        4,
        6,
        8,
        8,
        8,
        8,
        6,
        4]


    tmp_arr = []
    steps = 0
    h_rows = 4
    for i, idx in enumerate(rank):
        #print(f"rank: {i} {idx}")

        # 8-rows in onece, half-rows = 4
        arr_idx_1 = fv_list[steps : steps + idx]

        steps += idx

        tmp_arr.append(arr_idx_1)

        #print(arr_idx_1)
    #print(tmp_arr)
    return tmp_arr
