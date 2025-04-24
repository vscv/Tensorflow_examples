# 2021-12-28
# Generator to create command line for sending to task CCS.
#

log_dir_name=['TrainSaveDir-##']

m_start=0
m_end=26


weight = ["imagenet1k"] # and imagenet21k
crop = ["crop", "resize"]
lr_name = ['plateau', "plateau", "WCD", "CDR"]
augment = [None, 'AA', 'RA']


img_height = ['600', '512']
img_width = img_height

count = 0
for wt in weight:
    for cp in crop:
        for lr in lr_name:
            for ag in augment:
                for ih in img_height:
                    
                    print(f'{m_start} {m_end} {wt} {cp} {lr} {ag} {ih} {ih}')
                    count += 1
print(f'Total benchs: {count}')


"""
0 26 imagenet1k crop plateau None 600 600
0 26 imagenet1k crop plateau None 512 512
0 26 imagenet1k crop plateau AA 600 600
0 26 imagenet1k crop plateau AA 512 512
0 26 imagenet1k crop plateau RA 600 600
0 26 imagenet1k crop plateau RA 512 512
0 26 imagenet1k crop plateau None 600 600
0 26 imagenet1k crop plateau None 512 512
0 26 imagenet1k crop plateau AA 600 600
0 26 imagenet1k crop plateau AA 512 512
0 26 imagenet1k crop plateau RA 600 600
0 26 imagenet1k crop plateau RA 512 512
0 26 imagenet1k crop WCD None 600 600
0 26 imagenet1k crop WCD None 512 512
0 26 imagenet1k crop WCD AA 600 600
0 26 imagenet1k crop WCD AA 512 512
0 26 imagenet1k crop WCD RA 600 600
0 26 imagenet1k crop WCD RA 512 512
0 26 imagenet1k crop CDR None 600 600
0 26 imagenet1k crop CDR None 512 512
0 26 imagenet1k crop CDR AA 600 600
0 26 imagenet1k crop CDR AA 512 512
0 26 imagenet1k crop CDR RA 600 600
0 26 imagenet1k crop CDR RA 512 512
0 26 imagenet1k resize plateau None 600 600
0 26 imagenet1k resize plateau None 512 512
0 26 imagenet1k resize plateau AA 600 600
0 26 imagenet1k resize plateau AA 512 512
0 26 imagenet1k resize plateau RA 600 600
0 26 imagenet1k resize plateau RA 512 512
0 26 imagenet1k resize plateau None 600 600
0 26 imagenet1k resize plateau None 512 512
0 26 imagenet1k resize plateau AA 600 600
0 26 imagenet1k resize plateau AA 512 512
0 26 imagenet1k resize plateau RA 600 600
0 26 imagenet1k resize plateau RA 512 512
0 26 imagenet1k resize WCD None 600 600
0 26 imagenet1k resize WCD None 512 512
0 26 imagenet1k resize WCD AA 600 600
0 26 imagenet1k resize WCD AA 512 512
0 26 imagenet1k resize WCD RA 600 600
0 26 imagenet1k resize WCD RA 512 512
0 26 imagenet1k resize CDR None 600 600
0 26 imagenet1k resize CDR None 512 512
0 26 imagenet1k resize CDR AA 600 600
0 26 imagenet1k resize CDR AA 512 512
0 26 imagenet1k resize CDR RA 600 600
0 26 imagenet1k resize CDR RA 512 512
Total benchs: 48
"""
