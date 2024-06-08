```
import os

# 하이퍼파라미터 설정 파일 생성
hyp_finetune_yaml = """
lr0: 0.005
lrf: 0.1
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 2.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.20
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10.0
translate: 0.1
scale: 0.5
shear: 2.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
"""

with open("data/hyp.finetune.yaml", "w") as f:
    f.write(hyp_finetune_yaml)

# 배치 크기 리스트
batch_sizes = [8, 16, 32]

# 데이터셋 경로
data_path = "/content/drive/MyDrive/your_dataset/data.yaml"

# 학습 및 검증 반복
for batch_size in batch_sizes:
    # 출력 디렉토리 설정
    output_dir = f"runs/train/exp_batch_{batch_size}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 학습 스크립트 실행
    !python train.py --img 416 --batch {batch_size} --epochs 50 --data {data_path} --cfg cfg/training/yolov7-tiny.yaml --weights yolov7-tiny.pt --device 0 --hyp data/hyp.finetune.yaml --project {output_dir}
    
    # 검증 스크립트 실행
    !python test.py --data {data_path} --img 416 --batch {batch_size} --conf 0.001 --iou 0.65 --device 0 --weights {output_dir}/weights/best.pt --task val --project {output_dir}


```
