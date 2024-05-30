import os
import cv2, torch
from ultralytics import YOLO
import shutil
from diffusers import DiffusionPipeline
from PIL import Image
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

pipe = DiffusionPipeline.from_pretrained(
    "prs-eth/marigold-lcm-v1-0",
    custom_pipeline="marigold_depth_estimation",
    torch_dtype = torch.float16,
    variant = 'fp16'
)
pipe = pipe.to("cuda")

def iou(box1, box2):
    """두 바운딩 박스의 Intersection over Union (IoU) 값을 계산합니다."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection

    return intersection / union if union > 0 else 0


# (필터 1) : 사람이 탐지되지 않는 경우
# (필터 2) : 두 ROI가 threshold 이상 겹치는 경우
def refine_data(dataset_path):
    denoising_steps = 4
    ensemble_size = 5
    processing_res = 768
    match_input_res = True
    
    # 원본 폴더 경로
    source_folder = './musicbank_230923_231201'
    # 전처리 후 txt 파일 경로
    destination_folder = 'refine1'
    
    # 대상 폴더가 없으면 생성
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        
    # 원본 폴더에서 모든 파일 목록을 루프 돌면서 작업
    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):  # .txt 파일인지 확인
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            
            # 파일을 대상 폴더로 복사
            shutil.copy2(source_file, destination_file)
    
    model = YOLO('yolov8m.pt')
    refine_path = '/home/aiuser/junoflow/detection/refine1'
    
    os.makedirs(refine_path, exist_ok=True)
    
    data_files = [f for f in os.listdir(dataset_path) if f.endswith('.txt')]
    
    for index, data_file in enumerate(tqdm(data_files)):
        # print(f"Processing file {index + 1} of {len(data_files)}: {data_file}")
        image_file = data_file.replace('.txt', '.jpg')
        image_path = os.path.join(dataset_path, image_file)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found: {image_path}")
            continue
            
        modified = False        
        with open(os.path.join(dataset_path, data_file), 'r', encoding='cp949') as file:
            lines = file.readlines()
        
        while True:
            new_lines = []
            all_rois = []
            result = []
                
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                x_center, y_center, width, height = map(float, parts[2:])
                
                x_center, y_center, width, height = x_center * image.shape[1], y_center * image.shape[0], width * \
                                                    image.shape[1], height * image.shape[0]
                x1, y1, x2, y2 = int(x_center - width / 2), int(y_center - height / 2), int(x_center + width / 2), int(
                    y_center + height / 2)
                
            # ROI의 크기가 0보다 큰 경우에만 처리
                roi = image[y1:y2, x1:x2]
                if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                    results = model.predict(roi, conf=0.5, classes=[0], verbose=False)  # 클래스 0 ('person')에 대해서만 탐지
                
                    # (1번 필터) : Person이 탐지되지않는 경우 필터링
                    if results[0].boxes.xyxy.size(0) == 0:  # 'person' 탐지되지 않은 경우
                        continue  # 이 ROI는 겹침 검사에서 제외
                    
                    result.append(results)
                    all_rois.append((line_idx, (x1, y1, x2, y2), line))
                
                else:
                    print(f"Invalid ROI size for line {line_idx} in file {data_file}. Skipping.")
                    continue
                
            # 'person'이 탐지된 ROI들 사이의 겹침 확인
            for idx1, (roi1_idx, roi1, line1) in enumerate(all_rois):
                overlap = False
                for idx2, (roi2_idx, roi2, line2) in enumerate(all_rois):
                    if idx1 >= idx2:
                        continue 

                    # (2번 필터) : 두 ROI가 threshold 이상 겹치면 필터링
                    threshold = 0.60 # 60% 이상 겹침
                    if iou(roi1, roi2) > threshold:
                        
                        # roi1 
                        roi_img1 = image[roi1[1]:roi1[3], roi1[0]:roi1[2]] # 앞 roi로 이미지 자르기
                        # bx1 = result[idx1][0].boxes.xyxy.cpu() # 박스 안의 사람으로 판단되는 객체들의 좌표
                        # mid_x1 = (bx1[:, 0] + bx1[:, 2]) / 2 # 박스의 x 중간 좌표
                        
                        # img_center1 = result[idx1][0].orig_shape[1] / 2 # roi1의 좌표 너비의 중간 x좌표
                        # center_bx1 = torch.argmin(torch.abs(mid_x1 - img_center1)) # 중간에 가까운 객체가 박스 주인공 객체
                        
                        # r_img1 = roi_img1[int(bx1[center_bx1][1]):int(bx1[center_bx1][3]), int(bx1[center_bx1][0]):int(bx1[center_bx1][2])]
                    
                        # roi2
                        roi_img2 = image[roi2[1]:roi2[3], roi2[0]:roi2[2]]
                        
                        # bx2 = result[idx2][0].boxes.xyxy.cpu()
                        # mid_x2 = (bx2[:, 0] + bx2[:, 2]) / 2

                        # img_center2 = result[idx2][0].orig_shape[1] / 2
                        # center_bx2 = torch.argmin(torch.abs(mid_x2 - img_center2))
                        # # print(center_bx2)
                        
                        # r_img2 = roi_img2[int(bx2[center_bx2][1]):int(bx2[center_bx2][3]), int(bx2[center_bx2][0]):int(bx2[center_bx2][2])]
                        
                        pipeline_output1 = pipe(
                            Image.fromarray(roi_img1),
                            denoising_steps=denoising_steps,     # optional
                            ensemble_size=ensemble_size,       # optional
                            processing_res=processing_res,     # optional
                            match_input_res=match_input_res,   # optional
                            batch_size=0,           # optional
                            color_map="Spectral",   # optional
                            show_progress_bar=False # optional
                        )
                        
                        pipeline_output2 = pipe(
                            Image.fromarray(roi_img2),
                            denoising_steps=denoising_steps,     # optional
                            ensemble_size=ensemble_size,       # optional
                            processing_res=processing_res,     # optional
                            match_input_res=match_input_res,   # optional
                            batch_size=0,           # optional
                            color_map="Spectral",   # optional
                            show_progress_bar=False # optional
                        )                    
                            
                        # # 각 ROI에서 픽셀의 평균값이 작은 객체가 앞에 있는 것
                        # if np.mean(pipeline_output1.depth_colored) < np.mean(pipeline_output2.depth_colored):
                        if pipeline_output1.depth_np.mean() < pipeline_output2.depth_np.mean():     
                            overlap = True
                            break                
                
                if not overlap:
                    new_lines.append(line1)
            
            if len(lines) == len(new_lines):
                modified = True
                break
            else: lines = new_lines
                
        if modified:
            # 수정된 데이터 파일 저장
            refined_data_file_path = os.path.join(refine_path, data_file)
            with open(refined_data_file_path, 'w', encoding='cp949') as file: # iso-88
                file.writelines(new_lines)

if __name__ == '__main__':
    dataset_path = './musicbank_230923_231201'
    refine_data(dataset_path)