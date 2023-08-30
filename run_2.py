from fastsam import FastSAM, FastSAMSegment
import os, ast, torch, time
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy

img_dim = 256
x_ = 128 * img_dim/1024
y_ = 340 * img_dim/1024
w_ = 768 * img_dim/1024
h_ = 340 * img_dim/1024
box_prompt = f"[[{x_},{y_},{w_},{h_}]]"
print(box_prompt)
src_dir = f"./images/{img_dim}/"
output = f"./output/{img_dim}/"
image_files = os.listdir(src_dir)
model_path = "./weights/FastSAM.pt"
model = FastSAM(model_path)
box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(box_prompt))
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    )

start_time = time.time()
img_path = os.path.join(src_dir, image_files[0])
input = Image.open(img_path)
input = input.convert("RGB")
everything_results = model(
    input,
    device=device,
    retina_masks=True,
    imgsz=img_dim,
    conf=0.4,
    iou=0.9
    )
prompt_process = FastSAMSegment(input, everything_results, device=device)
ann = prompt_process.box_prompt(bboxes=box_prompt)
bboxes = box_prompt
prompt_process.plot(
    annotations=ann,
    output_path=output+img_path.split("/")[-1],
    bboxes = bboxes,
    points = None,
    point_label = None,
    withContours=False,
    better_quality=False,
)
end_time = time.time()
elapsed_time = end_time - start_time
time_per_image = elapsed_time/len(image_files)
print(time_per_image)
