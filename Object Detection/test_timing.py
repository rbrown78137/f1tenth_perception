import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

yolo_model = torch.jit.load("saved_models/yolo.pt")
yolo_model.to(device)

for i in range(20):
    test_input = torch.rand((1, 3, 256, 256)).to(device)
    start_body = time.time()
    yolo_output = yolo_model(test_input)
    convert_to_cpu = yolo_output[0].to("cpu")
    end_body = time.time()

    print(f"Object Detection Time: {end_body-start_body}")
