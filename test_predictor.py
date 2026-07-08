from PIL import Image
from backend.services.predictor import predict_image

image = Image.open(r"C:\Users\manda\Documents\AI-MEDICAL-REPORT-ASSISTANT\dataset\chest-xray\test\NORMAL\IM-0001-0001.jpeg")

result = predict_image(image)

print(result)