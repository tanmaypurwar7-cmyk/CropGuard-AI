import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from huggingface_hub import hf_hub_download


class PlantDiseaseClassifier(nn.Module):

    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()

        self.backbone = models.efficientnet_b2(weights=None)

        num_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Identity()

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_features // 4),
            nn.ReLU(),
            nn.Linear(num_features // 4, num_features),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),

            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(dropout_rate * 0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Dropout(dropout_rate * 0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        features = self.backbone.features(x)

        pooled = F.adaptive_avg_pool2d(features, 1)
        pooled = torch.flatten(pooled, 1)

        attention_weights = self.attention(features)

        attended_features = pooled * attention_weights

        output = self.classifier(attended_features)

        return output


# Disease classes
class_names = [
    "Apple_Apple_Scab",
    "Apple_Black_Rot",
    "Apple_Cedar_Apple_Rust",
    "Apple_Healthy",

    "Corn_(maize)_Cercospora_Leaf_Spot",
    "Corn_(maize)_Common_Rust_",
    "Corn_(maize)_Healthy",
    "Corn_(maize)_Northern_Leaf_Blight",

    "Tomato_Bacterial_Spot",
    "Tomato_Early_Blight",
    "Tomato_Healthy",
    "Tomato_Late_Blight",
    "Tomato_Septoria_Leaf_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus"
]


print("Loading AI model...")

model_path = hf_hub_download(
    repo_id="Abuzaid01/plant-disease-classifier",
    filename="model.pth"
)

checkpoint = torch.load(
    model_path,
    map_location="cpu"
)

model = PlantDiseaseClassifier(
    num_classes=len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("AI model ready!")


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict(image):

    image = image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )[0]

    top_probs, top_indices = torch.topk(
        probabilities,
        3
    )

    results = []

    for i in range(3):

        disease = class_names[
            top_indices[i].item()
        ]

        confidence = top_probs[i].item() * 100

        results.append({
            "disease": disease,
            "confidence": round(confidence, 2)
        })

    return results