import json
import time
from model import get_model
from utils import tensor2im
from PIL import Image
from torchvision import transforms
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

pixel_art_app = FastAPI()
pixel_art_app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@pixel_art_app.post('/pixel-art')
async def get_pixelized_image(img: UploadFile = File(...)):
    net = get_model()
    image = Image.open(BytesIO(img.file.read())).convert('RGB')
    transform_A = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    A_img = transform_A(image)
    A_img = A_img.unsqueeze(0).to("cpu")
    fake_img_tensor = net(A_img)
    logger.info("Conversion done! Saving image..")
    fake_img = tensor2im(fake_img_tensor)
    fake_image = Image.fromarray(fake_img)
    bytes_image = BytesIO()
    fake_image.save(bytes_image, format='PNG')

    return Response(content=bytes_image.getvalue(),  media_type="image/png")
