import streamlit as st
import pandas as pd
import numpy as np
import base64
from model import *
from utils import *
from io import BytesIO
from torchvision import transforms

st.set_page_config(layout='wide')
_, mid, right = st.columns(3)
mid.title('Pixel Art')

hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)




@st.cache
def get_model():
    norm_layer = get_norm_layer(norm_type='instance')
    net = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gpu_ids = [0] if torch.cuda.is_available() else []

    state_dict = torch.load("latest_net_G_A.pth", map_location=device)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    net.load_state_dict(state_dict)

    return net


net = get_model()

image_file = mid.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])


@st.cache
def convert_img(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = buffered.getvalue()
    return img_str


if image_file is not None:
      image = Image.open(image_file).convert("RGB")

      transform_A = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
      A_img = transform_A(image)
      A_img = A_img.unsqueeze(0).to("cpu")
      fake_img_tensor = net(A_img)
      fake_img = tensor2im(fake_img_tensor)
      fake_image = Image.fromarray(fake_img)

      mid.image(fake_image, caption=f"Pixelized Image")

      fake_image_convert = convert_img(fake_image)

      _, _, download, _ = st.columns([1, 1, 1.5, 1])

      download.download_button(
              label="Download Pixelized image",
              data=fake_image_convert,
              file_name="result.png",
              mime="image/png"
          )


