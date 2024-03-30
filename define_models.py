import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from archs.FourLLIE_arch import FourLLIE
from archs.CCNet_arch import BCNet
import time

def define_model(type,data='huawei'):
    checkpoint = torch.load(f'./checkpoint/{type}_{data}.pth')
    if type == 'FourLLIE':
        model = FourLLIE().cuda()
    elif type == 'CCNet':
        model = BCNet().cuda()
    model.load_state_dict(checkpoint)
    return model

# convert PIL to Tensor
def pre_process_data(img):
    img = (np.asarray(img)/255.)
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).cuda()
    return img

def process(type,img, ref = None, saturation = None, data = 'huawei'):
    model = define_model(type,data)
    model.eval()
    to_pil = transforms.ToPILImage()
    img = pre_process_data(img)
    if ref is None and saturation is None:
        t1 = time.time()
        with torch.no_grad():
            res = model(img)
        t2 = time.time()
        return to_pil(torch.clamp(res[0].squeeze(0),0,1)), t2-t1
    else:
        # ref = ref.resize((img.shape[3],img.shape[2]))
        ref = pre_process_data(ref)
        t1 = time.time()
        with torch.no_grad():
            res = model(img,img,ref,saturation)
        t2 = time.time()
        return to_pil(torch.clamp(res[0].squeeze(0),0,1)), t2-t1

if __name__ =='__main__':
    a = Image.open('/home/gyn/wcx_2403/data/SYSU-FVL-T2/ui_imgs/refs/ref_5k1.png')
    print(process('CCNet',a,a,1))