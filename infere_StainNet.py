from options.infere_options import InfereOptions
from models import StainNet
import torch
from PIL import Image
import torchvision
import os

def get_img(path):
    transforms = torchvision.transforms.ToTensor()  

    try:
        img = Image.open(path).convert('RGB')
    except FileNotFoundError:
        return None
    img = transforms(img).unsqueeze(0)

    return img

def loader(args):  
    if args.one_sample is not None:
        img = get_img(args.one_sample)
        if img is None:
            raise RuntimeError(f"File {args.one_sample} does not exist")
        yield img, args.one_sample.split("/")[-1]
        return
    df = pd.read_csv(args.csv_path) 
    for row in df:
        img = get_img(os.path.join(args.dataroot, row["filename"]))
        if img is None:
            continue
        yield img, row["filename"]
        

def main(args):
    model = StainNet.load_from_checkpoint(args.checkpoints_dir)
    model.eval()
    for batch_idx, (batch, filename) in enumerate(loader(args)):
        pred = model.predict_step(batch, batch_idx)
        path = os.path.join(args.results, filename)
        torchvision.utils.save_image(pred, path)

if __name__ == "__main__":
    parser = InfereOptions().parser
    args = parser.parse_args()

    main(args)