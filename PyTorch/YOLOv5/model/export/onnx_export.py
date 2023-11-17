"""
Yolov5 ONNX Exporter

Requirements:
    $ pip install -r requirements.txt  onnx onnxruntime  

Usage:
    $ python onnx_converter.py --weights yolov5s.pt 

"""

import argparse, os, platform, sys, time, warnings,inspect,io ,torch, torch.nn as nn
from pathlib import Path
from typing import Optional
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Ensemble(nn.ModuleList):
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output
    
def attempt_load(weights, device=None, inplace=True, fuse=True):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location='cpu')  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model
        model.append(ckpt.fuse().eval() if fuse else ckpt.eval())  # fused or un-fused model in eval mode

    if len(model) == 1:
        return model[-1]  # return model
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model  # return ensemble
    
def export_onnx(model, im, file, opset):
    # YOLOv5 ONNX export
    try:
        import onnx
        print(f'\nstarting export with onnx {onnx.__version__}...')

        f = file[0][:-3] + '.onnx'
        print(f'Path: {f}')
        torch.onnx.export(
            model, 
            im,
            f,
            verbose=False,
            opset_version=opset,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'])

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Metadata
        d = {'stride': int(max(model.stride)), 'names': model.names}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(model_onnx, f)

        print(f'export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        print(f'export failure: {e}')


@torch.no_grad()
def run(
        data=ROOT / 'data/coco128.yaml',  # 'dataset.yaml path'
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        inplace=False,  # set YOLOv5 Detect() inplace=True
        opset=12,  # ONNX: opset version
):
    t = time.time()
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model        
    nc, names = model.nc, model.names  # number of classes, class names
    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    assert nc == len(names), f'Model class count {nc} != len(names) {len(names)}'

    # Input
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    model.eval()  # training mode = no Detect() layer grid construction
    for k, m in model.named_modules():
        m.inplace = inplace
        m.onnx_dynamic = False
        m.export = True
    
    # Exports
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    f = export_onnx(model, im, weights, opset)
    # Finish
    print(f'\nExport complete ({time.time() - t:.2f}s)')
    print(f"\nResults saved!")
    return f  # return list of exported files/dirs

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version. Currently, the ATC tool supports only opset_version=11.')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))
  
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
