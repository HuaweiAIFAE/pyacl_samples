"""
Usage python onnx_exporter.py --model [MODEL_NAME] --output [OUTPUT_NAME]

"""
# -*- coding:utf-8 -*-
import os
os.environ['TF_KERAS'] = '1'
import wget
import argparse
from onnx import save_model
from keras2onnx import convert_keras
import arcface_model

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=str, help='model weight path ')
    parser.add_argument('--model', type=str, help='model name is: arcface')
    parser.add_argument('--output', type=str, help='output onnx model name')

    opt = parser.parse_args()
    return opt

def weight_download(model,model_name):
	cwd = os.getcwd()

	model_dict = {'arcface': 'https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5'}
	
	filename = wget.download(model_dict[model_name])
	
	model.load_weights(f"{cwd}/{filename}")
	os.remove(f'{cwd}/{filename}')
	return model
	
def models(model_name,weight):
	if model_name == 'arcface':
		model = arcface_model.Arcface()
		model = weight_download(model,model_name)
		return model
	else:
		raise NameError(f'{model_name} is not in the model list or written model name wrong!')

def onnx_extraction(model,output_name):
	onnx_model = convert_keras(model, model.name, target_opset=11)
	save_model(onnx_model, f"{output_name}.onnx")

def run(weights,model,output):
	selected_model = models(model,weights)
	onnx_extraction(selected_model,output)

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)