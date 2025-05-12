from transformers import pipeline
from PIL import Image
import requests

pipe = pipeline(task="depth-estimation", model="xingyang1/Distill-Any-Depth-Large-hf")
