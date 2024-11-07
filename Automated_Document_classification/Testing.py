import io
import streamlit as st
from PIL import Image
import easyocr
from transformers import (LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification)
from tqdm import tqdm
import torch
import numpy as np
import pytorch_lightning as pl
from torchmetrics import Accuracy
from pathlib import Path

def create_bbox(bbox):
    left, top = bbox[0]
    right, bottom = bbox[2]
    return [int(left), int(top), int(right), int(bottom)]


def scale_bounding_box(box, width_scale, height_scale):
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale)
    ]



feature_Extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_Extractor , tokenizer)
   

classes=sorted([p.name for p in list(Path('D:\\real_world_projects\\Document_classifi\\Dataset').glob('*'))])

class ModelModule(pl.LightningModule):
    def __init__(self, n_classes: int):
        super().__init__()
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=n_classes
        )
        self.train_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        labels = batch["labels"].long()
        outputs = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["bbox"],
            batch["pixel_values"],
            labels
        )
        loss = outputs.loss

        print(f"Processing batch {batch_idx}")

        self.log("train_loss", loss)
        self.train_accuracy(outputs.logits, labels)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        labels = batch["labels"].long()
        outputs = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["bbox"],
            batch["pixel_values"],
            labels
        )
        loss = outputs.loss
        
        print(f"Processing batch {batch_idx}")

        self.log("val_loss", loss)
        self.val_accuracy(outputs.logits, labels)
        self.log("val_acc", self.val_accuracy, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.00001)




def predict_document_class(im_path , model , reader ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    image=im_path.convert('RGB')
    width , height=image.size
    width_scale=1000/width
    height_scale=1000/height
    np_img = np.array(image)
    
    ocr_data=[]
    ocr_result=reader.readtext(im_path)
    for bbox , word , confidence in ocr_result:
        ocr_data.append({'word':word ,
                         'bbox':create_bbox(bbox)})

    
    words=[]
    boxes=[]
    for row in ocr_data:
        words.append(row['word'])
        boxes.append(scale_bounding_box(row['bbox'],width_scale,height_scale))

    encoding=processor(
    np_img,
    words,
    boxes=boxes,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt')

    with torch.inference_mode():
        output=model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device),
            bbox=encoding['bbox'].to(device),
            pixel_values=encoding['pixel_values'].to(device)
        )

    predict_class=output.logits.argmax()
    return classes[predict_class.item()]

reader = easyocr.Reader(['en'])  
model=ModelModule(len(classes))
model.load_state_dict(torch.load("D:\\real_world_projects\\Document_classifi\\best_model2.pth",map_location=torch.device('cpu')))
st.title("Document Classification 📑")
uploadpath=st.file_uploader("Upload Document Image",['jpg','png'])
if uploadpath is not None:
    bytedata = io.BytesIO(uploadpath.getvalue())
    file = Image.open(bytedata)
    st.image(file,"Your Document")
    with st.spinner("processing...."):
        predict_class = predict_document_class(file , model , reader)
    st.markdown(f"Predicted Document Type : **{predict_class}**")