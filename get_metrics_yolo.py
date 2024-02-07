import os
import time #FOR DEBBUG
from ultralytics import YOLO
import json

#BEGIN OF SETTINGS
load_models = []
output = 'predictions'

load_models.append(
    {
        'name': 'RGB',
        'weight': 'RESULTS TRAIN YOLO MODALITIES/RGB/train/weights/last.pt',
        'img_dir': '/home/kmota/inhouse/inhouse/DEI/rgb/images',
        'gt_file': '/home/kmota/inhouse/inhouse-newCOCO/validation/annotations/annotations.json'
    }
)

#END OF SETTINGS
########################################
def extract_results(results):
    data = []
    for i in results:
        for j in i:
            data.append({
                'file': j.path,
                'shape': j.orig_shape,
                'names': j.names,
                'cls': int(j.boxes.cls.tolist()[0]),
                'conf': j.boxes.conf.tolist()[0],
                'xywh': j.boxes.xywh.tolist()[0],
                'xywhn': j.boxes.xywhn.tolist()[0],
                'xyxy': j.boxes.xyxy.tolist()[0]
            })
    return data

for i in load_models:
    #CREATE FOLDER
    try:
        os.makedirs(os.path.join(output, i['name'],'YOLO'))
    except:
        pass

    #LOAD MODEL
    model = YOLO(i['weight'])
    
    #LOAD IMAGES ID FROM GROUND TRUTH
    GT_raw = open(i['gt_file'])
    GT = json.load(GT_raw)
    GT = GT['images']
    images_db = {}
    for j in GT:
        images_db.setdefault(j['file_name'],j['id'])

    #INFERENCE IMAGES WITH LOADED MODEL
    results = model(os.path.join(i['img_dir']),device='cpu') #REMOVE DEVICE OPTION
    data = extract_results(results)

    #SAVE INFERENCES TO YOLO MODEL
    yolo_names = []
    for j in data:
        if j['names'][0] not in yolo_names:
            yolo_names.append(j['names'][0])
            
        filename = os.path.basename(j['file'])
        with open(os.path.join(output,i['name'],'YOLO',filename.split('.')[0]+'.txt'),'a') as f:
            f.write(str(j['cls'])+' '+str(j['conf'])+' '+str(j['xywhn'][0])+' '+str(j['xywhn'][1])+' '+str(j['xywhn'][2])+' '+str(j['xywhn'][3])+'\n')
        f.close()

    #SAVE CLASSES ON FILE
    with open(os.path.join(output,i['name'],'obj.names'),'a') as f:
        for j in yolo_names:
            f.write(j+'\n')

    #SAVE INFERENCES TO COCO MODEL
    annotations = []
    for j in data:
        annotations.append({
            "file": os.path.basename(j['file']),
            "image_id": images_db[os.path.basename(j['file'])], #HOW TO SOLVE THIS SHIT
            "category_id": j['cls']+1,
            "bbox": j['xywh'],
            "score": j['conf'],
        })

    with open(os.path.join(output,i['name'],'coco_annotations.json'),'w') as f:
        json.dump(annotations,f,indent=4)