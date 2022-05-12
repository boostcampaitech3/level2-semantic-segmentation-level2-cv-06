import gc
import os
import os.path as osp

import pandas as pd
import numpy as np
import albumentations as A

import torch
from tqdm import tqdm

def test(args, model, data_loader, device):
    save_dir = args.save_dir
    gc.collect()
    torch.cuda.empty_cache()

    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model = model.to(device)
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(data_loader)):
            
            outs = model(torch.stack(imgs).to(device))            
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    submission = pd.read_csv('/opt/ml/input/code/submission/sample_submission.csv', index_col=None)
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append(
            {"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())
            }, ignore_index=True)

    submission.to_csv(osp.join(save_dir, f"{args.name}.csv"), index=False)