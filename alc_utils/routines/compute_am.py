import os
import sys
import alc_utils.assurance_monitor
from imutils import paths
import cv2
import torch
from torchvision import transforms
from PIL import Image
from torchvision import transforms



alc_working_dir_env_var_name = "ALC_WORKING_DIR"
alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)

transform = transforms.Compose([
            transforms.ToTensor()
        ])
def fix_folder_path(folder_path):
    if (not folder_path):
        return None
    pos = folder_path.find('jupyter')
    if (pos == -1):
        return folder_path
    folder_path = folder_path[pos:]
    if (alc_working_dir_name):
        ret = os.path.join(alc_working_dir_name, folder_path)
        return ret
    return None


def run(am_path, input_data, threshold=-2, param_dict = {}):
    results=[]
    high_logM_index = []
    nominal_logM_index = []

    # to initialize the am
    _detectors = alc_utils.assurance_monitor.load_assurance_monitor("multi")
    _detectors.load([am_path])
    _detector = _detectors.ood_detectors[0]
    
    # running through the data set
    for i in range(len(input_data)):
        #x = torch.from_numpy(input_data[i])
        x = input_data[i]
        x1 =x.reshape(1,1,16,180)
        result1=_detector.evaluate(x1, None, **param_dict)
        
        if (result1[0][-2]>threshold):
            print (str(i) + "   =========  " + str(result1[0][-2]))
            high_logM_index.append(i)
        else:
            #print(result1[0][-2])
            nominal_logM_index.append(i)
        results.append(result1)
    del _detector
    return results, nominal_logM_index, high_logM_index 
        



def test_am(data_path, am_path):
    am_path = fix_folder_path(am_path)
    data_path = fix_folder_path(data_path)

    imagesp = []
    content = list(paths.list_images(os.path.join(data_path,"scan")))
    for imagePath in sorted(content):
        imagesp.append(imagePath)

    images = []
    for imagePath in imagesp:
        # image = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
        # image = transforms.functional.to_tensor(image)
        # image = torch.div(image, 256.0)
        #images.append(image)

        image = Image.open(imagePath)
        #print(filepath)
        image = transforms.Grayscale()(image)
        image = transform(image)
        image = torch.div(image, 256.0)
        images.append(image)

    am_result, nominal_index, high_index = run(am_path,images)
    print('done')


     


