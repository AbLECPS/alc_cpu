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


#am_test_path = '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2Lite_AM/vae_sim_hw_10_recalib' #'/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2Lite_AM/vae/'
am_test_path= '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2Lite_AM/vaes/vae_trained' # has issues with hw, zigzag, diameter_change and some points in nominal
am_test_path= '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2Lite_AM/vaes/vae_sim_hw_10'
am_test_path= '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2Lite_AM/vaes/vae_sim_hw_10_prior_to_recalib'
am_test_path= '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2Lite_AM/vaes/vae_sim_hw_10_recalib'
#am_test_path= '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2Lite_AM/vae/'
data_test_path = ['/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/hw/tube1',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/hw/tube2',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/hw/saltnpepper',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/hw/battery',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/nominal_1',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/nominal_2',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/nominal_3',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/0.75zigzag',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/1.25zigzag',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/1.5zigzag',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/diameter_change/sim_0.5x',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/diameter_change/sim_1x',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/diameter_change/sim_5.5x'
                  ]

data_test_path_id = ['/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/hw/tube1',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/hw/tube2',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/nominal_1',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/nominal_2',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/nominal_3',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/diameter_change/sim_1x',
                  ]

data_test_path_ood = ['/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/hw/saltnpepper',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/hw/battery',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/0.75zigzag',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/1.25zigzag',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/1.50zigzag',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/diameter_change/sim_0.5x',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/diameter_change/sim_5.5x'
                  ]

parent_data_obs = '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/obsdata/'
data_test_obs = ['1obs','1obs_avoided_far_side','1obs_avoided_far_side2','1obsnoobszigzag','1obs_zig_zag_avoidance_bad_lec_pred','1obs_zig_zag_bad_lec_pred']


data_test_path_dia_change = ['/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/diameter_change/sim_0.6x',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/diameter_change/sim_0.8x',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/diameter_change/sim_1.75x',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/diameter_change/sim_2.5x',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/diameter_change/sim_4.5x',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/diameter_change/sim_6.5x',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/diameter_change/sim_10x',
                  '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2_data/data/scenarios/sim/diameter_change/sim_14x',
                  ]

print('**********************************')
am_test_path = '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2Lite_AM/vaes/vae_sim_hw_10_recalib2'
am_test_path = '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2Lite_AM/vaes/vae_sim_hw_10_recalib3'
am_test_path = '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2Lite_AM/vaes/vae_sim_hw_10_recalib4'
am_test_path= '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2Lite_AM/vaes/vae_sim_hw_10_orig'
am_test_path= '/home/alcuser/alc_workspace/jupyter/admin_BlueROV/LEC2Lite_AM/vaes/vae_sim_hw_10_orig1'
print('am_path: ', am_test_path)

# for data_test_path in data_test_path_id:
#      print('=========================================')
#      print('ID data_path: ', data_test_path)
#      test_am(data_test_path, am_test_path)
#      print('=========================================')

# for data_test_path in data_test_path_ood:
#     print('=========================================')
#     print('OOD data_path: ', data_test_path)
#     test_am(data_test_path, am_test_path)
#     print('=========================================')

# for data_test_path in data_test_path_dia_change:
#     print('=========================================')
#     print('OOD data_path: ', data_test_path)
#     test_am(data_test_path, am_test_path)
#     print('=========================================')

for data in data_test_obs:
     data_test_path = parent_data_obs + data
     print('=========================================')
     print('OOD data_path: ', data_test_path)
     test_am(data_test_path, am_test_path)
     print('=========================================')
     break
     


