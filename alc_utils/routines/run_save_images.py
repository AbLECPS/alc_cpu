import subprocess
import os
alc_home_path = os.getenv('ALC_HOME')
def run(folder):
    # Call the script with arguments
    topics = [ "/vu_sss/waterfall_gt_l","/vu_sss/waterfall_gt_r","/vu_sss/waterfall_l","/vu_sss/waterfall_r"]
    output_folder = ["./results/images/gt","./results/images/gt","./results/images/scan", "./results/images/scan"]
    file_prefixs = ["l_","r_","l_", "r"]
    script_path = alc_home_path + "/alc_utils/routines/save_images.sh"
    for i in range(0,4):
        print(i)
        subprocess.call([script_path, folder, os.path.join(folder,output_folder[i]), file_prefixs[i], topics[i]])

