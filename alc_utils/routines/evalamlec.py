import sys
import os
import json
import yaml
import csv
import alc_utils.routines.run_test_lec2_ as rt_gpu
import numpy as np


def evaluate_lec(lec_folder, test_data, weights_file='model_weights.h5'):
    print("Evaluation Began")
    lec_weights_filename = os.path.join(lec_folder, weights_file)
    lec_size = os.stat(lec_weights_filename).st_size
    ml_metrics = {'model_size': lec_size}
    scenario_results = []
    for scene_folder in test_data:
        scene_name           = os.path.basename(scene_folder)
        print(scene_name)
        output_filename      = os.path.join("eval1", scene_name + ".pkl")
        all_results, results, red_count_results, nominal_index, high_index , compute_time, am_results = rt_gpu.run(lec_weights_filename, scene_folder,output_filename)
        scenario_results.append({'scene': scene_name,
                                'avg_prediction_time_(s)': compute_time,
                                'lec_results': all_results,
                                'snapshot_results': results,
                                'gt_ood': red_count_results,
                                'am_nominal_index': nominal_index,
                                'am_high_index': high_index,
                                'am_results':am_results,
                                'avg_prediction_time_(s)': compute_time})
    return scenario_results
  
  
       	
  
def run(lecs, test_data):
    print('eval all started')
    current_folder = os.getcwd()
    scenario_results = evaluate_lec(lecs, test_data)
    
    with open("eval1/output.json", "w") as f:
        json.dump(scenario_results, f)

