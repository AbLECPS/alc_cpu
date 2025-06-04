#!/usr/bin/env python
# This contains python code for executing ALC jobs
# Based on the jobtype, it invokes the appropriate functions in alc_utils.
from __future__ import print_function

import os
import json
import sys
import tempfile
from pathlib import Path
import os
import re
import itertools
from alc_utils.LaunchActivity.ActivityInterpreterBase import ActivityInterpreterBase
from alc_utils.LaunchActivity.KeysAndAttributes import Keys, Attributes
from alc_utils.config import WORKING_DIRECTORY, JUPYTER_WORK_DIR
import traceback



class EvaluationRunner:

    alc_working_dir_env = "ALC_WORKING_DIR"
    jupyter_name        = "jupyter"
    notebook_dir        = "result"

    input_lec_key_name       = "LECModels"
    input_training_data_key_name   = "TrainingData"
    input_test_data_key_name       = "TestData"
    input_validation_data_key_name = "ValidationData"
    metadata_directory_key_name    = "directory"

    eval_context_key_name    = "Evaluation"
    eval_definition_key_name = "eval_definition"
    eval_parameter_key_name     = "Parameters"
             
    eval_support_code_filename     = "eval_code.py"
    eval_script_filename      = "eval_script.sh"
    eval_folder_name          = "Evaluation"
    eval_definition_filename  = "Evaluation.py"
    eval_support_code_param_key = "eval_code"
    eval_script_param_key = "eval_script"

    result_filename              = "result.json"   

    def __init__(self):
        self.input_path = '.'
        self.temp_dir = '.'

        self.inputs = {}

        self.lecs = None
        self.lecs_metadata = None
        
        self.training_data = []
        self.test_data     = []
        self.validation_data = []

        self.training_metadata = []
        self.test_metadata     = []
        self.validation_metadata = []

        self.eval_parameters = {}

        
        self.eval_definition = None
        self.eval_definition_path = None
        self.eval_definition_folder = None
        self.eval_support_code_path = None
        self.eval_script_path = None


        self.alc_working_dir = None
        self.mesgs = []
    
    def update_dir(self, dir):
        folder_path = dir
        pos = folder_path.find(self.jupyter_name)
        if (pos > -1):
            folder_path = folder_path[pos:]
        return os.path.join(self.alc_working_dir,folder_path)

    def get_input_data_dirs(self, key):
        dirs = []
        metadata = []

        result = self.inputs.get(key)
        if (not result):
            self.mesgs.append(" no input of {0} found ".format(key))
            return dirs, metadata
        
        metadata = result.get(Keys.input_set_name)
        if (not metadata):
            self.mesgs.append(" empty input set for {0} found ".format(key))
            return dirs, metadata

        
        ret_metadata = []
        for md in metadata:
            dir = md.get(self.metadata_directory_key_name)
            if (not dir):
                self.mesgs.append("Directory entry not found in metadata")
                continue
            updated_dir = self.update_dir(dir)
            dir_path = Path(updated_dir).absolute()
            if (not dir_path.exists()):
                self.mesgs.append('directory path {0} not found in input {1}'.format(updated_dir,key))
                continue
            md[self.metadata_directory_key_name] = str(dir_path)
            ret_metadata.append(md)
            dirs.append(str(dir_path))

        return dirs, metadata
    
    def get_eval_definition(self):
        eval_definition = ''
        parameters = self.input_map.get(Keys.parameters_key_name)
        if (parameters):
            eval_params = parameters.get(self.eval_context_key_name)
            if eval_params:
                eval_definition = eval_params.get(self.eval_definition_key_name)
        return eval_definition

    
    def get_eval_support_code_script(self):
        
        eval_support_code = ''
        eval_script = ''
        
        if (self.eval_parameters):
            eval_support_code = self.eval_parameters.get(self.eval_support_code_param_key)
            eval_script = self.eval_parameters.get(self.eval_script_param_key)
                
        return eval_support_code, eval_script

    

    def get_parameters(self):
        parameters = self.input_map.get(Keys.parameters_key_name)
        if parameters:
            eval_params = parameters.get(self.eval_parameter_key_name)
        return eval_params

    def create_folder(self, folder_path):
        x = Path(folder_path)
        if (x.exists()):
            return
        x.mkdir()
    
    def write_to_file(self, file_path,contents):
        f = open(file_path,'w')
        f.write(contents)
        f.close()

        
    def create_files(self):
        self.create_folder(self.eval_definition_folder)
        self.write_to_file(self.eval_definition_path, self.eval_definition)
        if (self.eval_support_code):
            self.write_to_file(self.eval_support_code_path, self.eval_support_code)
        if (self.eval_script):
            self.write_to_file(self.eval_script_path, self.eval_script)
    
    # def createTrainingOutputDirectory(self):
    #     import time
    #     x = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
    #     folder_path = os.path.join(self.input_path,"TrainingResult_{0}".format(x))
    #     self.create_folder(folder_path)
    #     return folder_path



    def setup(self):

        import argparse
        import json
        import os

        
        

        json_file = os.path.join(self.input_path, 'launch_activity_output.json')
        with Path(json_file).open("r") as json_fp:
            input_json_map = json.load(json_fp)


        self.alc_working_dir = os.getenv(self.alc_working_dir_env)
        if not self.alc_working_dir:
            raise Exception("environment variable {0} not found".format(self.alc_working_dir_env))
        
        
        self.input_map = input_json_map
        self.inputs = self.input_map[Keys.inputs_key_name]


        self.training_data, self.training_metadata     = self.get_input_data_dirs(self.input_training_data_key_name)
        self.test_data, self.test_metadata             = self.get_input_data_dirs(self.input_test_data_key_name)
        self.validation_data, self.validation_metadata = self.get_input_data_dirs(self.input_validation_data_key_name)
        self.lecs, self.lecs_metadata                  = self.get_input_data_dirs(self.input_lec_key_name)

        self.eval_parameters                            = self.get_parameters()
        self.eval_support_code, self.eval_script        = self.get_eval_support_code_script()
        self.eval_definition                            = self.get_eval_definition()

        self.eval_definition_folder                     = os.path.join(self.input_path, self.eval_folder_name)
        self.eval_definition_path                       = os.path.join(self.eval_definition_folder, self.eval_definition_filename)
        self.eval_support_code_path                     = os.path.join(self.eval_definition_folder, self.eval_support_code_filename)
        self.eval_script_path                           = os.path.join(self.eval_definition_folder, self.eval_script_filename)
        self.create_files()



    def execute(self):
        # execute based on job type
        ret = {}
        from alc_utils.common import load_python_module
        from alc_utils.routines.setup import createResultNotebook2

        param_dict = self.eval_parameters
        

        if (not self.validation_metadata or len(self.validation_metadata) == 0):
            self.validation_metadata = None

        if (not self.test_metadata or len(self.test_metadata) == 0):
            self.test_metadata = None

        if (not self.training_metadata or len(self.training_metadata) == 0):
            self.training_metadata = None

        
        if (not self.lecs or len(self.lecs_metadata) == 0):
            self.lecs_metadata = None


        #output_dir = self.createTrainingOutputDirectory()
        output_dir = self.input_path

        
        

        eval_module = None
        eval_result = None
        if os.path.exists(self.eval_definition_path):
            eval_module = load_python_module(self.eval_definition_path)
            eval_result = eval_module.run(self.training_data, self.test_data, self.validation_data, self.lecs, **param_dict)
                
        ret= eval_result
        ret['evalResultURL'] = eval_result.get('result_url','')
        createResultNotebook2(output_dir)
        full_output_dir = os.path.abspath(output_dir)
        relative_folder_path = full_output_dir[ len(JUPYTER_WORK_DIR)+1:]
        #ret['directory'] = full_output_dir
        ret['result_url'] = os.path.join('ipython','notebooks',relative_folder_path,'result.ipynb')
        if (not (ret.get('evalParams'))):
            ret['evalParams']= self.eval_parameters
        return ret



if __name__ == '__main__':
    eval = EvaluationRunner()
    eval.setup()
    result_output = eval.execute()
    
    try:
        with Path(eval.input_path, eval.result_filename).open("w",encoding="utf-8") as json_fp:
            outval = json.dumps(result_output, indent=4, sort_keys=True,ensure_ascii=False)
            json_fp.write(outval)
        
    except Exception as e:
        traceback.print_exc()
        print ('run invoked exception')
        print(e)