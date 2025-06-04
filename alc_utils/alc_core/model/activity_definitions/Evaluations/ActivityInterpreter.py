#!/usr/bin/env python
# This contains python code for executing ALC jobs
# Based on the jobtype, it invokes the appropriate functions in alc_utils.
from __future__ import print_function

import json
from pathlib import Path
import os
import sys
from alc_utils.LaunchActivity.ActivityInterpreterBase import ActivityInterpreterBase
from alc_utils.LaunchActivity.KeysAndAttributes import Keys, Attributes


class ActivityInterpreter(ActivityInterpreterBase):

    alc_working_dir_env             = "ALC_WORKING_DIR"
    jupyter_name                    = "jupyter"
    notebook_dir                    = "result"
    result_json                     = "result.json"
    deployment_base_dir_key         = 'base_dir'
    activity_home_dir_key           = 'activity_home'

    input_lec_key_name              = "LECModels"
    input_training_data_key_name    = "TrainingData"
    input_test_data_key_name        = "TestData"
    input_validation_data_key_name = "ValidationData"
    metadata_directory_key_name     = "directory"

    eval_context_key_name           = "Evaluation"
    eval_definition_key_name        = "eval_definition"
    eval_parameter_key_name         = "Parameters"
    execution_param_key             = "Execution"
    timeout_param_key               = "timeout"
    containers_param_key            = "containers"
    options_param_key               ="options"
    volumes_param_key               = "volumes"
    load_dev_param_key              = "load_dev"
    privileged_param_key            = "privileged"

    def __init__(self, folder_path):
        ActivityInterpreterBase.__init__(self, folder_path)

        self.temp_dir = folder_path

        self.inputs = self.input_map[Keys.inputs_key_name]
        

        self.lecs = []
        self.lecs_metadata = []
        
        self.training_data = []
        self.test_data     = []
        self.validation_data = []

        self.training_metadata = []
        self.test_metadata     = []
        self.validation_metadata = []

        self.eval_definition = None
        self.eval_definition_filename = None

        self.alc_working_dir = None
        self.mesgs = []

        self.parameters = self.get_parameters()
    
    def update_dir(self, local_dir):
        folder_path = local_dir
        folder_path_parts_tuple = folder_path.parts
        if self.jupyter_name in folder_path_parts_tuple:
            pos = folder_path_parts_tuple.index(self.jupyter_name)
            folder_path = Path(*folder_path_parts_tuple[pos:])
        return Path(self.alc_working_dir, folder_path)

    def get_input_data_dirs(self, key):
        dirs = []
        metadata = []

        result = self.inputs.get(key)
        if not result:
            self.mesgs.append(" no input of {0} found ".format(key))
            return dirs, metadata
        
        metadata = result.get(Keys.input_set_name)
        if not metadata:
            self.mesgs.append(" empty input set for {0} found ".format(key))
            return dirs, metadata

        ret_metadata = []
        for md in metadata:
            local_dir_string = md.get(self.metadata_directory_key_name)
            if not local_dir_string:
                self.mesgs.append("Directory entry not found in metadata")
                continue
            local_dir = Path(local_dir_string)
            updated_dir = self.update_dir(local_dir)
            dir_path = Path(updated_dir).absolute()
            if not dir_path.exists():
                self.mesgs.append('directory path {0} not found in input {1}'.format(updated_dir, key))
                continue
            md[self.metadata_directory_key_name] = str(dir_path)
            ret_metadata.append(md)
            dirs.append(dir_path)

        return dirs, metadata
    
    
    def get_eval_definition(self):
        eval_definition = ''
        if (self.parameters):
            self.eval_params = self.parameters.get(self.eval_context_key_name)
            if self.eval_params:
                eval_definition = self.eval_params.get(self.eval_definition_key_name)
        return eval_definition

    def get_parameters(self):
        self.parameters = self.input_map.get(Keys.parameters_key_name)
        if self.parameters:
            self.eval_parameter = self.parameters.get(self.eval_parameter_key_name)
        return self.parameters
        

    def setup(self):

        if self.alc_working_dir_env not in os.environ:
            raise Exception("environment variable {0} not found".format(self.alc_working_dir_env))
        self.alc_working_dir = Path(os.getenv(self.alc_working_dir_env))

        self.training_data, self.training_metadata      = self.get_input_data_dirs(self.input_training_data_key_name)
        self.test_data, self.test_metadata              = self.get_input_data_dirs(self.input_test_data_key_name)
        self.validation_data, self.validation_metadata  = self.get_input_data_dirs(self.input_validation_data_key_name)
        self.lecs, self.lecs_metadata                   = self.get_input_data_dirs(self.input_lec_key_name)

        self.eval_definition = self.get_eval_definition()

      
    def create_deployment_file(self):
        activity_folder = str(Path(__file__).absolute().parent)
        sys.path.append(activity_folder)
        import Dep
        dep_dict = Dep.dep_dict
        dep_dict[self.deployment_base_dir_key] = str(self.input_dir_path)
        dep_dict[self.activity_home_dir_key] = str(activity_folder)
        execution_params = self.parameters.get(self.execution_param_key, None)
        if execution_params:
            timeout = execution_params.get(self.timeout_param_key, None)
            if timeout:
                dep_dict[self.timeout_param_key] = timeout
            mount_dev = execution_params.get(self.load_dev_param_key, False)
            if mount_dev:
                if (not dep_dict[self.containers_param_key][0].get(self.options_param_key,None)):
                    dep_dict[self.containers_param_key][0][self.options_param_key] = {}
                if (not dep_dict[self.containers_param_key][0][self.options_param_key].get(self.volumes_param_key,None)):
                    dep_dict[self.containers_param_key][0][self.options_param_key][self.volumes_param_key] = {}
                dep_dict[self.containers_param_key][0][self.options_param_key][self.volumes_param_key]["/dev"]= {"bind": "/dev","mode": "rw"}
                dep_dict[self.containers_param_key][0][self.options_param_key][self.privileged_param_key]=True
        
        
        file_path = Path(str(self.input_dir_path),'config.json')
        with file_path.open('w') as fp:
            json.dump(dep_dict, fp, indent=4, sort_keys=True)
        
        return file_path     
      
    def execute(self):
        from alc_utils import execution_runner
        exec_config_file_path = self.create_deployment_file()
        runner = execution_runner.ExecutionRunner(str(exec_config_file_path))
        
        result, resultdir = runner.run()

        ret = {}
        
        if result == 0:
            with Path(self.result_json).open("r") as json_fp:
                ret = json.load(json_fp)


        return ret

    def check(self):
        mesgs = []
                
        if mesgs:
            mesgs.extend(self.mesgs)

        return mesgs

    # method invoked to run the jobs
    def run(self):
        self.setup()
        mesgs = self.check()
        print('check messages {0}'.format(mesgs))
        if mesgs:
            ret = {
                "errors": mesgs
            }
            return ret
        
        return self.execute()

