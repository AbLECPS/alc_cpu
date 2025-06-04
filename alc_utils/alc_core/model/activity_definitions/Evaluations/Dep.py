# This contains deployment dictionary  that is passed to the execution_runner
#

dep_dict = { "name": "EvalJob",
  "ros_master_image": None,
  "base_dir": "ver_example_job/example_timestamp",
  "results_dir": ".",
  "timeout": 7200,
  "containers": [
    { "name": "EvalJob",
      "image": "alc_37:latest",
      "command": "${ACTIVITY_HOME}/runner.sh",
      "input_file": "launch_activity_output.json",
      "options": {
          "hostname": "EvalJob",
          "runtime": "nvidia"
          }
      }
  ]
}
