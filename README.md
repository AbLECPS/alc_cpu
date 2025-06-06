# ALC Toolchain

Source distribution for the Assurance-based Learning-enabled Cyber-Physical Systems (ALC) Toolchain. For more information on the project please see https://assured-autonomy.isis.vanderbilt.edu/ and http://ablecps.github.io/. This repository contains the sources for the toolchain, as well as examples using models of BlueROV, an Unmanned Underwater Vehicle.

Contact: alc-team@vanderbilt.edu

--------------------------------------------------------------------------------------------------------------------------------------

Setup for CPU 
==============

The release was succesfully tested on machines running Ubuntu 24.04, and 22.04 . 



Prerequisites
-------------


- Linux OS (tested with Ubuntu 22.04, Ubuntu 24.04) or on Windows with [WSL](https://documentation.ubuntu.com/wsl/latest/howto/install-ubuntu-wsl2/) and  Ubuntu 24.04.


- [Docker ](https://docs.docker.com/desktop/setup/install/linux/ubuntu/)

    - Tested with docker version 20.10.(7,8,11)
    - Configure your user account to [use docker without 'sudo'](https://github.com/sindresorhus/guides/blob/master/docker-without-sudo.md)
    - Be sure to log out, then log back in so group changes will be applied

- [Docker Compose File Format](https://www.cherryservers.com/blog/install-docker-compose-ubuntu)

    - Version should be >= 2.3 and < 3.0



- Other tools required as part of the setup (these come standard with Ubuntu 24.04 install on WSL)
        
    - Install Git

        ```
        sudo apt-get update
        sudo apt-get install git
        
        ```

    - Install [Git LFS](https://packagecloud.io/github/git-lfs/install)

    - Install openssl  (Version 1.1.1. or above). Please refer to this [link](https://cloudwafer.com/blog/installing-openssl-on-ubuntu-16-04-18-04/) to update the openssl version to 1.1.1.

    - Install openssh-client

        ```
        sudo apt-get install openssh-client
        ````

Installation
==============

Setup environment
-----------------


1.) Edit ~/.bashrc and add the following Environment Variables:

- ALC_HOME - Directory where ALC repository is/will be installed
- ALC_WORKING_DIR - Workspace directory used for storing job results (simulation, training, etc.)
- ALC_DOCKERFILES - Runtime directory for docker containers
        (contains WebGME MongoDB database, configuration files, etc.)


These directories can be anywhere on the system, but the user must have write permissions within each directory.
    If any permission issues are encountered, change the directory owner with the following command 
    
    `sudo chown $USER:$USER <desired_dir>`



Example bash_rc addition

```
        # ALC Variables
        export ALC_HOME=$HOME/alc
        export ALC_WORKING_DIR=$HOME/alc_workspace
        export ALC_DOCKERFILES=$HOME/alc_dockerfiles
```


2.) Source updated bashrc and create directories


```
        source ~/.bashrc
        sudo mkdir $ALC_WORKING_DIR
        sudo chown $USER:$USER $ALC_WORKING_DIR
```



Get ALC docker images, codebase and data.
----------------------------------------

    
1.) Clone this repository - branch cpu

```   
       git clone git@github.com:AbLECPS/alc_cpu.git $ALC_HOME

 ```


2.) Pull the docker images (about ~20 GB) (Please read the notes below.)

```    
$ALC_HOME/docker/alc/pull_images_from_server.sh
```
        
The docker images and containers are stored in /var/lib/docker. If you don't have space on the disk where this folder exists, please follow the [instructions](https://r00t4bl3.com/post/how-to-move-docker-data-directory-to-another-location-on-ubuntu) to configure the docker daemon to use another directory on a drive with sufficient space.


3.) Copy data and pre-trained models

```
    $ALC_HOME/docker/alc/pull_data.sh
```
    
This step will copy several megabytes of data and trained models to $ALC_WORKING_DIR folder. The BlueROV models depends on the data downloaded and copied on to the `$ALC_WORKING_DIR`.
It also includes bluerov2 description file for running BlueROV simulation in Gazebo.

    
Setup the ALC toolchain
-----------------------

1.)
This step sets up the ALC toolchain folders, sshkeys and certificates.

```
    cd $ALC_HOME/docker/alc
    ./setup.sh

```

2.)
The ALC toolset runs a local GIT server the following commands initializes the git repositories associated in the server and builds  them for future execution.

```
   $ALC_HOME/docker/alc/setup_and_build_repos.sh ALL
```
    
    
Perform Slurm workload manager configuration
--------------------------------------------


Using your preferred text editor, edit `slurm.conf` and `gres.conf` files in the  directory `$ALC_DOCKERFILES/slurm/etc` to match
    your machine's hardware (CPU count, available Memory, & GPU information). Note that the default compute node
    configuration info is located near the bottom of the `slurm.conf` file, and looks as follows:
         
```
        # COMPUTE NODES
        NodeName=alc_slurm_node0 CPUs=16 RealMemory=30000 Sockets=1 CoresPerSocket=8 ThreadsPerCore=2 State=UNKNOWN 
```
Slurm configuration requires that provided values be strictly less than or equal to actual available hardware.
    For example, a computer with 16 GB of RAM may actually report 15,999 MB available.
    In this case, setting 'RealMemory' to 16,000 will cause an error. Must be set <= 15,999.

The `gres.conf` file may also need to be edited depending on the available GPU(s). 
The information in this file must match what is specified in the "Gres" field of the `slurm.conf` configuration, and looks as follows:
    
```
    NodeName=alc_slurm_node0 

```

This [listing](https://nouveau.freedesktop.org/wiki/CodeNames/) provides the codes for different GPU architectures.
See [Slurm documentation](https://slurm.schedmd.com/) on these two files if more information is needed.


   

Start ALC toolchain
===================

```
    cd $ALC_HOME/docker/alc
    ./run_services.sh
```

This script launches the ALC toolchain services using docker-compose. It does not detach from docker-compose by default, and all logs will be displayed in this terminal.

To run services in the background, use 

```
    ./run_services.sh -d
```



Using the ALC Toolchain
=======================

- Follow the instructions in  [**First Login**](doc/sphinx/source/getting_started/_first_login_.rst) to access the toolchain.  
- Follow the instructions in [**ALC Initial Steps**](doc/sphinx/source/getting_started/_alc_initial_steps_.rst) to ensure the toolchain was installed correctly.
- Follow the instructions in [**Working With IDE**](doc/sphinx/source/getting_started/_working_with_ide_.rst) to use the IDE integrated with WebGME for the toolchain.
- The pdf documentation of ALC toolchain guide is available [here](doc/ALC_Toolchain_Documentation.pdf).


Additional Notes
---------------

- If you are accessing the Toolchain remotely, then replace ``localhost`` with the address of the server hosting the ALC services.
- The certificates are self-generated. So browsers prompt warning about the certificates. Please ignore the warnings and proceed to the site.
 


Acknowledgement
===============

This work was supported by the DARPA Assured Autonomy project and Air Force Research Laboratory. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of DARPA or AFRL.

