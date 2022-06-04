#!/usr/bin/env bash

# Check args
if [ "$#" -ne 1 ]; then
	  echo "usage: ./run.sh IMAGE_NAME"
	    return 1
    fi

	# Get this script's path
    pushd `dirname $0` > /dev/null
    SCRIPTPATH=`pwd`
    popd > /dev/null
    
    set -e

	# Should you run into a "docker: invalid reference format."-error, please check for whitespaces after \ . This can cause it. 
	# see https://codereviewvideos.com/blog/how-i-fixed-docker-invalid-reference-format-with-makefile/ 

	# Run the container with shared X11
	docker run\
	    --shm-size 12G\
	    --gpus all\
		--publish-all=true\
		--net=host\
		    -e SHELL\
			--mount type=bind,source="/home/nfs/inf6/data/datasets/SemanticKitti",target=/workspace/Data/SemanticKitti \
			--mount type=bind,source="/home/user/schuett/icra_temp_latticenet",target=/workspace \
				-e DISPLAY\
				-e DOCKER=1\
				-e WORKSPACE="/workspace/" \
					-it $1

			
