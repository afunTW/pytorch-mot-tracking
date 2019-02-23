#!/bin/bash

download_from_gdrive() {
    file_id=$1
		file_name=$2
				    
		# first stage to get the warning html
		curl -c /tmp/cookies \
		"https://drive.google.com/uc?export=download&id=$file_id" > \
		/tmp/intermezzo.html
		
		# second stage to extract the download link from html above
		download_link=$(cat /tmp/intermezzo.html | \
		grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | \
		sed 's/\&amp;/\&/g')
		curl -L -b /tmp/cookies \
		"https://drive.google.com$download_link" > $file_name
}

if [ ! -d "darknet" ]; then
	# install darknet and compiling with CUDA
	git clone https://github.com/pjreddie/darknet.git
	cd darknet
	make GPU=1
	wget https://pjreddie.com/media/files/yolov3.weights -P cfg/
	cd ..
fi

if [ ! -d "demo" ]; then
	# download demo video
	mkdir demo && \
	download_from_gdrive "1av2UGYErDb6FxE867zbvhc6B2ZpMZOGz" "demo/pedestrian-1.mp4"
fi
