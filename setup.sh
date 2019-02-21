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

if [ ! -d "mmdetection" ]; then
	# install mmdetection and mmcv
	git clone https://github.com/open-mmlab/mmdetection
	cd mmdetection && \
	pip3 install cython && \
	./compile.sh && \
	python3 setup.py install

	# download pretrain model
	mkdir models && \
	cd models && \
	wget -nc "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_fpn_1x_20181010-4a9c0712.pth" && \
	cd ../..
fi

if [ ! -d "demo" ]; then
	# download demo video
	mkdir demo && \
	download_from_gdrive "1av2UGYErDb6FxE867zbvhc6B2ZpMZOGz" "demo/pedestrian-1.mp4"
fi
