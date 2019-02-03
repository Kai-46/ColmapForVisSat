#!/usr/bin/env bash

# generate eclipse project
work_dir=../eclipse_colmap_debug
if [ ! -d $work_dir ]; then
	mkdir $work_dir
fi

cd $work_dir
cmake -G "Eclipse CDT4 - Unix Makefiles" \
	-DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE \
	-DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j8 \
	-DCMAKE_ECLIPSE_VERSION=4.10 \
	../colmap_for_satellite_stereo