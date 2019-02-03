#!/usr/bin/env bash

# compile colmap
build_dir=./build
if [ ! -d $build_dir ]; then
	mkdir $build_dir
fi

cd $build_dir
CC=/usr/bin/gcc-6 CXX=/usr/bin/g++-6 cmake ..
make -j8
sudo make install