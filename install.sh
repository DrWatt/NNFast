#!/bin/bash

if [ -e /usr/bin/python ]; then
	
	currentver="$(python -V)"
	requiredver="3.0.0"
		if [ "$(echo '%s\n' "$requiredver" "$currentver"| sort -V | head -n1 )" = "$requiredver" ]; then 
	        python="python"
	        echo "Python = python3"
		else
			if [ -e /usr/bin/python3 ]; then
	        	python="python3"
	        	echo "Python = python2"
	        else
	        	sudo apt update
	        	sudo apt install python3
	        	python="python3"
	        fi
		fi

	else

	sudo apt update
	sudo apt install python3
	python="python3"
fi

$python -m pip --version > /dev/null 2>&1
if [[ !($? -eq 0) ]]; then
	if [[ !(-e get-pip.py) ]]; then
	wget https://bootstrap.pypa.io/get-pip.py
	fi
	sudo apt update
	sudo apt install python3-distutils python3-apt -y

	$python get-pip.py
fi

tar --version > /dev/null 2>&1
if [[ !($? -eq 0) ]]; then
	sudo apt install tar gunzip
fi
wget --version > /dev/null 2>&1
if [[ !($? -eq 0) ]]; then
	sudo apt install tar gunzip
fi

wget https://github.com/DrWatt/NNFast/archive/refs/heads/main.tar.gz
tar -xvf main.tar.gz
rm -v main.tar.gz

$python -m pip install -r NNFast-main/requirements
wget https://github.com/google/qkeras/archive/refs/tags/v0.9.0.tar.gz
tar -xvf v0.9.0.tar.gz
rm -v v0.9.0.tar.gz
cd qkeras-0.9.0
$python -m pip install .
cd ..
rm -rvf qkeras-0.9.0

echo " $python  ${PWD}/NNFast-main/NNfast.py \"\$@\" " > fastNN
chmod a+x fastNN
cp ${PWD}/NNFast-main/lay.json .

