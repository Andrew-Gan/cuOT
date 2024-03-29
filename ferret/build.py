#!/usr/python
import subprocess
install_packages = '''
if [ "$(uname)" == "Darwin" ]; then
	brew list openssl || brew install openssl
 	brew list pkg-config || brew install pkg-config
 	brew list cmake || brew install cmake
else
    if command -v apt-get >/dev/null; then
        sudo apt-get install -y software-properties-common
        sudo apt-get update
        sudo apt-get install -y cmake git build-essential libssl-dev
    elif command -v yum >/dev/null; then
        sudo yum install -y python3 gcc make git cmake gcc-c++ openssl-devel
    else
        echo "System not supported yet!"
    fi
fi
'''

install_template = '''
if [ ! -d "REPO" ]; then
    git clone https://github.com/emp-toolkit/REPO.git --branch Y
fi
cd REPO
cmake -DCMAKE_INSTALL_PREFIX=../lib \
    -DCMAKE_C_FLAGS='-g'

make -j4
make install
cd ..
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-install', '--install', action='store_true')
parser.add_argument('-deps', '--deps', action='store_true')
parser.add_argument('--tool', nargs='?', const='master')
parser.add_argument('--ot', nargs='?', const='master')
parser.add_argument('--sh2pc', nargs='?', const='master')
parser.add_argument('--ag2pc', nargs='?', const='master')
parser.add_argument('--agmpc', nargs='?', const='master')
parser.add_argument('--zk', nargs='?', const='master')
args = parser.parse_args()

for k in ['tool', 'ot', 'zk', 'sh2pc', 'ag2pc', 'agmpc']:
	if vars(args)[k]:
		template = install_template.replace("REPO", "emp-"+k).replace("Y", vars(args)[k])
		print(template)
		subprocess.call(["bash", "-c", template])
