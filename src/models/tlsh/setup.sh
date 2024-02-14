#!/bin/bash

ln -s /impa/home/a/rodrigo.loro/shared_r_libraries ./local_r_libraries

## UNCOMMENT TO COMPILE YOURSELF (NOT RECOMMENDED)
#
#unlink ./local_r_libraries
#
#set -x
#
## changing this path requires updates in src/models/tlsh/r/
#mkdir "$HOME/local_r_libraries"
#./src/models/tlsh/r/install_packages.R
