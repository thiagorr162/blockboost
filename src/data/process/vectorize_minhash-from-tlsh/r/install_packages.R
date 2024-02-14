#!/usr/bin/env Rscript

.libPaths( c( "./local_r_libraries" , .libPaths() ) )

install.packages('devtools')
install.packages('plyr')

# local packages
devtools::install_github('cleanzr/blink')
devtools::install_github('cleanzr/tlsh')

