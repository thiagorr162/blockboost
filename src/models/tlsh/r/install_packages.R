#!/usr/bin/env Rscript

.libPaths( c( "./local_r_libraries" , .libPaths() ) )

install.packages('devtools',repos = "http://cran.us.r-project.org")
install.packages('plyr',repos = "http://cran.us.r-project.org")
install.packages('igraph',repos = "http://cran.us.r-project.org")

# local packages
devtools::install_github('cleanzr/blink')
devtools::install_github('cleanzr/tlsh')
 
