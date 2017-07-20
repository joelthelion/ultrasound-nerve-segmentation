#!/bin/bash
c3d $1*.png -thresh -inf 128 0 1 -type uchar -o seg.png
itksnap -g datadir/$1*segmentation*.png -s seg.png
#itksnap -g datadir/$1_4CH_ED.png -s seg.png
