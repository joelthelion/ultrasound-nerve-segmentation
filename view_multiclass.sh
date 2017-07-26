#!/bin/bash
# use like this: for i in patient0*.png; do echo ${i%_pred.png}; ./view_multiclass.sh ${i%_pred.png}; done
itksnap -g datadir/${1}*segmentation*.png -s ${1}*.png
itksnap -g datadir/${1}_4CH_ED.png -s ${1}*.png
