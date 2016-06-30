# joint-demosaicing-denoising-sem
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------

Demo Code for the paper "Joint Demosaicing and Denoising Based on
Sequential Energy Minimization"

Copyright (C) 2016 Institute for Computer Graphics and Vision (ICG)
Graz University of Technology, Austria

-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------

This is a plain python implementation of the forward path corresponding to the SEM16
model presented in the paper. The required trained parameters are provided and named
'theta.npy'. The code has been written and tested under Linux using Python 2.7 and
requires following packages: numpy, scipy, matplotlib, and skimage. Please note that the
code is not optimized at all, but runs without dependencies. The timings in the paper
are computed with a different version of the code (using the theano package on a gpu).

To run the code to jointly demosaic and denoise your images, please load
demosaicing_demo.py into your python IDE and run it, or start it directly
from the terminal via
$ python demosaicing_demo.py

Some test images are provided, see the remark below.

-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------

If you use this code, please cite the following publication:

~~~
@inproceedings{klatzer_iccp2016,
author = {Teresa Klatzer and Kerstin Hammernik and Patrick Knöbelreiter and Thomas Pock},
title = {{Joint Demosaicing and Denoising Based on Sequential Energy Minimization}},
booktitle = {2016 IEEE International Conference on Computational Photography (ICCP)},
year = {2016},
doi={10.1109/ICCPHOT.2016.7492871},
}
~~~

For further questions, feel free to contact me via: klatzer@icg.tugraz.at

-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------

5 test images are provided from the Microsoft Demosaicing Dataset (folder MSR-Demosaicing).
A separate license file is enclosed in this folder.

~~~
@article{ msrdemosaicing2015,
    title = "Joint Demosaicing and Denoising via Learned Nonparametric Random Fields",
    author = "Daniel Khashabi and Sebastian Nowozin and
              Jeremy Jancsary and Andrew W. Fitzgibbon",
    journal = "IEEE Transactions on Image Processing",
    year = "2014",
    number = "12",
    volume = "23",
    pages = "4968--4981",
    URL = "http://dx.doi.org/10.1109/TIP.2014.2359774",
}
~~~
