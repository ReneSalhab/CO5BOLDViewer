# CO5BOLDViewer 0.8

    # Description

      A QT-based viewer of .uio-data used by CO5BOLD. Written in Python 3.5.

    # Requirements
    
      The following modules are used:
      
      - os
      - re
      - sys
      - time
      - math
      - h5py
      - numpy
      - scipy
      - PyQt5
      - struct
      - bisect
      - numexpr
      - astropy
      - matplotlib
      - collections

    # Installation
    
      No compilation needed, but highly recommended (see section "Compilation").
      Go to the directory you installed CO5BOLDViewer and start with
      
        bash$ python CO5BOLDViewer.py
      
      or
      
        bash$ ipython CO5BOLDViewer.py

      If all necessary modules are present it should work. If modules are missing you´ll have to install them.
      With anaconda type into the terminal/console:
      
        bash$ conda install pyqt5
      
      If python was not installed with anaconda
      
        bash$ pip install pyqt5
      
      should work.

    # Compilation (optional)

      To compile functions for computation of temperature, pressure, entropy, opacity and optical depth
      located in eosinterx.pyx, go to the directory you installed CO5BOLDViewer and type the following command

        bash& python setup.py build_ext --inplace

      ipython won´t work for compilation.

      ## Benefits:

        - for opacity and optical depth the results will be more accurate.
        - better performance for most functions
        - less memory consumption when computing temperature, pressure and entropy

       The quantities will still be available, if eosinterx.pyx is not compiled, but opacity and optical depth
       are not computed accurately, yet. Also the memory usage is much worse.

    # Change-Log
    
      ## Version 0.8.1:
   
        - changed EOS-based computations from double to single precision (closed memory leak)
        - enhanced speed of opacity computation
        - introduced computation of divergence of magnetic field for test-purpose

      ## Version 0.8:

        - computing-option implemented
        - eosinter changed to class (EosInter)
        - par-file reader introduced (not implemented, yet)
        - Opac-class introduced. Replaces numpy-functions within setPlotData in rootelements.py
        - version-number introduced
        - cython-code for EosInter and Opac introduced. Compilation needed and highly recommended.
        - minor bugs fixed

    # Outlook
    
        - in the next version (0.8.2) compilation of eosinterx.pyx will be mandatory
