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
    
      To be able to compute temperature, pressure, entropy, optical depth and opacity, the compilation of 
	  the cython-based eosinterx.pyx is necessary. For this, go to the directory you installed CO5BOLDViewer
	  compile eosinterx.pyx with
	  
	    bash& python setup.py build_ext --inplace
	 
	  Ipython won´t work for compilation.

	  Notes: It is necessary that a C-compiler is installed. Some functions are parallelized with openmp.
	  
	  If modules are missing you´ll have to install them. In case anaconda is installed, the command
      
        bash$ conda install pyqt5
		
	  can be used (pyqt5 is an example).
      
      If python was not installed with anaconda, use
      
        bash$ pip install pyqt5
		
	  If everything is installed and compiled, the command

	    bash$ python CO5BOLDViewer.py
      
      or
      
        bash$ ipython CO5BOLDViewer.py

	  will start the CO5BOLDViewer. If something goes wrong, please let me know.

    # Change-Log
    
	  ## Version 0.8.2:
	  
	    - removed possibility of computing pressure, temperature, entropy, opacity and optical depth without
		  compilation
		- compilation of eosinterx.pyx is now mandatory
	
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
