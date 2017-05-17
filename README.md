# CO5BOLDViewer 0.8.5

A QT-based viewer of .uio-data used by CO5BOLD. Written in Python 3.5.

## Requirements
    
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

## Installation
    
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

## HOW TO

### ...use the uio-module

The uio-module provides the possibility to read uio-files like .full, .mean, .sta, .end and also .eos-files.

To use the uio module, you will have to go to the directory, where the uio.py-file is stored. Open a python-shell
and import the module via

    module load uio

To open a specific file, store the path and name as a string and give it as a parameter to the File object of the
uio module

    filename = r"/path/to/your/file/rhd.full"
    fil = uio.File(filename)

The File object reads through the desired file and searches for the positions of any "Dataset", "Box" and "Entry",
or "Block" in case of .eos-files (for more information about the uio file-format, read the manual of CO5BOLD), thus
that data can be read just-in-time. So, if you load the density of snapshot 0 within your model, via

    rho = fil.dataset[0].box[0]['rho'].data

the values of rho will put into RAM just then. This spares memory. Some explanations of the syntax above:
The File-instance fil consists of the lists dataset and box. dataset is a snapshot, so fil.dataset[0] is the
first snapshot of the uio-file, fil.dataset[1] is the second, and so on. datasets include entries and boxes.
One entry of a dataset is the modeltime, which can be accessed by

    fil.dataset[1]['modeltime'].data

The "data"-member of the Entry-class (here the instance of 'modeltime') returns the values of the sought-after
quantity. Other members are "pos" (position within the uio-file), "type" (a string describing the FORTRAN-type of the
quantity like it is stored within the uio-file), "name" (the name of the quantity, here "modeltime"), "params" (a
dictionary of different parameters, like number of bytes "b", unit "u" and others), "dtype" (the PYTHON-type) and "shape"
(the shape of the cube, or scalar). As you can see , the Dataset-class (elements of the dataset-list) is handled like an
ordinary dictionary.

Example:

    print(fil.dataset[1].box[0]['v1'].data)     # returns the whole data-cube of 'v1' as numpy-array
    print(fil.dataset[1].box[0]['v1'].pos)      # returns, e.g. 1204466642
    print(fil.dataset[1].box[0]['v1'].type)     # returns 'real'
    print(fil.dataset[1].box[0]['v1'].name)     # returns 'v1'
    print(fil.dataset[1].box[0]['v1'].dtype)    # returns 'f4'
    print(fil.dataset[1].box[0]['v1'].shape)    # returns, e.g. (188, 400, 400)
    print(fil.dataset[1].box[0]['v1'].params)   # returns, e.g. {'b': '4', 'd': '(1:400,1:400,1:188)', 'f': 'E13.6',
                                                #                'n': 'Velocity 1', 'p': '4', 'u': 'cm/s'}

The same holds for the Box-class. They consist of a list of entries, which itself is handled like a dictionary. The
elements of the box-list are the data-cubes of the different quantities stored by CO5BOLD, so "rho", "ei", "v1", "v2", "v3", the axes
"xc1", "xc2", "xc3", "xb1", "xb2", "xb3" and if the mhd-module was used, "bb1", "bb2" and, "bb3". These are entries
of the box-list.

As the File-, DataSet and Box-classes are behaving as dictionaries, all included entries of each object can be printed
out with the keys-method, i.e.

    print fil.keys()         # python 2.x
    print(list(fil.keys()))  # python 3.x

As mentioned the uio-module can also be used with .eos-files. The File-object in this case has only entries and
block-members, which behave like datasets and boxes. With the keys-method, the entries can be printed out. It is
recommended to use the eosinter-module described below.

It is also possible to use the "with-statement", which closes the opened file automatically.

    with uio.File(filename) as fil:
        # do something with the file

### ...use the eosinter-module

The eosinter-module can be used to read out .eos-files. In contrast to the use of the uio-module, the eosinter-module
reads out the data provided by the .eos-file. The module consists of a EosInter-class. It uses the uio-module to read
the desired .eos-file and stores the values directly into memory, modified in a way that further usage is more
comfortable. To instantiate the EosInter-class, the path and name of the .eos-file has to provided as a string:

    from eosinter import EosInter
    
    eosname = r"/path/to/eos/file/xyz.eos"
    eos = EosInter(eosname)

The eos-object has methods to compute entropy, temperature, or pressure (STP), pressure and temperature simultaneously
(PandT), pressure and its derivations (Pall), or temperature and its derivations (Tall). All methods need the
mass-density and internal energy to compute the desired quantities. Furthermore, STP (abbreviation for entropy S,
temperature T and pressure P) needs to know which quantity is shall compute.

Examples:

    P = eos.STP(rho, ei)                  # computes the pressure, or quantity='P', 'p', 'pressure', 'Pressure'
    T = eos.STP(rho, ei, quantity='T')    # computes the temperature, or quantity='t', 'temperature', 'Temperature'
    S = eos.STP(rho, ei, quantity='E')    # computes the entropy, or quantity='e', 'entropy', 'Entropy', 'S', 's'
    P, T = eos.PandT(rho, ei)
    P, dPdrho, dPdei = eos.Pall(rho, ei)
    T, dTdei = eos.Tall(rho, ei)

To be able to use the eosinter-module, the eosinterx.pyx module has to be compiled. For compilation-introductions read
the "Installation"-section. 

### ...use the opta-module

The syntax follows the eosinter-module. The opta-module consists of the Opac-class. Instantiation:

    from opta import Opac
    
    opaname = r"/path/to/opta/file/xyz.opta"
    opa = Opac(opaname)

The opa-object comes with different methods. The most important methods are the kappa-method that computes the opacity
for a specified band and tau, which computes the optical depth.

The kappa-method needs temperature and pressure as parameters. The band-parameter is optional and set to 0 (bolometric)
ss a default. Usage:
 
    opa.kappa(T, P, iBand=1)    # computes the opacity of the 2. band.
 
For tau there are several possibilities. The mandatory parameter is mass-density rho. Then a scale-height has to be
handed, the cell-centered height-scale z, or, preferably, the boundary-centered height-scale zb. If the latter is provided, the
vertical cell-sizes are computed directly. If the former is handed, the upper-most cell-size will be set by the cell-size
one layer below. For the computation the opacity can be handed directly (keyword 'kappa'), or the temperature and
pressure which leads to the computation of the opacity before computing optical depth. Here the band can be specified
again. Also it has to be clear, which axis is the LOS-axis. Internally the quantities will be transposed in a way that
last axis is parallel to the LOS and after computation transposed back. By default axis is set to -1 (last axis of the
arrays) So these are the possibilities for using tau:

    opa.tau(rho, axis=0, z=z, kappa) 
    opa.tau(rho, axis=1, zb=zb, T=T, P=P, iBand=1)
    opa.tau(rho, zb=zb, T=T, P=P)

Additional methods are height and quant_at_tau. The former is used to compute the height-profile of specified
tau-value(s). The latter is used to compute the values of a desired quantity at specified tau-values.

The method height needs the cell-centered height-scale z. Other parameters are value, which can be a float, or numpy
array. In the latter case, the height-profiles of all tau-values will be computed. The parameter axis is the
LOS-axis. Furthermore tau has to be provided, or rho and kappa, or rho, T and P. An optional parameter is iBand
and the boundary-centered height-scale zb.

Examples:

    opa.height(z, axis=0, value=np.logspace(-4, 5, 200), tau=tau)
    opa.height(z, axis=0, rho=rho, kappa=kappa)     # computes the height-profile for tau=1
    opa.height(z, axis=1, value=2/3, rho=rho, T=T, P=P)

The usage of quant_at_tau is similar to height, but the sought-after quantity has to be provided.

Examples:

    opa.quant_at_tau(T, new_tau=np.logspace(-4, 5, 200), axis=0, tau=tau)
    opa.quant_at_tau(P, axis=0, rho=rho, kappa=kappa)     # computes the pressure at tau=1
    opa.quant_at_tau(rho, axis=1, value=2/3, rho=rho, T=T, P=P)

### ...use the par-module

The par-module consists of the ParFile-class, which can be used like the File-class of the uio-routine. It only has
Entry-objects, which means that there is no sub-class. The File-object behaves like a dictionary, but the values are
stored in the member "data" (like in the uio-module).

Example:

    from par import ParFile
    
    parname = r"/path/to/par/file/rhd.par"
    parf = ParFile(parname)
    
    print(list(parf.keys())) # prints the available entries of parf
    print(parf['opafile'])   # prints the file-name of the used opacity-file
    print(parf['opapath'])   # prints the path to the used opacity-file

## Change-Log

### Version 0.8.6

- FIX: Computation of height-profiles of iso-tau levels resulted in NANs when tau is not provided
- FIX: Computation of quantities at iso-tau levels resulted in NANs if tau provided.
- MODIFICATION: height-scale z for computation of tau was not necessary. Is removed as parameter
- MODIFICATION: ParFile-class in par-module behaves similar to File-class of uio-module.
- MODIFICATION: representation of File-class in uio-module changed
- NEW: Introduced HOW TOs in README.md
- NEW: If new model is loaded, "rhd.par" is searched within same directory. If available, it will be loaded and used to find the right eos- and opta-files
- NEW: Option to load parameter-file included in "File"-dropdown menu of the menu-bar
- NEW: Introduced indicators for showing if parameter-, eos-, or opta-files is loaded
- FIX: some minor fixes

### Version 0.8.5

- FIX: When aborting file-load, error occurred
- FIX: menu-bar was not usable under MacOS. Now it is in the window of CO5BOLDViewer and working.
- NEW: tau=1 location at side-view plotable, of eos and opa are provided
- NEW: new setup_np.py (not parallel) provided, if compilation fails with setup.py, because of openmp

### Version 0.8.4

- FIX: cross-hair is not working
- FIX: arrows of magnetic field don´t have correct length in Gausz
- NEW: current file is shown, when going through time-series

### Version 0.8.3

- log-10 and log-10 of absolute value implemented
- fixed a bug, when loading little-endian eos-files
- fixed a bug, when clicking outside of plot-area
- introduced check-box for inverting colormap
- set default colormap to "inferno"

All changes in this version were provided by Derek Homeier. Thank you on this occasion.

### Version 0.8.2:

- removed possibility of computing pressure, temperature, entropy, opacity and optical depth without
  compilation
- compilation of eosinterx.pyx is now mandatory

### Version 0.8.1:

- changed EOS-based computations from double to single precision (closed memory leak)
- enhanced speed of opacity computation
- introduced computation of divergence of magnetic field for test-purpose

### Version 0.8:

- computing-option implemented
- eosinter changed to class (EosInter)
- par-file reader introduced (not implemented, yet)
- Opac-class introduced. Replaces numpy-functions within setPlotData in rootelements.py
- version-number introduced
- cython-code for EosInter and Opac introduced. Compilation needed and highly recommended.
- minor bugs fixed
