# AMPLE Python
AMPLE - A Material Point Learning Environment (python version)

AMPLE is a quasi-static implicit implementation of the material point method in MATLAB.  
More informatio about AMPLE in MATLAB can be obtained from the project webapges:
https://wmcoombs.github.io/

In this repository, I have ported the functionalities of AMPLE to Python, allowing it to run in a Python environment.

AMPLE is an elasto-plastic large deformation material point code with a regular quadrilateral background mesh 
(the main code is ample.py).   The continuum framework is based on an updated Lagrangian formation and two 
different constitutive models are included: linear elasticity and a linear elastic-perfectly plastic model 
with a von Mises yield surface.  

The code implements the example SetupGridCollapse from the original code, and you can run ample.py directly to see the results of this example (which are consistent with the original code). Additionally, the code should be able to run other examples from the original code, but I have not tested this. You can refer to utils\AnalysisParameters.py to add your test cases.

This code requires the following dependencies:
- numpy
- scipy
- joblib

It is important to note that this code exhibits some performance degradation compared to the original code.
