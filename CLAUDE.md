Here I'm developing a project that convert a rvv kernel into a c scalar kernel, to locate where the behavior error lies, And where the precision error inccured.

The workflow here should be 
1. run convert_rvv_to_scalar.py and choose an input file to genrate the .h file including scalar version.
2. build validate_scalar.cpp including the header file to generate testcase and compare the result.
