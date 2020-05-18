# Non-convex ADMM for solving Sudoku
This repo contains convex and non-convex admm implementations for solving 9x9 sudokus (in python and matlab). 

Note that despite the fact that this is a greedy approach, it does relatively well at solving not deep sudoku problems. This is an interesting phenomena. It is interesting to figure out which kind of sudokus are solvable/not solvable by this method and why.

The MATLAB implementation is much faster due to the efficiancy of matrix computations in MATLAB. Note that it si possible to make the implementations much faster if one uses distributed multi-thread computations. This is because the ADMM is very flexible in this regard. 

For the examples of how to give the input to the python implementation, see the end of the .py file. The MATLAB code takes in an instance that is a 9x9 matrix where the hidden values are replaced with zeros.

I know that the codes are not well cleaned. I will try to clean them up further in the near future :-)
