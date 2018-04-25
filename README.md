# TNS1_deflated_invTask2_sqr_cholCpu

Preconditioned Conjugate Gradient on GPU for Ax=b

- A is a 5 diagonal symmetric pos def matrix 
- condition number of A is improved by deflation
- the coarse system is solved Cholesky decomp. and the direct method on CPU.

- application: inverse task/heat diffusion in multimaterial domain 
