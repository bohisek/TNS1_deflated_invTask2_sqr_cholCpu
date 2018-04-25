#ifndef CUDAFUNCTIONSDEFLATION_H_
#define CUDAFUNCTIONSDEFLATION_H_

// deflation
T *dazc, *dazw, *dazs;
T *decc, *deww, *dess;
T *dyZ, *drZ, *dpZ , *dqZ;
T *drhZ, *dsgZ;        // partial dot products

// initialize CUDA fields
template <class T>
void cudaInitDeflation( T *azc,
		T *azw,
		T *azs,
		T *ecc,
		T *eww,
		T *ess,
		const int NxZ,
		const int blocks,
		const int nDV,
		const int Nx,
		const int Ny)
{
	cudaMalloc((void**)&dazc ,sizeof(T)*(Nx*Ny));
	cudaMalloc((void**)&dazw ,sizeof(T)*(Nx*Ny+1));
	cudaMalloc((void**)&dazs ,sizeof(T)*(Nx*Ny+Nx));
	cudaMalloc((void**)&dyZ  ,sizeof(T)*(nDV+2*NxZ));
	cudaMalloc((void**)&drZ  ,sizeof(T)*(nDV+2*NxZ));
	cudaMalloc((void**)&dpZ  ,sizeof(T)*(nDV+2*NxZ));
	cudaMalloc((void**)&dqZ  ,sizeof(T)*(nDV+2*NxZ));
	cudaMalloc((void**)&decc ,sizeof(T)*(nDV+2*NxZ));
	cudaMalloc((void**)&deww ,sizeof(T)*(nDV+1));
	cudaMalloc((void**)&dess ,sizeof(T)*(nDV+NxZ));
	cudaMalloc((void**)&drhZ ,sizeof(T)*(blocks));
    cudaMalloc((void**)&dsgZ ,sizeof(T)*(blocks));


	cudaMemcpy(dazc ,azc ,sizeof(T)*(Nx*Ny)    ,cudaMemcpyHostToDevice);
	cudaMemcpy(dazw ,azw ,sizeof(T)*(Nx*Ny+1)  ,cudaMemcpyHostToDevice);
	cudaMemcpy(dazs ,azs ,sizeof(T)*(Nx*Ny+Nx) ,cudaMemcpyHostToDevice);
	cudaMemcpy(decc ,ecc ,sizeof(T)*(nDV+2*NxZ),cudaMemcpyHostToDevice);
	cudaMemcpy(deww ,eww ,sizeof(T)*(nDV+1)    ,cudaMemcpyHostToDevice);
	cudaMemcpy(dess ,ess ,sizeof(T)*(nDV+NxZ)  ,cudaMemcpyHostToDevice);

	cudaMemset(dyZ ,0,sizeof(T)*(nDV+2*NxZ));
	cudaMemset(drZ ,0,sizeof(T)*(nDV+2*NxZ));
	cudaMemset(dpZ ,0,sizeof(T)*(nDV+2*NxZ));
	cudaMemset(dqZ ,0,sizeof(T)*(nDV+2*NxZ));
	cudaMemset(drhZ,0,sizeof(T)*(blocks));
	cudaMemset(dsgZ,0,sizeof(T)*(blocks));
}

void cudaFinalizeDeflation()
{
	cudaFree(dazc);
	cudaFree(dazw);
	cudaFree(dazs);
	cudaFree(dyZ);
	cudaFree(drZ);
	cudaFree(dpZ);
	cudaFree(dqZ);
	cudaFree(decc);
	cudaFree(deww);
	cudaFree(dess);
	cudaFree(drhZ);
	cudaFree(dsgZ);
}

// ---- y1 = Z' * y (works!) ----
template <class T>
__global__ void ZTransXYDeflation(T *y,
		const T *x,
		const int nRowsZ,
		const int NxZ,
		const int Nx)
{
	// nDV (number of deflation vectors) threads must be launched!
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	int id0 = nRowsZ*Nx*(tid/NxZ) + nRowsZ*(tid%NxZ);         // a global index of the first cell in a coarse cell  (parenthesis are necessary!)

	y[tid+NxZ] = 0.;	                                      // reset

	for (int j=0; j<nRowsZ*nRowsZ; j++)                       // loop over cells of a single coarse cell
	{
		y[tid+NxZ] += x[id0 + Nx*(j/nRowsZ) + j%nRowsZ + Nx];
	}
}

// ---- y (= Py) = y - AZ * y2 ----
template <class T>
__global__ void YMinusAzXYDeflation(T *y,
		const T *x,
		const T *azc,
		const T *azw,
		const T *azs,
		const int nRowsZ,
		const int NxZ,
		const int Nx)
{
		// Nx*Ny threads must be launched!
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		int ixZ = (tid%Nx)/nRowsZ;        // x-index of coarse system
		int jyZ = (tid/Nx)/nRowsZ;        // y-index
		int idZ = jyZ*NxZ + ixZ;          // linear index of course system

		idZ += NxZ;

		y[tid+Nx]  -= + azc[tid]    * x[idZ]  // center
		         + azw[tid]    * x[idZ-1]     // west
		         + azw[tid+1]  * x[idZ+1]     // east
		         + azs[tid]    * x[idZ-NxZ]   // south
		         + azs[tid+Nx] * x[idZ+NxZ];  // north
	}


// AXPY (y := alpha*x + beta*y)
template <class T>
__global__ void AXPYDeflation(T *y,
		const T *x,
		const T alpha,
		const T beta,
		const int NxZ)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + NxZ;
	y[tid] = alpha * x[tid] + beta * y[tid];
}

// DOT PRODUCT
template <class T, unsigned int blockSize>
__global__ void DOTGPUDeflation(T *c,
		const T *a,
		const T *b,
		const int nDV,
		const int NxZ)
{
	extern __shared__ T cache[];

	unsigned int tid = threadIdx.x;
	unsigned int i = tid + blockIdx.x * (blockSize * 2);
	unsigned int gridSize = (blockSize*2)*gridDim.x;


	cache[tid] = 0;

	while(i<nDV) {
		cache[tid] += a[i+NxZ] * b[i+NxZ] + a[i+NxZ+blockSize] * b[i+NxZ+blockSize];
		i += gridSize;
	}

	__syncthreads();

	if(blockSize >= 512) {	if(tid < 256) { cache[tid] += cache[tid + 256]; } __syncthreads(); }
	if(blockSize >= 256) {	if(tid < 128) { cache[tid] += cache[tid + 128]; } __syncthreads(); }
	if(blockSize >= 128) {	if(tid < 64 ) { cache[tid] += cache[tid + 64 ]; } __syncthreads(); }

	if(tid < 32) {
		cache[tid] += cache[tid + 32];
		cache[tid] += cache[tid + 16];
		cache[tid] += cache[tid + 8];
		cache[tid] += cache[tid + 4];
		cache[tid] += cache[tid + 2];
		cache[tid] += cache[tid + 1];
	}

	if (tid == 0) c[blockIdx.x] = cache[0];
}

// SPMV (sparse matrix-vector multiplication)
template <class T>
__global__ void SpMVv1Deflation(T *y,
		const T *x,
		const T *stC,
		const T *stW,
		const T *stS,
		const int NxZ)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + NxZ;

	y[tid]  = stC[tid]       * x[tid]      // center
	        + stS[tid]       * x[tid+NxZ]  // north               N
	        + stW[tid-NxZ+1] * x[tid+1]    // east              W C E
	        + stS[tid-NxZ]   * x[tid-NxZ]  // south               S
	        + stW[tid-NxZ]   * x[tid-1];   // west

}


#endif /* CUDAFUNCTIONSDEFLATION_H_ */
