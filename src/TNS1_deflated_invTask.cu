#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <stdlib.h>
#include <math.h>

#define min(a,b) a<b ? a : b
#define max(a,b) a>b ? a : b
#define DEFL

using namespace std;

typedef double T;

typedef struct {
	unsigned int blocks;
	unsigned int blockSize;
	unsigned int blocksZ;
	unsigned int blockSizeZ;
	unsigned int Nx;
	unsigned int Ny;
	unsigned int size;
	unsigned int steps;
	unsigned int maxiter;
	T dt;
	T maxres;
	T maxresZ;
	unsigned int nRowsZ;
	unsigned int nDV;
    unsigned int NxZ;
} Parameters;


typedef struct {
	T rho;
	T cp;
	T k;
} MaterialProperties;

Parameters params;

MaterialProperties steel;
MaterialProperties Ag;
MaterialProperties MgO;
MaterialProperties inconel;
MaterialProperties NiCr;

#include "cpuFunctions.h"
#include "cpuFunctionsDeflation.h"
#include "cudaFunctions.h"
#include "cudaFunctionsDeflation.h"

int main(void)
{
    // Parameters
	params.blocks = 128;  
	params.blockSize = 128;  
	params.blocksZ = 4;  
    params.blockSizeZ = 128; 
	params.Nx  = 1024;
	params.Ny  = 1024;
	params.size = params.Nx*params.Ny + 2*params.Nx;
	params.steps  = 1;
	params.maxiter  = 1000000;
	params.dt = 0.003333333333;    // 0.0033333333333333;
	params.maxres  = 1e-7;
	params.maxresZ = 1e-20;
	params.nRowsZ = 16;    // number of rows/columns for one deflation vector
	params.nDV = (params.Nx*params.Ny) / (params.nRowsZ*params.nRowsZ);
	params.NxZ = params.Nx/params.nRowsZ; // number of course cells in a row
		
	// steel
	steel.rho = 7610.0; // 7700
    steel.cp  = 545.0; // 560
    steel.k   = 21.0;

    // Ag
    Ag.rho = 8957.0;
    Ag.cp  = 362.0; // 368
    Ag.k   = 120; // 111.5

    // MgO
    MgO.rho = 3150.0;
    MgO.cp  = 1110.0; // 1140
    MgO.k   = 11.5;   // 10
    
    // inconel
    inconel.rho = 8470.0;
    inconel.cp  = 500.0; // 520
    inconel.k   = 20.5; 
    
    // NiCr
    NiCr.rho = 8200.0;
    NiCr.cp  = 528.0;
    NiCr.k   = 24.5;
    
    T t = 0; // time
    T totalIter = 0;
    
    
    dim3 dimGrid(params.blocks);
    dim3 dimBlock(params.blockSize);
    dim3 dimGridZ(params.blocksZ);    // (deflation)
    dim3 dimBlockZ(params.blockSizeZ);
	
	cout << "example 1: inverse task (Pohanka)" << endl;
	cout << "nDV: " << params.nDV << ", params.NxZ: " << params.NxZ << endl;
	
	cpuInit(params.blocks, params.Nx, params.Ny);
	cpuInitDeflation(params.blocksZ, params.Nx, params.Ny, params.NxZ, params.nDV);
	
	readGeometry(hm, params.Nx, params.Ny);	// materials
	
	readCoords(xc, dx, "xCoords1024.txt");
	readCoords(yc, dy, "yCoords1024.txt");
	
	readBC(tHF, params.steps);
	
	initX(hT, params.Nx, params.Ny);
	initA(params.dt, params.Nx, params.Ny);
	
	initAZ(params.Nx, params.Ny, params.nRowsZ);
	initE(params.Nx, params.Ny, params.nRowsZ, params.NxZ);
	spChol(params.NxZ, params.nDV);
	
	//check_varZ(ecc, ess, params.nDV, params.NxZ);
	
	//saveData<int>(hm, "materials1024", params.Nx, params.Ny);
	//saveDataInTime(hT, t, "temperature1024_res1e-3_TNS1", params.Nx, params.Ny);
	
	//check_var(dy, params.Nx, params.Ny);

	// CUDA
	cudaInit(hT, hV, hcc, hss, hww, hqB, params.blocks, params.Nx, params.Ny);
	cudaInitDeflation(azc, azw, azs, ecc, eww, ess, params.NxZ, params.blocksZ, params.nDV, params.Nx, params.Ny);
	
	//makeTNS1<<<1024,1024>>>(kcc, kss, kww, dcc, dss, dww, params.Nx, params.Ny);
	makeTNS1f<<<1024,1024>>>(kcc, kss, kww, k3, dcc, dss, dww, params.Nx, params.Ny);

	
	cudaEvent_t startT, stopT;
	float elapsedTime;
	cudaEventCreate(&startT);
	cudaEventCreate(&stopT);
	cudaEventRecord(startT,0);

	for (int miter=0; miter<params.steps; miter++) {

		cudaMemcpy(dr, dT, sizeof(T)*params.size, cudaMemcpyDeviceToDevice);	// r = rhs
		elementWiseMul<<<1024,1024>>>(dr, dV, params.Nx);	          // r = V*r
		
		// add Neumann boundary here ... r = r + NeumannBC (dqB)
		//addNeumannBC<<<1,1024>>>(dr, dqB, (T)-1.0e6, params.Nx);    // constant
		addNeumannBC<<<1,1024>>>(dr, dqB, tHF[miter], params.Nx);     // time dependent

		SpMVv1<<<1024,1024>>>(dq, dcc, dss, dww, dT, params.Nx);      // q = Ax (version 1)
		AXPY<<<1024,1024>>>(dr, dq, (T)-1., (T)1., params.Nx);        // r = r - q
		
#ifdef DEFL
		
		// ---  (DEFLATION) r = Pr  ---
		///cudaMemset(dyZ,0,sizeof(T)*(params.nDV+2*params.NxZ)); // reset;
		ZTransXYDeflation<<<4, 1024>>>(drZ, dr, params.nRowsZ, params.NxZ, params.Nx);  // y1 = Z'*y
		// E*y2 = y1 (begin)
		cudaMemcpy(hrZ, drZ, (params.nDV+2*params.NxZ)*sizeof(double), cudaMemcpyDeviceToHost);   // copy drZ to hrZ
		solve(params.NxZ,params.nDV);
		cudaMemcpy(dyZ, hyZ, (params.nDV+2*params.NxZ)*sizeof(double), cudaMemcpyHostToDevice);   //copy hyZ to dyZ
		///printX(hyZ, params.NxZ,params.nDV);
		// E*y2 = y1 (end)
		YMinusAzXYDeflation<<<1024,1024>>>(dr,dyZ,dazc,dazw,dazs,params.nRowsZ,params.NxZ,params.Nx);  // r = P*r
#endif	    

		//SpMVv1<<<1024,1024>>>(dz, kcc, kss, kww, dr, params.Nx);    // z = M^(-1)r (version 1)
		SpMVv2<<<1024,1024>>>(dz, kcc, kss, k3, kww, dr, params.Nx);  // z = M^(-1)r (version 2)
		DOTGPU<T,128><<<dimGrid,dimBlock,params.blockSize*sizeof(T)>>>(drh, dr, dz, params.Nx, params.Ny);
		cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
		rhNew = dot(hrh,params.blocks);
		
		
		// --- stop criterion here ---
		DOTGPU<T,128><<<dimGrid,dimBlock,params.blockSize*sizeof(T)>>>(drh, dz, dz, params.Nx, params.Ny);
		cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
		stop = dot(hrh,params.blocks) * params.maxres * params.maxres;
		
		iter = 0;
		
		//cout << "stop:" << stop << ", residual: " << rhNew << endl;
		
		while (rhNew > stop && iter < params.maxiter) {
		//while (iter < 40) {  // only testing

			iter++;
			totalIter++;
			//cout << "iteration:" << iter << ", residual: " << rhNew << endl;
			//cout << rhNew << endl;

			if (iter==1) {
				cudaMemcpy(dp, dz, sizeof(T)*params.size,cudaMemcpyDeviceToDevice);
			}
			else {
				bt = rhNew/rhOld;	
				AXPY<<<1024,1024>>>(dp, dz, (T)1., bt, params.Nx);   // p = z + beta*p	
			}

			SpMVv1<<<1024,1024>>>(dq, dcc, dss, dww, dp, params.Nx);  // q = Ap (version 1)
			
#ifdef DEFL
			
			// ---  (DEFLATION) q = Pq  ---
			ZTransXYDeflation<<<4, 1024>>>(drZ, dq, params.nRowsZ, params.NxZ, params.Nx);  // y1 = Z'*y
			// E*y2 = y1 (begin)
			cudaMemcpy(hrZ, drZ, (params.nDV+2*params.NxZ)*sizeof(T), cudaMemcpyDeviceToHost);   // copy drZ to hrZ
			solve(params.NxZ,params.nDV);
			cudaMemcpy(dyZ, hyZ, (params.nDV+2*params.NxZ)*sizeof(T), cudaMemcpyHostToDevice);   //copy hyZ to dyZ
			// E*y2 = y1 (end)
			YMinusAzXYDeflation<<<1024,1024>>>(dq,dyZ,dazc,dazw,dazs,params.nRowsZ,params.NxZ,params.Nx);  // q = Pq
#endif



			DOTGPU<T,128><<<dimGrid,dimBlock,params.blockSize*sizeof(T)>>>(dsg, dp, dq, params.Nx, params.Ny);   // sigma = <p,q>
			//cudaMemcpy(&sg, dsg, 1*sizeof(T), cudaMemcpyDeviceToHost);
			//cudaMemset(dsg,0,sizeof(T)); // reset;
			cudaMemcpy(hsg, dsg, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
			sg = dot(hsg,params.blocks);
			ap = rhNew/sg;	// alpha = rhoNew / sigma
			AXPY<<<1024,1024>>>(dr, dq, -ap, (T)1., params.Nx);   // r = r - alpha*q
			AXPY<<<1024,1024>>>(dT, dp,  ap, (T)1., params.Nx);   // x = x + alpha*p

			//SpMVv1<<<1024,1024>>>(dz, kcc, kss, kww, dr, params.Nx);  // z = M^(-1)r (version 1)
			SpMVv2<<<1024,1024>>>(dz, kcc, kss, k3, kww, dr, params.Nx);  // z = M^(-1)r (version 2)

			rhOld = rhNew;

			DOTGPU<T,128><<<dimGrid,dimBlock,params.blockSize*sizeof(T)>>>(drh, dr, dz, params.Nx, params.Ny);   // rhoNew = <r,z>		
			//cudaMemcpy(&rhNew, drh, 1*sizeof(T), cudaMemcpyDeviceToHost);
			//cudaMemset(drh,0,sizeof(T));   // reset;
			cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
		    rhNew = dot(hrh,params.blocks);
		}
		
		// x = x +  (deflation)
		
		t += params.dt;
		cout << endl << "time: " << t << " ,timestep:" << miter << " ,iteration:" << iter << endl;
		
		//if ((miter+1)%4000==0)
		//{
		//cudaMemcpy(hT, dT, sizeof(T)*params.size, cudaMemcpyDeviceToHost);
		//saveDataInTime(hT, t, "temperature1024_res1e-3_TNS1float", params.Nx, params.Ny);
		//}

	}
	
	cudaEventRecord(stopT,0);
	cudaEventSynchronize(stopT);
	cudaEventElapsedTime(&elapsedTime, startT, stopT);
	cout<< "ellapsed time (cuda): " << elapsedTime << " miliseconds"	<< endl;
	
	cout << "Simulation finished." << endl;
	cout << "total number of iterations: " << totalIter << endl;
	
	cudaEventDestroy(startT);
	cudaEventDestroy(stopT);
	
	cudaFinalizeDeflation();
	cudaFinalize();
	
	cpuFinalizeDeflation();
	cpuFinalize();
	
	return 0;
}

