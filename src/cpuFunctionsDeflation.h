#ifndef CPUFUNCTIONSDEFLATION_H_
#define CPUFUNCTIONSDEFLATION_H_

// deflation
T *azc, *azw, *azs;
T *ecc, *eww, *ess;
double *L; // sparse lower triangular matrix after Cholesky decomposition
T *hrhZ, *hsgZ;
T rhNewZ, rhOldZ, sgZ, apZ, btZ;
T stopZ;
unsigned int iterZ;
T *auxVar;
T *hrZ, *hyZ;

// initialize CPU fields
//template <class T>
void cpuInitDeflation(const int blocks,
		const int Nx,
		const int Ny,
		const int NxZ,
		const int nDV)
{

	azc = new T[Nx*Ny];      // A*Z ... constructed explicitly on CPU
	azw = new T[Nx*Ny+1];
	azs = new T[Nx*Ny+Nx];
    ecc = new T[nDV+2*NxZ];      // E ... invertible Galerkin matrix
	eww = new T[nDV+1];
	ess = new T[nDV+NxZ];
	hrhZ = new T[blocks];
	hsgZ = new T[blocks];
	auxVar = new T[nDV+2*NxZ];
	L = new double[nDV*(NxZ+1)];
	hrZ = new T[nDV+2*NxZ];
	hyZ = new T[nDV+2*NxZ];


	memset(azc, 0, (Nx*Ny)*sizeof(T));
	memset(azw, 0, (Nx*Ny+1)*sizeof(T));
	memset(azs, 0, (Nx*Ny+Nx)*sizeof(T));
	memset(ecc, 0, (nDV+2*NxZ)*sizeof(T));
	memset(eww, 0, (nDV+1)*sizeof(T));
	memset(ess, 0, (nDV+NxZ)*sizeof(T));
	memset(hrhZ, 0, blocks*sizeof(T));
	memset(hsgZ, 0, blocks*sizeof(T));
	memset(auxVar, 0, (nDV+NxZ)*sizeof(T));
	memset(L, 0, (nDV*(NxZ+1))*sizeof(double));
	memset(hrZ, 0, (nDV+2*NxZ)*sizeof(T));
	memset(hyZ, 0, (nDV+2*NxZ)*sizeof(T));
}

void cpuFinalizeDeflation()
{
	delete[] azc;
	delete[] azw;
	delete[] azs;
	delete[] ecc;
	delete[] eww;
	delete[] ess;
	delete[] hrhZ;
	delete[] hsgZ;
	delete[] auxVar;
	delete[] L;
	delete[] hrZ;
	delete[] hyZ;
}

// initialize AZ (=A*Z); Z ... deflation subspace matrix
void initAZ(const int Nx,
		const int Ny,
		const int nRowsZ)
{
	// --- azc, azw, azs ---
	for (int j=0; j<Ny; j++)
	{
		for (int i=0; i<Nx; i++)
		{
			int id  = j*Nx + i;                    			// linear system of original system
			//azc                                            		   _
			azc[id] = hcc[id+Nx];                 			//          |
			if (i%nRowsZ!=0)     azc[id] += hww[id];    	//          |
			if ((i+1)%nRowsZ!=0) azc[id] += hww[id+1];  	//          |
			if (j%nRowsZ!=0)     azc[id] += hss[id];    	//          |
			if ((j+1)%nRowsZ!=0) azc[id] += hss[id+Nx];		//           >  compact and works fine!
			//azw                                              			|
			if (i%nRowsZ==0) 	 azw[id]  = hww[id];   		//          |
			//azs                                              			|
			if (j%nRowsZ==0)     azs[id]  = hss[id];    	//         _|
		}
	}
}


// initialize E (Rspace = nDVxnDV); E ... invertible Galerkin matrix
void initE(const int Nx,
		const int Ny,
		const int nRowsZ,
		const int NxZ)
{
	for (int j=0; j<Ny; j++)                 // works just fine!
	{
		for (int i=0; i<Nx; i++)
		{
			int ixZ = i/nRowsZ;               // x-index of coarse system
			int jyZ = j/nRowsZ;               // y-index
			int idZ = jyZ*NxZ + ixZ;          // linear index of course system
			int id  = j  *Nx  + i;            // linear system of original system

			// ECC
			ecc[idZ+NxZ] += hcc[id+Nx];

			if (i%nRowsZ!=0)
			{
				ecc[idZ+NxZ] += 2*hww[id];        // 2*hww due to the fact that for one cell it is WEST flux, for the neighbor it is EAST flux.
			}
			if (j%nRowsZ!=0)
			{
				ecc[idZ+NxZ] += 2*hss[id];        // adding hww as well as hww could be avoided if hcc did NOT include fluxes!
			}

			//EWW
			if (i%nRowsZ==0)  eww[idZ] += hww[id];

			//ESS
			if (j%nRowsZ==0)  ess[idZ] += hss[id];
		}
	}
}

void spChol(const int NxZ,const int nDV)
{
	T sumL2jk, sumLikLjk;

	for (int i=0; i<nDV; i++)
		{
			int ixm = min(NxZ,i);
			sumL2jk = 0.;                      // reset sum of squares of Ljk for forthcoming Lj,j calculation

			for (int j=0; j<ixm; j++)
			{
				int jr = int(max(i-NxZ, 0)) + j;
				int jy = ixm - j;
				int idL = jr + nDV * jy;
				sumLikLjk = 0.;                // reset sum of squares of Lik*Ljk

				 for (int k=j-1; k>=0; k--)
				 {
					 int jrk = jr - k - 1;   // for Lik as well as Ljk
					 int jyi = jy + k + 1;     // Lik
					 int jyj =    + k + 1;     // Ljk
					 sumLikLjk += L[jrk+nDV*jyi] * L[jrk+nDV*jyj];
				 }
				 switch(jy)
				 {
				 case 1:
					 L[idL] = 1/L[jr]  * (eww[jr+1] - sumLikLjk);
					 break;
				 case 64:   // N/p !!!
					 L[idL] = 1/L[jr]  * (ess[jr+NxZ] - sumLikLjk);
					 break;
				 default:
					 L[idL] = 1/L[jr]  * (            - sumLikLjk);

				 }
				 sumL2jk = sumL2jk + L[idL]*L[idL];
			}
			L[i] = sqrt(ecc[i+NxZ] - sumL2jk);
		}
}

void solve(const int NxZ,const int nDV)
{
	// forward sweep
	for (int i=0; i<nDV; i++)
	{
		hyZ[i+NxZ] = hrZ[i+NxZ];
		int ixm = min(NxZ,i);
		int iym = max(i-NxZ, 0);

		for (int j=0; j<ixm; j++)
		{
			//int jr = iym + j;
			//int jy = ixm - j;
			//int idL = jr + nDV * jy;
			//hyZ[i+NxZ] -= L[idL] * hyZ[jr+NxZ];
			hyZ[i+NxZ] -= L[iym+j+nDV*(ixm-j)] * hyZ[iym+j+NxZ];
		}
		hyZ[i+NxZ] /= L[i];
	}

	// back sweep
	for (int i=nDV-1; i>=0; i--)
	{
		for (int j=0; j<NxZ; j++)
		{
			//int idL = i + nDV * (j+1);
			hyZ[i+NxZ] -= L[i + nDV * (j+1)] * hyZ[i+j+1+NxZ];
		}
		hyZ[i+NxZ] /= L[i];
	}
}

#endif /* CPUFUNCTIONSDEFLATION_H_ */
