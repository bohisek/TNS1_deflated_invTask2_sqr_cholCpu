#ifndef CPUFUNCTIONS_H_
#define CPUFUNCTIONS_H_

// declare CPU fields
int *hm; // materials
T *hT, *hcc, *hss, *hww;
T *hrh, *hsg;
T *xc, *dx;
T *yc, *dy;
T rhNew, rhOld, sg, ap, bt;
T stop;
unsigned int iter;
T *hV;
T *hqB; // Neumann condition at the bottom
T tHF[400001];  // static array (size = params.steps)

// initialize CPU fields
void cpuInit( const int blocks,
		const int Nx,
		const int Ny)
{
	hm  = new int[Nx*Ny+2*Nx];
	hT  = new T[Nx*Ny+2*Nx];
	hcc = new T[Nx*Ny+2*Nx];
	hV  = new T[Nx*Ny+2*Nx];
	hss = new T[Nx*Ny+Nx];
	hww = new T[Nx*Ny+1];
	hrh = new T[blocks];
    hsg = new T[blocks];

	xc  = new T[Nx];
	dx  = new T[Nx];
	yc  = new T[Ny];
	dy  = new T[Ny];
	hqB = new T[Nx];

	memset(hm, 0, (Nx*Ny+2*Nx)*sizeof(int));
	memset(hT, 0, (Nx*Ny+2*Nx)*sizeof(T));
	memset(hcc,0, (Nx*Ny+2*Nx)*sizeof(T));
	memset(hV ,0, (Nx*Ny+2*Nx)*sizeof(T));
	memset(hss,0, (Nx*Ny+Nx  )*sizeof(T));
	memset(hww,0, (Nx*Ny+1   )*sizeof(T));
	memset(hrh,0, (blocks    )*sizeof(T));
    memset(hsg,0, (blocks    )*sizeof(T));
	memset(xc, 0, Nx*sizeof(T));
	memset(dx, 0, Nx*sizeof(T));
	memset(yc, 0, Ny*sizeof(T));
	memset(dy, 0, Ny*sizeof(T));
	memset(hqB,0, Nx*sizeof(T));

	rhNew = 0;
	sg    = 0;
}

// free memory
void cpuFinalize()
{
	delete[] hm;
	delete[] hT;
	delete[] hV;
	delete[] hcc;
	delete[] hss;
	delete[] hww;
	delete[] hrh;
	delete[] hsg;
	delete[] xc;
	delete[] dx;
	delete[] yc;
	delete[] dy;
	delete[] hqB;
}

T thermCond(int i) {
	T k = 0;
	switch(i) {
	case 0:
		k = Ag.k;
		break;
	case 2:
		k = inconel.k;
		break;
	case 3:
		k = MgO.k;
		break;
	case 4:
		k = NiCr.k;
		break;
	case 5:
		k = steel.k;
		break;
	default:
		cout << "Material (" << i << ") not found!" << endl;
	}
	return k;
}

T specHeat(int i) {
	T cp = 0;
	switch(i) {
	case 0:
		cp = Ag.cp;
		break;
	case 2:
		cp = inconel.cp;
		break;
	case 3:
		cp = MgO.cp;
		break;
	case 4:
		cp = NiCr.cp;
		break;
	case 5:
		cp = steel.cp;
		break;
	default:
		cout << "Material (" << i << ") not found!" << endl;
	}
	return cp;
}

T density(int i) {
	T rho = 0;
	switch(i) {
	case 0:
		rho = Ag.rho;
		break;
	case 2:
		rho = inconel.rho;
		break;
	case 3:
		rho = MgO.rho;
		break;
	case 4:
		rho = NiCr.rho;
		break;
	case 5:
		rho = steel.rho;
		break;
	default:
		cout << "Material (" << i << ") not found!" << endl;
	}
	return rho;
}

// initialize vector x
template <class T>
void initX(T *x,
		const int Nx,
		const int Ny)
{
	for (int i=0; i<Nx*Ny; i++)	{
		//int px  = i % Nx;
		//int py  = i / Nx;
		x[Nx+i] = 400.0;
		//if ((py-28)*(py-28) <= 4*4 - (px-19)*(px-19))    // patch circle
		//{
		//	x[Nx+i] = 200.0;
		//}
	}
}


// initialize symmetric matrix A
template <class T>
void initA( const T dt,
		const int Nx,
		const int Ny)
{
	// south
	for (int i=0; i<Nx*Ny-Nx; i++) {
		int px  = i % Nx;
		int py  = i / Nx;
		int idS = hm[i];
		int idC = hm[i+Nx];
		T kS = thermCond(idS);
		T kC = thermCond(idC);
		hss[Nx+i] = 2 * kS * kC / (dy[py+1]*kS + dy[py]*kC);
		hss[Nx+i] *= -dt * xc[px] * dx[px];
	}

	// west
	for (int i=0; i<Nx*Ny+1; i++) {
			if (i%Nx!=0) {
				int px  = i % Nx;
				int py  = i / Nx;
				int idW = hm[i-1];
				int idC = hm[i];
				T kW = thermCond(idW);
				T kC = thermCond(idC);
				hww[i] = 2 * kW * kC / (dx[px]*kW + dx[px-1]*kC);
				hww[i] *= -dt * dy[py] * (xc[px] - 0.5*dx[px]);
			}
		}

	// center
	for (int i=0; i<Nx*Ny; i++) {
			int px = i % Nx;
			int py = i / Nx;
			T V = xc[px] * dx[px] * dy[py];  // cell volume (1 rad)
			int id = hm[Nx+i];
			hV[Nx+i]  = density(id) * specHeat(id) * V;  // rho * cp * V
			//                            N            E           S         W
			hcc[Nx+i] = hV[Nx+i] - (hss[i+Nx] + hww[i+1] + hss[i] + hww[i]);
		}

	// Neumann BC at the bottom
	for (int i=0; i<Nx; i++) {
				hqB[i] = -dt * xc[i] * dx[i];
			}
}


// save data
template <class U>
void saveData(const U *x,
		const string name,
		const int Nx,
		const int Ny)
{
	ofstream File;
	stringstream fileName;
	fileName << name << ".vtk";
	File.open(fileName.str().c_str());
	File << "# vtk DataFile Version 3.0" << endl << "vtk output" << endl;
	File << "ASCII" << endl << "DATASET STRUCTURED_GRID" << endl;
	File << "DIMENSIONS " << Nx << " " << Ny << " 1" << endl;
	File << "POINTS " << Nx*Ny << " float" << endl;
	for (int i=0; i<Nx*Ny; ++i) {
		int px = i % Nx;
		int py = i / Nx;
		File << xc[px] << " " << yc[py] << " 0" << endl;
		}
	File << "POINT_DATA " << Nx*Ny << endl;
	File << "SCALARS " << name << " float" << endl;
	File << "LOOKUP_TABLE default" << endl;
	for (int i=0; i<Nx*Ny; ++i) {
		File << x[i+Nx] << endl;
	}
	File.close();
	cout << "saving VTK (" << fileName.str() << ")" << endl;
}

// save data in time
template <class T>
void saveDataInTime(const T *x,
		const T t,
		const string name,
		const int Nx,
		const int Ny)
{
	ofstream File;
	stringstream fileName;
	fileName << name << "-" << fixed << t << ".vtk";
	File.open(fileName.str().c_str());
	File << "# vtk DataFile Version 3.0" << endl << "vtk output" << endl;
	File << "ASCII" << endl << "DATASET STRUCTURED_GRID" << endl;
	File << "DIMENSIONS " << Nx << " " << Ny << " 1" << endl;
	File << "POINTS " << Nx*Ny << " float" << endl;
	for (int i=0; i<Nx*Ny; ++i) {
		int px = i % Nx;
		int py = i / Nx;
		File << xc[px] << " " << yc[py] << " 0" << endl;
		}
	File << "POINT_DATA " << Nx*Ny << endl;
	File << "SCALARS " << name << " float" << endl;
	File << "LOOKUP_TABLE default" << endl;
	for (int i=0; i<Nx*Ny; ++i) {
		File << x[i+Nx] << endl;
	}
	File.close();
	cout << "saving VTK (" << fileName.str() << ") at t = " << t << " sec." << endl;
}


// read geometry (material indices)
void readGeometry(int *x,
		const int Nx,
		const int Ny)
{
	cout << "Reading geometry from the file ..., ";
	int id = 0;
	ifstream File;
	string line;
	stringstream fileName;
	fileName << "geometry1024.txt";
	File.open(fileName.str().c_str());
	if (File.is_open()) {
		cout << "Geometry file exists." << endl;
		while ( getline(File,line) ) {
			stringstream stream(line);
			stream >> x[id+Nx];
			id++;
			//cout << id << endl;
		}
	}
	else {
		cout << "Geometry file NOT found!" << endl;
	}
	File.close();
}


// read X-coordinates
template <class T>
void readCoords(T *z,
		T *dz,
		const string name)
{
	cout << "Reading coordinates from the file ..., ";
	int id = 0;
	ifstream File;
	T pos, delta;
	stringstream fileName;
	fileName << name;
	File.open(fileName.str().c_str());
	if (File.is_open()) {
		cout << name << " file exists." << endl;
		while ( File >> pos >> delta ) {
			z[id]  = pos   * 0.001;
			dz[id] = delta * 0.001;
			//cout << id << ", " << z[id] << ", " << dz[id] << endl;
			id++;
		}
	}

	else {
		cout << name << " file NOT found!" << endl;
	}
	File.close();
}

// read Neumann BC (qTotal)
template <class T>
void readBC(T *x,
		const int Nt)
{
	cout << "Reading Neumann BC from the file ..., ";
	int id = 0;
	ifstream File;
	T HF;
	stringstream fileName;
	fileName << "qBC";     //"NeumannBC1024dt0000025";
	File.open(fileName.str().c_str());
	if (File.is_open()) {
		cout << "BC file exists." << endl;
		while ( File >> HF && id<Nt) {
			x[id] = HF;
			//cout << id << ", " << HF << endl;
			id++;
		}
	}
	else {
		cout << "qBC file NOT found!" << endl;
	}
	File.close();
}

// finalize dot product on cpu
T dot(const T *prtDot,
		const int blocks)
{
	T c = 0;
	for (int i=0; i<blocks; i++) {
		c += prtDot[i];
	}
	return c;
}


void check_var(const T *var,
		const int Nx,
		const int Ny)
{
	for (int i=0; i<Ny; i++) {
		//if (var[Nx+i]<0) cout << "smaller than zero!" << endl;
		cout << var[i+Nx] << endl;
	}
}

void check_varZ(const T *var1,
		const T *var2,
		const int nDV,
		const int NxZ)
{
	for (int i=0; i<nDV; i++) {
		//if (var[Nx+i]<0) cout << "smaller than zero!" << endl;
		//cout << var1[i+NxZ] << ",   " << var2[i]<< endl;
		cout << var1[i+NxZ] << endl;
	}
}

void printX(const T *x, const int NxZ,const int nDV)
{
	for (int i=0; i<nDV; i++)
		{
			cout << x[i+NxZ] << endl;
		}
}


#endif /* CPUFUNCTIONS_H_ */
