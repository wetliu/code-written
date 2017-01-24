#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
using namespace std;

 #define CUDA_SAFE_CALL_NO_SYNC( call) do {                              \
   cudaError err = call;                                                 \
   if( cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                 __FILE__, __LINE__, cudaGetErrorString( err) );         \
     exit(EXIT_FAILURE);                                                 \
     } } while (0)

 #define CHECK( call) do {                                      \
   CUDA_SAFE_CALL_NO_SYNC(call);                                         \
   cudaError err = cudaThreadSynchronize();                              \
   if( cudaSuccess != err) {                                             \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                  __FILE__, __LINE__, cudaGetErrorString( err) );        \
      exit(EXIT_FAILURE);                                                \
      } } while (0)

#define htod cudaMemcpyHostToDevice
#define dtoh cudaMemcpyDeviceToHost

void scanCPU(float *A, float *B, int size){
	B[0] = 0;
	for(int i = 1; i < size; i ++){
		B[i] = A[i-1] + B[i-1];
	}
}

void initializeData(float *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float) (rand()%512);
    }

    return;
}

inline double cpuSecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void scanGPU1(float *g_idata, float *g_odata,int n){
	//	extern __shared__ float temp[];
	extern __shared__ float temp[];

	int thid = threadIdx.x;
	int pout = 0, pin = 1;
	float *idata = blockDim.x*blockIdx.x+g_idata;

	temp[thid] = (thid == 0) ? 0:idata[thid-1];
	__syncthreads();


	for (int offset = 1; offset < blockDim.x; offset = offset*2){
		  pout = 1 - pout; // swap double buffer indices
  		pin = 1 - pout;
  		if (thid >= offset)
			   temp[pout*blockDim.x+thid] = temp[pin*blockDim.x+thid] + temp[pin*blockDim.x + thid - offset];
  		else
    		 temp[pout*blockDim.x+thid] = temp[pin*blockDim.x+thid];
		__syncthreads();
	}

	//printf("%f %d\n",idata[thid], thid);
	g_odata[thid] = temp[pout*blockDim.x+thid];
}


__global__ void scanGPU2(float *g_idata, float *g_odata, int n){

	extern __shared__ float temp[];// allocated on invocation

	int thid = threadIdx.x;
	int offset = 1;

	//temp[2*thid] = g_idata[2*thid]; // load input into shared memory
	//temp[2*thid+1] = g_idata[2*thid+1];

  temp[thid] = g_idata[thid];		
	__syncthreads();
	for (int d = n>>1; d > 0; d >>= 1){// build sum in place up the tre
		if (thid < d){
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			temp[bi] += temp[ai];
		}
		offset <<= 1; //multiply by 2 implemented as bitwise operation
		__syncthreads();
	}

	if (thid == 0) { temp[n - 1] = 0; } // clear the last element
  offset >>= 1;
	//__syncthreads();
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		if (thid < d){
		int ai = offset*(2*thid+1)-1;
		int bi = offset*(2*thid+2)-1;

		float t = temp[ai];
		temp[ai] = temp[bi];
		temp[bi] += t;
		}

		offset >>= 1;
		__syncthreads();
	}

	g_odata[thid] = temp[thid];
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
    float epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
	        match = 0;
            printf("host %f gpu %f at element %d \n", hostRef[i], gpuRef[i],i);
				}

            printf("host %f gpu %f at element %d \n", hostRef[i], gpuRef[i],i);
	}

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

int main(int argc, char **argv){
		srand(time(NULL));
		int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp,dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	//malloc on the device side

	int size;
	if(argc > 2) size = atoi(argv[1]);//1<<9; //1<<24;
	else size = 1024; //MAtrix size
	int nBytes = size*sizeof(float);
	cout << "matrix size is " << size << endl;

	//malloc on host side
	float *h_A, *cpuRef, *gpuRef;
	h_A = new float[size];
	cpuRef = new float[size];
	gpuRef = new float[size];

	//initialize the data at host side
	double iStart = cpuSecond();
	initializeData(h_A,size);
	double iElaps = cpuSecond() - iStart;

	memset(cpuRef,0,nBytes);
	memset(gpuRef,0,nBytes);

	iStart = cpuSecond();
	scanCPU(h_A,cpuRef,size);
	iElaps = cpuSecond() - iStart;
	cout << "cpu time is " << iElaps << endl;

	//malloc on device side
	float *d_A, *d_C;
	CHECK(cudaMalloc((void **) &d_A,nBytes));
	CHECK(cudaMalloc((void **) &d_C,nBytes));

	CHECK(cudaMemcpy(d_A,h_A,nBytes,htod));

	int input = 1024;
	if(argc > 3) input = atoi(argv[2]); //block size
	dim3 block(input,1,1); 
	dim3 grid((block.x+size-1)/block.x);

	//kernel
	cout << "kernal block size is " << block.x << endl;
	cout << "kernal grid size is " << grid.x << endl;
  cout << "scan alogrithm you choose: (0, 1)" << endl;
  int choice = 2;
		
	cout << "sample is" <<  h_A[0] <<' ' << h_A[1] << endl;
  if(choice == 1){
	   iStart = cpuSecond();
	   scanGPU1<<<grid, block.x, block.x*2*sizeof(float)>>>(d_A,d_C,size);
	   cudaDeviceSynchronize();
	   iElaps = cpuSecond() - iStart;
   }
   else if(choice == 2){
			 cout << "You choose option 2" << endl;
 	   iStart = cpuSecond();
 	   scanGPU2<<<grid, block.x, block.x*sizeof(float)>>>(d_A,d_C,size);
	   cudaDeviceSynchronize();	
	   iElaps = cpuSecond() - iStart;
   }
  else return 1;

  cout << "GPU time is " << iElaps << endl;

	CHECK(cudaMemcpy(gpuRef,d_C,nBytes,dtoh));
	checkResult(cpuRef, gpuRef, size);

	
  CHECK(cudaFree(d_A));
  CHECK(cudaFree(d_C));
  delete [] cpuRef;
  delete [] gpuRef;
  
 cudaDeviceReset(); 
  return 0;
}
