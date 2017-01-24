#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define WSIZE 1024
#define LOOPS 1
#define UPPER_BIT 20
#define LOWER_BIT 0
#define htod cudaMemcpyHostToDevice
#define dtoh cudaMemcpyDeviceToHost

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
\

inline double cpuSecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

//__device__ unsigned int tmp0[WSIZE];
//__device__ unsigned int tmp1[WSIZE];
//__device__ unsigned int scanResult0[WSIZE];
//__device__ unsigned int scanResult1[WSIZE];

// naive warp-level bitwise radix sort
__device__ void scanGPU2(unsigned int *g_idata, unsigned int *g_odata, int n);

__device__ void scanGPU(unsigned int *g_idata, unsigned int *g_odata,int n){
		//	extern __shared__ float temp[];
		__shared__ unsigned int temp[WSIZE*2];

		int thid = threadIdx.x;
		int pout = 0, pin = 1;
		unsigned int *idata = blockDim.x*blockIdx.x+g_idata;

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

		g_odata[thid] = temp[pout*blockDim.x+thid];
}

__global__ void myKernel(unsigned int *ddata, unsigned int *dodata, int N){
		__shared__ unsigned int sdata[WSIZE*2];
		__shared__ unsigned int carry;
		__shared__ unsigned int tmp0[WSIZE];
		__shared__ unsigned int tmp1[WSIZE];
		__shared__ unsigned int scanResult0[WSIZE];
		__shared__ unsigned int scanResult1[WSIZE];

		// load from global into shared variable
		sdata[threadIdx.x] = ddata[threadIdx.x];
		__syncthreads(); //not very likely here

		unsigned int out = 0;
		unsigned int in	 = 1;
		unsigned int offset = blockDim.x;

		unsigned int value;

		unsigned int pos = 0;
		while(pos < UPPER_BIT){
				value = sdata[threadIdx.x+offset*out];
	
				unsigned int tmp = (value >> pos) & 1;

				tmp1[threadIdx.x] = tmp;
				tmp0[threadIdx.x] = 1-tmp;
				__syncthreads();
				scanGPU2(tmp0,scanResult0,WSIZE);
				scanGPU2(tmp1,scanResult1,WSIZE);
				__syncthreads(); // make sure the scanResults0 is correct

				if(threadIdx.x == blockDim.x-1) {
						carry = scanResult0[blockDim.x-1];
						if(tmp == 0)
						carry++;
				}
				__syncthreads();

				scanResult0[threadIdx.x] = (tmp==0) ? (scanResult0[threadIdx.x]):(scanResult1[threadIdx.x]+carry);		

				__syncthreads();
				sdata[scanResult0[threadIdx.x]+offset*in] = value;
				
				in = 1 - in;
				out = 1 - in;
				pos++;
				__syncthreads();
		}
				dodata[threadIdx.x] = sdata[threadIdx.x+offset*out];
				//printf("%d \n",dodata[threadIdx.x]);
  }




int main(){
		
		srand(time(NULL));
	//setup
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp,dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	//cpu setup
	unsigned int hInData[WSIZE];
	unsigned int hOutData[WSIZE];
	unsigned int range = 1U << UPPER_BIT;
	for(int i = 0; i < WSIZE; i++){
		hInData[i] = rand()%range;
	}

	printf("Sample Input are %d, %d \n", hInData[0],hInData[1]);

	int nBytes = WSIZE*sizeof(unsigned int);

	double iStart, iElaps;

	printf("Done:	cpu setup done \n");

	//gpu setup
	unsigned int *dInData;
	unsigned int *dOutData;

	// malloc space for the input output
	CHECK(cudaMalloc((void**) &dInData, nBytes));
	CHECK(cudaMalloc((void**) &dOutData, nBytes));

	//memcpy
	CHECK(cudaMemcpy(dInData,hInData,nBytes,htod));

	printf("Done:	gpu setup done \n");

	//kernel launch
	iStart = cpuSecond();
	myKernel<<<1,WSIZE>>>(dInData,dOutData,WSIZE);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;

	//copy back
	CHECK(cudaMemcpy(hOutData,dOutData,nBytes,dtoh));

	printf("Done:	cuda copy back done, time for execution is %f. \n",iElaps);


	//result compare
	for (int i = 0; i < WSIZE-1; i++){
		if (hOutData[i] > hOutData[i+1])
			printf("error: hdata[%d] = %d, hdata[%d] = %d\n",i, hOutData[i],i+1, hOutData[i+1]);
		if(i == WSIZE-2) printf("Done: checking\n");	
		//printf("%d \n",hOutData[i]);
	}


	printf("Everything is done\n");

	//delete[] hInData;
	//delete[] hOutData;

	CHECK(cudaFree(dInData));
	CHECK(cudaFree(dOutData));

	//CHECK(cudaFree(tmp0));
	//CHECK(cudaFree(tmp1));

	//CHECK(cudaFree(scanResult0));
	//CHECK(cudaFree(scanResult1));

	return 0;
}


__device__ void scanGPU2(unsigned int *g_idata, unsigned int *g_odata, int n){

		__shared__ unsigned int temp[WSIZE];// allocated on invocation

		int thid = threadIdx.x;
		int offset = 1;
		//if(thid == 0) printf("We are scan Kernel 2 :)");

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
