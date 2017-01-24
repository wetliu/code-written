#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
//#include <ctime>
//#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;
#define htod cudaMemcpyHostToDevice
#define dtoh cudaMemcpyDeviceToHost
int recursiveReduce(int* tmp,int N);
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

inline double seconds(){
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
__global__ void reduceNeighbored(int *g_idata,int *g_odata, unsigned int n){
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx.x+ blockIdx.x*blockDim.x;

	int *idata = g_idata + blockIdx.x*blockDim.x;

	if(idx >= n) return;
	//	if(blockIdx.x > 0 && blockIdx.x < 4 && tid == 0){
	//         printf("%d\n",g_odata[blockIdx.x]);
	//	}


	for(int stride = 1; stride < blockDim.x; stride*= 2){
		if((tid%(2*stride))== 0 && blockIdx.x*blockDim.x+tid+stride < n){
			idata[tid] += idata[tid+stride];
		}
		__syncthreads();
	}
	if(tid == 0) g_odata[blockIdx.x] = idata[0];
	//if(tid == 0){
          printf("block: %d,  %d\n",blockIdx.x,g_odata[blockIdx.x]);
    //}

	//if(blockIdx.x == 5)
	//	printf("%d\n",g_idata[idx]);
	return;

}
__global__ void reduceNeighboredLess (int *g_idata, int *g_odata,
		                                      unsigned int n) //neighbor-paired with less divergence
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x*blockDim.x + tid;

	int *idata = blockIdx.x*blockDim.x+g_idata;

	//if( idx > 9999990 && idx < 10000001){
	//	printf("%d\n",idata[tid]);
	//}

	if(idx >= n) return;
	//#pragma unroll
	for(int stride = 1; stride < blockDim.x; stride *= 2){
		int index = 2*stride*tid;
		if(index < blockDim.x && blockIdx.x*blockDim.x+index + stride< n){
			idata[index] += idata[index+stride];
		}
		__syncthreads();
	}

	if(tid == 0) g_odata[blockIdx.x] = idata[0];

	    //
}
__global__ void reduceInterleaved (int *g_idata, int *g_odata, unsigned int n){
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x*blockDim.x;
	
	int *idata = g_idata+blockIdx.x*blockDim.x;

	if(idx+tid >= n) return;

	for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
		if(tid < stride && idx+tid + stride < n){
			idata[tid] += idata[tid+stride];
		}
		__syncthreads();
	}
	if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrolling2 (int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x*2+tid;

    int *idata = g_idata+blockIdx.x*blockDim.x*2;

    if(idx + blockDim.x < n) g_idata[idx] += g_idata[idx+blockDim.x];
	__syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < stride && idx + stride < n){
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}
__inline__ __device__ int warpreduce(int mysum){
	mysum+=__shfl_xor(mysum,16);	
	mysum+=__shfl_xor(mysum,8);
	mysum+=__shfl_xor(mysum,4);
	mysum+=__shfl_xor(mysum,2);
	mysum+=__shfl_xor(mysum,1);
	return mysum;
	//printf("%d\n",mysum);
}
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnrolling(int *g_idata, int *g_odata, unsigned int n){
	
	// set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

	__shared__ int sem[iBlockSize];

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // urolling 8
	int a1 = 0, a2 = 0, a3 = 0, a4= 0, b1 = 0,b2= 0, b3 = 0, b4= 0;
    if (idx < n) 					a1 = g_idata[idx];
	if (idx + blockDim.x < n) 		a2 = g_idata[idx + blockDim.x];
    if (idx + 2 * blockDim.x < n) 	a3 = g_idata[idx + 2 * blockDim.x];
	if (idx + 3 * blockDim.x < n)   a4 = g_idata[idx + 3 * blockDim.x];
	if (idx + 4 * blockDim.x < n)   b1 = g_idata[idx + 4 * blockDim.x];
	if (idx + 5 * blockDim.x < n)   b2 = g_idata[idx + 5 * blockDim.x];
	if (idx + 6 * blockDim.x < n)   b3 = g_idata[idx + 6 * blockDim.x];
	if (idx + 7 * blockDim.x < n)   b4 = g_idata[idx + 7 * blockDim.x];

    sem[tid] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;

    __syncthreads();
    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512 && idx+512 < n) sem[tid] += sem[tid + 512];

    __syncthreads();

    if (iBlockSize >= 512 && tid < 256 && idx+256 < n)  sem[tid] += sem[tid + 256];

   __syncthreads();

    if (iBlockSize >= 256 && tid < 128 && idx+128 < n)  sem[tid] += sem[tid + 128];

    __syncthreads();

    if (iBlockSize >= 128 && tid < 64 && idx+64 < n)   sem[tid] += sem[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid >= 32) return;
	if (iBlockSize >= 64 && tid < 32 && idx+32 < n) sem[tid] += sem[tid+32];     
	/*
	volatile int *vsmem = idata; 
	if(idx+32 < n)	 vsmem[tid] += vsmem[tid + 32];
    if(idx+16 < n)   vsmem[tid] += vsmem[tid + 16];
    if(idx+8 < n)    vsmem[tid] += vsmem[tid +  8];
    if(idx+4 < n) 	 vsmem[tid] += vsmem[tid +  4];
    if(idx+2 < n)    vsmem[tid] += vsmem[tid +  2];
    if(idx+1 < n)    vsmem[tid] += vsmem[tid +  1];
	*/
	
	int mysum = 0;
	if(idx < n) mysum = sem[tid];
	//if(idx+32 < n) mysum += sem[tid+32];
	mysum = warpreduce(mysum);
	
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mysum; //warpreduce(mysum);	
	return;	
}
#define DIM 512
#define SMEMDIM 4
/*
__global__ void reduceShfl(int *g_idata, int *g_odata, unsigned int n){
	__shared__ int smem[SMEMDIM];

	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx>=n) return;

	int mySum = g_idata[idx];
	int laneIdx = threadIdx.x%warpSize;
	int warpIdx = threadIdx.x%warpSize;

	mySum = warpreduce(mySum);

	if(laneIdx==0) smem[warpIdx] = mySum;
	__syncthreads();

	mySum = (threadIdx.x < SMEMDIM) ? smem[laneIdx]:0;

	if(warpIdx==0) mySum = warpreduce(mySum);
	if(threadIdx.x ==0) g_odata[blockIdx.x] = mySum;

}
*/
//template<unsigned int >
__global__ void reduceShfl(int *g_idata, int *g_odata,
                                     unsigned int n)
{
    // static shared memory
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;

    // global index
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // unrolling 4 blocks
    int localSum = 0;

    if (idx + 3 * blockDim.x < n)
    {
        float a1 = g_idata[idx];
        float a2 = g_idata[idx + blockDim.x];
        float a3 = g_idata[idx + 2 * blockDim.x];
        float a4 = g_idata[idx + 3 * blockDim.x];
        localSum = a1 + a2 + a3 + a4;
    }

    smem[tid] = localSum;
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];
    __syncthreads();
    if (blockDim.x >= 64 && tid < 32) smem[tid] += smem[tid + 32];
    __syncthreads();

    // unrolling warp
    localSum = smem[tid];
    if (tid < 32)
    {
        localSum += __shfl_xor(localSum, 16);
        localSum += __shfl_xor(localSum, 8);
        localSum += __shfl_xor(localSum, 4);
        localSum += __shfl_xor(localSum, 2);
        localSum += __shfl_xor(localSum, 1);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = localSum;
}
int main(int argc, char**argv){

	/*set device */
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp,dev);
	printf("%s starting reduction at ",argv[0]); //the executable of this file
	printf("device %d: %s ",dev, deviceProp.name); // the device name
	cudaSetDevice(dev); //set device

	bool bResult = false;
	int size = 1<<24; //1000000;
	if(argc > 1){
		size = atoi(argv[1]);
	}
	printf("with array size %d ", size);

	//execution config
	int blocksize = 512;
	if(argc > 2){
		blocksize = atoi(argv[2]); //blocksize from command line argu
	}
	dim3 block (blocksize,1);
	dim3 grid  ((size+block.x-1)/block.x);
	printf("grid %d block %d\n",grid.x, block.x);

	//memory allocation
	size_t bytes = size * sizeof(int);
	int *h_idata = (int *) malloc(bytes);
	int *h_odata = (int *) malloc(grid.x*sizeof(int));
	int *tmp 	 = (int *) malloc(bytes);

	//data initialization
	for(int i = 0; i < size; i++){
		h_idata[i]= 1;//(int) (rand() & 0xFF);
	}
	memcpy (tmp, h_idata, bytes);

	double iStart, iElaps;
	//int gpu_sum = 0;

	//allocate device memory
	int *d_idata = NULL;
	int *d_odata = NULL;
	CHECK(cudaMalloc((void **) &d_idata, bytes));
	CHECK(cudaMalloc((void **) &d_odata, grid.x*sizeof(int)));

	for(int i = 0; i < grid.x; i++) h_odata[i] = 0;
	CHECK(cudaMemcpy(d_odata, h_odata, grid.x * sizeof(int),cudaMemcpyHostToDevice));

	iStart = seconds ();
	int cpu_sum = recursiveReduce(tmp,size);
	iElaps = seconds() - iStart;
	printf("cpu reduce 	 elapsed %f ms cpu_sum: %d\n", iElaps, cpu_sum);

	//kernel 1: reduceNeighboared
	CHECK(cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = seconds();
  	reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
	CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),cudaMemcpyDeviceToHost));
	int gpu_sum1 = 0;

	for (int i = 0; i < grid.x; i++) gpu_sum1 += h_odata[i];

	printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
										           "%d>>>\n", iElaps, gpu_sum1, grid.x, block.x);

	for(int i = 0; i < grid.x; i++) h_odata[i] = 0;
	CHECK(cudaMemcpy(d_odata, h_odata, grid.x * sizeof(int),cudaMemcpyHostToDevice));


	// kernel 2: reduceNeighbored with less divergence
	CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = seconds();
	reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),cudaMemcpyDeviceToHost));
	int gpu_sum2 = 0;

	for (int i = 0; i < grid.x; i++) gpu_sum2 += h_odata[i];

	printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum2, grid.x, block.x);

	for(int i = 0; i < grid.x; i++) h_odata[i] = 0;
	cudaMemcpy(d_odata, h_odata, grid.x * sizeof(int),cudaMemcpyHostToDevice);

	//kernel 3: reduce with interleave
	CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size); ////
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;

    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    int gpu_sum3 = 0; ////

    for (int i = 0; i < grid.x; i++) gpu_sum3 += h_odata[i];

    printf("gpu Interleave elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum3, grid.x, block.x);
	
	for(int i = 0; i < grid.x; i++) h_odata[i] = 0;
	cudaMemcpy(d_odata, h_odata, grid.x * sizeof(int),cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

	//kernel 4: reduce with unrolling 2
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceUnrolling2<<<(grid.x+1)/2, block>>>(d_idata, d_odata, size); ////
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    int gpu_sum4 = 0; ////

    for (int i = 0; i < (grid.x+1)/2; i++) gpu_sum4 += h_odata[i];

    printf("gpu Unrolling2 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum4, (grid.x+1)/2, block.x);
	
	for(int i = 0; i < grid.x; i++) h_odata[i] = 0;
	cudaMemcpy(d_odata, h_odata, grid.x * sizeof(int),cudaMemcpyHostToDevice);

	//kernel 5: reduce with complete unrolling
	cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceCompleteUnrolling<512><<<(grid.x+7)/8, block>>>(d_idata, d_odata, size); ////
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, (grid.x+8) * sizeof(int)/8,cudaMemcpyDeviceToHost);
    int gpu_sum5 = 0; ////

    for (int i = 0; i < (grid.x+7)/8; i++) gpu_sum5 += h_odata[i];

    printf("gpu complete unrolling  elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum5, (grid.x+7)/8, block.x);
	
	for(int i = 0; i < grid.x; i++) h_odata[i] = 0;
	cudaMemcpy(d_odata, h_odata, grid.x * sizeof(int),cudaMemcpyHostToDevice);

	//kernel 6: reduce by shuffle
	cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceShfl<<<(grid.x+3)/4, block>>>(d_idata, d_odata, size); ////
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    int gpu_sum6 = 0; ////

    for (int i = 0; i < (3+grid.x)/4; i++) gpu_sum6 += h_odata[i];

    printf("gpu reduceshuffle elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum6, (grid.x+3)/4, block.x);
	
	for(int i = 0; i < (3+grid.x)/4; i++) h_odata[i] = 0;
	cudaMemcpy(d_odata, h_odata, grid.x * sizeof(int),cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
	
	// free host memory
	free(h_idata);
	free(h_odata);

			    // free device memory
	CHECK(cudaFree(d_idata));
	CHECK(cudaFree(d_odata));

					    // reset device
	cudaDeviceReset();

						    // check the results
	bResult = (gpu_sum1 == cpu_sum);

	if(!bResult) printf("Test failed!\n");

	return 0;
}

 int recursiveReduce(int* tmp,int N){
	int sum = 0;
	for(int i = 0; i < N; i++){
		sum+=tmp[i];
	}
	return sum;
 }
/*
template<T>
void cudamemory(T **drain, T *source, int size, bool alloc, bool htd, bool dth){
	if(alloc)
	cudaMaloc((void **) drain, size);
	if(transfer){
		cudaMemcpy(a,source,bytes,htod);
	}
	return a;

}*/
