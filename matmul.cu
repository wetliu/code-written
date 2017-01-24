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
double istart, iend;												

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
  
// Thread block size  
#define BLOCK_SIZE 32 
int gRow = 0, gCol = 0;  
// Forward declaration of the matrix multiplication kernel  
__global__ void MatMulKernel1(int *A, int *B, int *C, int m, int n, int p); 
__global__ void MatMulKernel2(int *A, int *B, int *C, int m, int n, int p);
// Matrix multiplication - Host code  
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE  

void matMulCPU(const int *A, const int *B, int *C, int m, int n, int p){
		istart = seconds();
		for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < p; k++)
								C[i*n+j] += A[i*p+k] * B[j*p+k];
		iend = seconds() - istart;
		cout << "CPU time is " << iend << endl;
}
 
void MatMul(const int *A, const int * B, int *C, int m, int n, int p)  
{  
    // Load A and B to device memory  
    int* d_A;
    size_t aSize = m * p * sizeof(int);
    cudaMalloc(&d_A, aSize);
    cudaMemcpy(d_A, A, aSize,
    cudaMemcpyHostToDevice);  
    
		int* d_B;
    size_t bSize = n * p * sizeof(int);
    cudaMalloc(&d_B, bSize);
    cudaMemcpy(d_B, B, bSize,
    cudaMemcpyHostToDevice);

    // Allocate C in device memory  
    int *d_C;
    size_t cSize = m * n * sizeof(int);
    cudaMalloc(&d_C, cSize);
  
    // Invoke kernel  
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n+dimBlock.x-1) / dimBlock.x, (m+dimBlock.y-1) / dimBlock.y);
	
	
	  istart = seconds();
    MatMulKernel2<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, p);  
		cudaDeviceSynchronize();
		iend = seconds() - istart;
		
		cout << "GPU time is " << iend << endl;
		int *C_gpu = new int[m*n];
    // Read C from device memory  
    CHECK(cudaMemcpy(C_gpu, d_C, cSize,  
    cudaMemcpyDeviceToHost));  
		
		cout << "checking GPU vs CPU result:  ";

		
		bool set = 0;
		for(int i = 0; i < m*n; i++){
				if(C[i] != C_gpu[i]){
						cout << "Error: " << C[i] << ' ' << C_gpu[i] << ' ' << i << endl;
						set = 1;
						break;
				}
		}
		if(set == 0)
		cout << "checking done. nothing weird found."  << endl;
    /*
		for(int i = 0; i < m; i++){
				for(int j = 0; j < n; j++){
						cout << C_gpu[i*n + j] << ' ';
				}
				cout << endl;
		}
		*/
		// Free device memory  
    cudaFree(d_A);  
    cudaFree(d_B);  
    cudaFree(d_C);  
}  

int main(int argc, char *argv[]){
  

  srand(time(NULL));
  int *A,*B,*C;
	
	int m = 32; 
	int n = 32;
	int p = atoi(argv[1]);

	cout << "Very early set up is done" << endl;
	A = new int[m*p];
	B = new int[n*p];
	C = new int[m*n];
	for(int i = 0; i < m*p; i++) A[i] = rand() & 0xFF;
	for(int i = 0; i < n*p; i++) B[i] = rand() & 0xFF;
	for(int i = 0; i < m*n; i++) C[i] = 0;
	
	cout << "A, B, C are set, Sample input are A: " << A[0] << ' ' << A[1] << " B: " << B[0] << ' ' << B[1] << endl;
	cout << "Done: with host data initilization" << endl;

	matMulCPU(A,B,C,m,n,p);
	MatMul(A, B, C, m, n, p);
		/*
	for(int i = 0; i < m; i++){
			for(int j = 0; j < p; j++){
				cout << A[i*p+j] << ' ';
			}
			cout << endl;
	}
  cout << endl;
	for(int i = 0; i < n; i++){
			for(int j = 0; j < p; j++){
				cout << B[i*p+j] << ' ';
			}
			cout << endl;
	}
		cout << endl;
	for(int i = 0; i < m; i++){
			for(int j = 0; j < n; j++){
				cout << C[i*n+j] << ' ';
			}
			cout << endl;
	}
*/
	cout << endl;
	return 0;

}  
// Matrix multiplication kernel called by MatMul()  
__global__ void MatMulKernel1(int *A, int *B, int *C, int m, int n, int p)  
{  
    // Each thread computes one element of C  
    // by accumulating results into Cvalue  
    int Cvalue = 0; 
    
		int row = blockIdx.y * blockDim.y + threadIdx.y;  
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
//		if(row*col == 1024)	printf("%d %d %d \n", row, col);


		if(row <  m && col < n){
		for (int e = 0; e < p; ++e)  
				Cvalue += A[row * p + e]* B[col * p + e]; 
				C[row * n + col] = Cvalue;
		}

		//if(row*col == 1024)	printf("%d %d %d \n", row, col, Cvalue);
}

__global__ void MatMulKernel2(int *A, int *B, int *C, int m, int n, int p){
		__shared__ int A_tile[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ int B_tile[BLOCK_SIZE][BLOCK_SIZE];

		int accu = 0;
		
		unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	  

		for(int tileIdx = 0; tileIdx < (blockDim.x+p-1)/blockDim.x;tileIdx++){

				unsigned int tileIndex = tileIdx * BLOCK_SIZE+threadIdx.x;
				
				if(tileIndex < n && row < m)
				A_tile[threadIdx.y][threadIdx.x] = A[row*p+tileIndex];
				if(tileIndex < m && col < n)
				B_tile[threadIdx.y][threadIdx.x] = B[row*p+tileIndex];
				
				__syncthreads();
				
				//if(row < m && col < n)
				//printf("row: %d, col: %d, Atile, %d, Btile, %d \n", row, col, A_tile[33],B_tile[33]);
				//printf("%d %d \n",A_tile[gRow],B_tile[gCol]);
				
				if(row < m && col < n)
						for(int k = 0; k < min(blockDim.x,p-blockDim.x*tileIdx); k++){
								//if(row == 1 && col == 1) printf("tiles: %d %d \n", A_tile[threadIdx.y*blockDim.x+k],B_tile[threadIdx.x*blockDim.y+k]);
						accu += A_tile[threadIdx.y][k]*B_tile[threadIdx.x][k];
				}
				
				__syncthreads();
		}
		
		if(row < m && col < n)
		C[row*n+col] = accu;
}

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif
 
__global__ void matMulKernel2(float *A, float *B, float *C, int wA, int wB){  
	/*
	__shared__ float A_tile[blockDim.y][blockDim.x];
	__shared__ float B_tile[blockDim.x][blockDim.y];
	*/
	// Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {


        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + wA * ty + tx];
        BS(tx, ty) = B[b + wB * tx + ty];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;	


}

