#include "grouping.h"

#define CUDA_SAFE_CALL_NO_SYNC( call) do {                              \
  cudaError err = call;                                                 \
  if( cudaSuccess != err) {                                             \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString( err) );         \
    exit(EXIT_FAILURE);                                                 \
    } } while (0)

#define CUDA_SAFE_CALL( call) do {                                      \
  CUDA_SAFE_CALL_NO_SYNC(call);                                         \
  cudaError err = cudaThreadSynchronize();                              \
  if( cudaSuccess != err) {                                             \
     fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
     exit(EXIT_FAILURE);                                                \
     } } while (0)

using namespace std;

template<typename T>
void cuda_alloc(T * *destination,int num,unsigned  int &mem_size,bool memcp,T *source);
__global__ void print(float *y);
__global__ void sgd(int *row, int *col,float *w,float *x, float *y, int rank, float rate, float lambda,float *J,float *x_rate, float *y_rate,float gama);
__global__ void sgd2(int *row, int *col,float *w,float *x, float *y, int rank, float rate, float lambda,float *J,float *x_rate, float *y_rate,float gama);

#define htod cudaMemcpyHostToDevice
#define dtoh cudaMemcpyDeviceToHost
#define random(x) (rand()%x)

inline double cpuSecond(){
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp,&tzp);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char **argv){
	srand((int)time(0));

    vector<int> row, col;
    vector<float> w;
    vector<vector<int> > test;
    int m, n, instances;
    read(row,col,w,m,n,instances,test);
    cout << "There are total " << instances << " instances" << endl;

    /************grouping part*****not necessary*********/
    vector<int> dummy1;
    vector<float> dummy2;
    vector<vector<int> > row_g(max(m,n),dummy1), col_g(max(m,n),dummy1);
    vector<vector<float> > w_g(max(m,n),dummy2);
    grouping(row_g,col_g,w_g,m,n,instances,row,col,w);


    /*****************begin of array-ordering*****************************/
    int h_row[800000];
    float host_w[800000];
    int h_col[800000];
    cout << "So far so good" << endl;

    float *h_w = host_w;
    array_ordering(instances,row,col,w,h_row,h_col,h_w);


    /**************begin host parameter setup***************/
    int rank = 32; //parameter that should change
    float x[rank*m];
    float y[rank*n];
    float *dummy3;
    float x_rate[rank*m];
    float y_rate[rank*n];

    for(int i=0;i<rank*m-1;i++){
        x[i] = (float)random(100)/100.0f;
        x_rate[i] = 0.0f;
    }
    for(int i=0;i<rank*n-1;i++){
        y[i] = (float)random(100)/100.0f;
        y_rate[i] = 0.0f;
    }
    int outputResult[100];

    /*****************begin of GPU data transfer**********************/
    int *d_row, *d_col;
    unsigned int mem_row, mem_col, mem_w, mem_x,mem_y,mem_J;
    float *d_w, *d_x, *d_y,*d_J,*d_x_rate,*d_y_rate;
    cuda_alloc<int>(&d_row, instances, mem_row, true, h_row);
    cuda_alloc<int>(&d_col, instances, mem_col, true, h_col);
    cuda_alloc<float>(&d_w, instances, mem_w, true, h_w);

    cuda_alloc<float>(&d_y, rank*n, mem_y, true, y);

    cuda_alloc<float>(&d_x, rank*m, mem_x, true, x);
    cuda_alloc<float>(&d_J, instances, mem_J, false, x);

    cuda_alloc<float>(&d_x_rate, rank*m, mem_x, true, x_rate);
    cuda_alloc<float>(&d_y_rate, rank*n, mem_y, true, y_rate);

    /*********GPU kernel call******/
    int threadperblock = 1024;
    int workfactor = 1;
    dim3 threads(threadperblock,1,1);
    dim3 blocks((instances*32+threadperblock-1)/(workfactor*threadperblock),1,1);
    int l = 0;
    float gama = 1.0f;

    double t = cpuSecond();
    for(int p = 0; p < 1000; p++){   //# of iterations
      double sum = 0;
      for(int i = 0; i < test.size(); i++){
        int r = test[i][0];
        int c = test[i][1];
        float tmp = 0;
        for(int j = 0; j < rank; j++){
          tmp += x[r*rank+j]*y[c*rank+j];
        }
        int tmpNum = floor(tmp+0.5);
        sum += (tmpNum - test[i][2])*(tmpNum - test[i][2]);

        if(p == 999 && l < 100) outputResult[l++] = tmpNum;
      }


      float rate = 0.001f;
      float lambda = 0.1f;

      cout << sum/test.size() << endl;
      gama = 0.999f;
      sgd<<<blocks,threads>>>(d_row, d_col,d_w,d_x, d_y, rank, rate, lambda);
      //sgd2<<<blocks,threads>>>(d_row, d_col,d_w,d_x, d_y, rank, rate, lambda,d_J,d_x_rate,d_y_rate,gama);
      cudaDeviceSynchronize();
      CUDA_SAFE_CALL(cudaMemcpy(y, d_y, mem_y, dtoh));
      CUDA_SAFE_CALL(cudaMemcpy(x, d_x, mem_x, dtoh));
    }
    cout << "Time spent on the gpu is " << cpuSecond()-t << endl;
    CUDA_SAFE_CALL(cudaMalloc( (void**) &dummy3, 1));

    CUDA_SAFE_CALL(cudaFree(d_w));
    CUDA_SAFE_CALL(cudaFree(d_x));
    CUDA_SAFE_CALL(cudaFree(d_y));
    CUDA_SAFE_CALL(cudaFree(d_row));
    CUDA_SAFE_CALL(cudaFree(d_col));
    for(int j = 0; j < 100; j++) cout << outputResult[j] << endl;

    return 0;
}




template<typename T>
void cuda_alloc(T * *destination,int num,unsigned  int &mem_size,bool memcp,T *source){
  mem_size = num*sizeof(T);
  cudaMalloc( (void**) destination, mem_size);
  if(memcp)
    cudaMemcpy(*destination, source, mem_size, htod);
}
