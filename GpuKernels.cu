#include "grouping.h"

#define workfactor 1
#define warpSize 32
__global__ void sgd(int *rowIndex, int *colIndex,float *w,float *x, float *y, const int rank, const float rate, const float lambda){
    int warp_index = threadIdx.x % 32; //thread index within a warp
    int warp_id = threadIdx.x / 32; //warp index within a block
    int index = blockIdx.x*blockDim.x/32 + warp_id; //global warp index

    int r_index = rowIndex[index]*rank + warp_index;
    int c_index = colIndex[index]*rank + warp_index;

    float w_x = x[r_index];
    float w_y = y[c_index];

    float _w = w[index];
    float val = w_x*w_y;

    //#pragma unroll
    for (int i = 16; i >= 1; i >>= 1) val += __shfl_xor(val,i);

    val = (val-_w);
    float grad_y = val*w_x + lambda*w_y;
    float grad_x = val*w_y + lambda*w_x;
    //if(index == 0) printf("%f \n",w_x);

    //atomicAdd(y+c_index,-rate*grad_y);
    //atomicAdd(x+r_index,-rate*grad_x);

    w_y = w_y-rate*grad_y;
    w_x = w_x-rate*grad_x;
    //if(w_y < 0.17) w_y += 0.1;
    //if(w_x < 0.17) w_x += 0.1;
    //if(w_y > 0.4) w_y *= 0.9;
    //if(w_x > 0.4) w_x *= 0.9;
    y[c_index] = w_y;
    x[r_index] = w_x;

    //if(index == 0) printf("%f %f %f\n",y[c_index],x[r_index],_w);
}

#define workfactor 1
#define warpSize 32
__global__ void sgd2(int *row, int *col,float *w,float *x, float *y, int rank, float rate, float lambda,float *J,float *x_rate, float *y_rate,float gama){

    int warp_index = threadIdx.x % 32; //thread index within a warp
    int warp_id = threadIdx.x / 32; //warp index within a block
    int index = blockIdx.x*blockDim.x/32 + warp_id; //global warp index

    int r_index = row[index]*rank + warp_index;
    int c_index = col[index]*rank + warp_index;

    //for(int i = index; i < index + workfactor*blockDim.x; i+= blockDim.x){
    #pragma unroll
    for(int j = 0; j < 1; j++){

      float w_x = x[r_index];
      float w_y = y[c_index];

      float rate_y = y_rate[c_index];
      float rate_x = x_rate[r_index];

      float _w = w[index];
      float val = w_x*w_y;

      #pragma unroll
      for (int i = 16; i >= 1; i >>= 1)
      val += __shfl_xor(val,i);

      val = (val-_w);
      float grad_y = val*w_x + lambda*w_y;
      float grad_x = val*w_y + lambda*w_x;

      rate_y = gama*rate_y + (1-gama)*grad_y*grad_y;
      rate_x = gama*rate_x + (1-gama)*grad_x*grad_x;

      //rate_y = rate_y + grad_y*grad_y;
      //rate_x = rate_x + grad_x*grad_x;

      y_rate[c_index] = rate_y;
      x_rate[r_index] = rate_x;

      //atomicAdd(y+c_index,- rate*grad_y/(0.000001f+sqrt(rate_y/(1-gama))));//-rate*grad_y);
      //atomicAdd(x+r_index,- rate*grad_x/(0.000001f+sqrt(rate_x/(1-gama))));//-rate*grad_x);
      rate_y = rate/(0.000001f+sqrt(rate_y/(1-gama)));
      rate_x = rate/(0.000001f+sqrt(rate_x/(1-gama)));
      //if(index == 1){
      //  printf("%f\n",rate_y);
      //}
      w_y = w_y - rate_y*grad_y;
      w_x = w_x - rate_x*grad_x;

      y[c_index] = w_y;
      x[r_index] = w_x; //x[r*rank+warp_index] = w_x;
    }


}
