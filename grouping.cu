#include "grouping.h"
#define htod cudaMemcpyHostToDevice
#define dtoh cudaMemcpyDeviceToHost

void array_ordering(int instances, vector<int> &row,vector<int> &col,vector<float> &w,int *h_row,int* h_col, float *h_w){
  for(int i = 0; i < instances; i++){
    h_row[i] = row[i];
    h_col[i] = col[i];
    h_w[i] = w[i];
  }

}
void grouping(vector<vector<int> > &row_g, vector<vector<int> > &col_g,vector<vector<float> > &w_g,int m, int n, int instances,
              vector<int> &row,vector<int> &col,vector<float> &w){
  for(int i = 0; i < instances; i++){
    int _row = row[i];
    int _col = col[i];
    int _w   = w[i];
    int index = _row - _col;
    if(index < 0){
      row_g[max(m,n)+index].push_back(_row);
      col_g[max(m,n)+index].push_back(_col);
      w_g[max(m,n)+index].push_back(_w);
    }
    else{
      row_g[index].push_back(_row);
      col_g[index].push_back(_col);
      w_g[index].push_back(_w);
    }
  }

  int k = 0;
    for(int i = 0; i < row_g.size(); i++){
      for(int j = 0; j < row_g[i].size(); j++){
        row[k] = row_g[i][j];
        col[k] = col_g[i][j];
        w[k++] = w_g[i][j];
      }
    }
}

void read(vector<int> &row,vector<int> &col,vector<float> &w,int &m, int &n, int &instances, vector<vector<int> > &test){
  ifstream in;
  in.open("train.txt",ios_base::in);
  if(!in.is_open()){
    cout << "error" << endl;
    exit(1);
  }
  //get instances, m, n
  string tmp;
  getline(in,tmp);
  std::istringstream is_tmp(tmp);
  is_tmp >> m;
  is_tmp >> n;
  is_tmp >> instances;

  cout << instances << endl;

  int count = 0;
  for(string line; getline(in,line); ){
    std::istringstream is(line);
    int t1;
    float t2;
    is >> t1;
    row.push_back(t1-1);
    is >> t1;
    col.push_back(t1-1);
    is >> t2;
    w.push_back(t2);
    count++;
  }
  in.close();


  ifstream testIn;
    in.open("test.txt",ios_base::in);
    if(!in.is_open()){
      cout << "error" << endl;
      exit(1);
    }

    for(string line; getline(in,line); ){
      std::istringstream is(line);
      int t1;
      vector<int> tmp;
      is >> t1;
      tmp.push_back(t1-1);
      is >> t1;
      tmp.push_back(t1-1);
      is >> t1;
      tmp.push_back(t1);
      test.push_back(tmp);
    }


    cout << "test data extraction complete, total: " << test.size() << " test instances"<< endl;
    in.close();


}

float standard_deviation(float data[], int n, float &m){
    float mean=0.0, sum_deviation=0.0;
    int i;
    for(i=0; i<n;++i)
    {
        mean+=data[i];
    }
    mean=mean/n;
    for(i=0; i<n;++i)
    sum_deviation+=(data[i]-mean)*(data[i]-mean);
    m = mean;
    return sqrt(sum_deviation/n);
}

__global__ void print(float *y){
  int warp_index = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;
  int index = blockIdx.x*blockDim.x/32 + warp_id;
  if (index == 933)
  printf("%f \n",y[index*32+warp_index]);
}


