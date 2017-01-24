/*
 * grouping.h
 *
 *  Created on: Sep 13, 2016
 *      Author: weitang
 */

#ifndef GROUPING_H_
#define GROUPING_H_

#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <sys/time.h>
using namespace std;

void array_ordering(int instances, vector<int> &row,vector<int> &col,vector<float> &w,int *h_row,int* h_col, float *h_w);
void grouping(vector<vector<int> > &row_g, vector<vector<int> > &col_g,vector<vector<float> > &w_g,int m, int n, int instances,
              vector<int> &row,vector<int> &col,vector<float> &w);
void read(vector<int> &row,vector<int> &col,vector<float> &w,int &m, int &n, int &instances,vector<vector<int> > &test);
float standard_deviation(float data[], int n, float &m);
__global__ void print(float *y);
__global__ void sgd(int *rowIndex, int *colIndex,float *w,float *x, float *y, const int rank, const float rate, const float lambda);
__global__ void sgd2(int *row, int *col,float *w,float *x, float *y, int rank, float rate, float lambda,float *J,float *x_rate, float *y_rate,float gama);

#endif /* GROUPING_H_ */
