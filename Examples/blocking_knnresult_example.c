#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cblas.h>
#include <time.h>
#include <math.h>

#define BLOCK_SIZE 2
// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

// calculate H=X.*Y
void hadamard(double* X,double* Y,int n,int d,double* H){
  for(int i=0; i<(n*d); i++)
    H[i]=X[i]*Y[i];
}


knnresult kNN(double * X, double * Y, int n, int m, int d, int k){
  //IMPLEMENTING MATRIX D WITH CBLAS
  //D=(X.*X)e*(e') - 2*X*(Y') + e*(e')*(Y.*Y)'

  //creating the blocks of array Y if it is too large
  int iterations=m/BLOCK_SIZE;
  int remainder=m%BLOCK_SIZE;
  double* X_had=(double*)malloc((n*d)*sizeof(double));
  hadamard(X,X,n,d,X_had);
  //inititalize matrix of all ones
  double* e_x=(double*)malloc((n*d)*(sizeof(double)));
  #pragma omp parallel for
    for(int i=0; i<(n*d); i++)
      e_x[i]=1;


  knnresult k1;
  k1.nidx=(int*)malloc(m*k*sizeof(int));
  k1.ndist=(double*)malloc(m*k*sizeof(double));

  for(int i=0; i<iterations; i++){

    double* e_z=(double*)malloc((BLOCK_SIZE*d)*(sizeof(double)));
    #pragma omp parallel for
      for(int i=0; i<(BLOCK_SIZE*d); i++)
        e_z[i]=1;

    double* Z=(double*)malloc(BLOCK_SIZE*d*sizeof(double));
    #pragma omp prallel for
      for (int j=0; j<(BLOCK_SIZE*d); j++)
        Z[j]=Y[(BLOCK_SIZE*i)*d +j];
    //Hadamard implementation in row major format
    //Assuming u have two arrays X,Y in row major format
    //The element aij of the X.*Y is xij*yij
    //The dimensions of the product array remain the same

    double* Z_had=(double*)malloc((BLOCK_SIZE*d)*sizeof(double));
    hadamard(Z,Z,BLOCK_SIZE,d,Z_had);

    // A = (X.*X)*e*(e')
    double* A=(double*)malloc((n*BLOCK_SIZE)*sizeof(double));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,BLOCK_SIZE,d,1,X_had,d,e_z,BLOCK_SIZE,0,A,BLOCK_SIZE);

    // B = (-2*X*(Y'))
    double* B=(double*)malloc((n*BLOCK_SIZE)*sizeof(double));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,BLOCK_SIZE,d,-2,X,d,Z,d,0,B,BLOCK_SIZE);

    // C = e*(e')*((Y.*Y)')
    double* C=(double*)malloc((n*BLOCK_SIZE)*sizeof(double));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,BLOCK_SIZE,d,1,e_x,d,Z_had,d,0,C,BLOCK_SIZE);

    // D = sqrt(A + B + C)
    double* D=(double*)malloc((n*BLOCK_SIZE)*sizeof(double));
    for(int j=0; j<(n*BLOCK_SIZE); j++)
      D[j]=sqrt(A[j]+B[j]+C[j]);

    // Dij element contains the distance between Xi and Yj element
    // so if i want to find the k nearest neighbours of an element Yj
    // all i need to do is perform a k min search in the column J of
    // D array.In order to do so in row major format i must add every time
    // the number of columns to get every XkYj distance
    for (int j=0; j<BLOCK_SIZE; j++){
      // array that holds all the previous elements
      int* temp_array=(int*)malloc(k*sizeof(int));
      double* temp_dist=(double*)malloc(k*sizeof(double));
      int temp=0;
      for (int iter=0; iter<k; iter++){
        double min_dist=INFINITY;
        int temp_index=-1;
        for(int jter=0; jter<n; jter++){
          double temp_dist=D[j+jter*BLOCK_SIZE];
          if ((temp_dist!=0)&&(min_dist>temp_dist)){
            // a flag to know if that element exists in the array
            bool flag=false;
            for(int kter=0; kter<temp; kter++)
              if (temp_array[kter]==((j+jter*BLOCK_SIZE)/BLOCK_SIZE)){
                flag=true;
                break;
            }
            if (flag)
              continue;
            else{
              min_dist=temp_dist;
              temp_index=(j+jter*BLOCK_SIZE)/BLOCK_SIZE;
            }
          }
        }
        temp_array[temp]=temp_index;
        temp_dist[temp]=min_dist;
        temp++;
      }
      for (int iter=0; iter<k; iter++){
        k1.nidx[(i*BLOCK_SIZE*k)+j*k+iter]=temp_array[iter];
        k1.ndist[(i*BLOCK_SIZE*k)+j*k+iter]=temp_dist[iter];
      }
    }
  }
  // now for the remainder
  if (remainder!=0){

    double* e_z=(double*)malloc((remainder*d)*(sizeof(double)));
    #pragma omp parallel for
      for(int i=0; i<(remainder*d); i++)
        e_z[i]=1;

    double* Z=(double*)malloc((remainder*d)*sizeof(double));
    #pragma omp prallel for
    for (int j=0; j<(remainder*d); j++)
      Z[j]=Y[(m-remainder)*d + j];
    double* Z_had=(double*)malloc((remainder*d)*sizeof(double));
    hadamard(Z,Z,remainder,d,Z_had);

    // A = (X.*X)*e*(e')
    double* A=(double*)malloc((n*remainder)*sizeof(double));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,remainder,d,1,X_had,d,e_z,remainder,0,A,remainder);

    // B = (-2*X*(Y'))
    double* B=(double*)malloc((n*remainder)*sizeof(double));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,remainder,d,-2,X,d,Z,d,0,B,remainder);

    // C = e*(e')*((Y.*Y)')
    double* C=(double*)malloc((n*remainder)*sizeof(double));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,remainder,d,1,e_x,d,Z_had,d,0,C,remainder);

    // D = sqrt(A + B + C)
    double* D=(double*)malloc((n*remainder)*sizeof(double));
    #pragma omp parallel for
    for(int i=0; i<(n*remainder); i++)
      D[i]=sqrt(A[i]+B[i]+C[i]);


      for (int j=0; j<remainder; j++){
        // array that holds all the previous elements
        int* temp_array=(int*)malloc(k*sizeof(int));
        double* temp_dist=(double*)malloc(k*sizeof(double));
        int temp=0;
        for (int iter=0; iter<k; iter++){
          double min_dist=INFINITY;
          int temp_index=-1;
          for(int jter=0; jter<n; jter++){
            double temp_dist=D[j+jter*remainder];
            if ((temp_dist!=0)&&(min_dist>temp_dist)){
              // a flag to know if that element exists in the array
              bool flag=false;
              for(int kter=0; kter<temp; kter++)
                if (temp_array[kter]==((j+jter*remainder)/remainder)){
                  flag=true;
                  break;
              }
              if (flag)
                continue;
              else{
                min_dist=temp_dist;
                temp_index=(j+jter*remainder)/remainder;
              }
            }
          }
          temp_array[temp]=temp_index;
          temp_dist[temp]=min_dist;
          temp++;
        }
        for (int iter=0; iter<k; iter++){
          k1.nidx[(m-remainder)*k+j*k+iter]=temp_array[iter];
          k1.ndist[(m-remainder)*k+j*k+iter]=temp_dist[iter];
        }
      }
  }
  k1.m=m;
  k1.k=k;
  return k1;
}


knnresult test(){
  knnresult k1;
  k1.nidx=(int*)malloc(3*sizeof(int));
  for(int i=0; i<3; i++)
    k1.nidx[i]=i;
  return k1;
}


int main() {
  int n=2;
  int m=3;
  int d=3;
  double* Y=(double*)malloc(m*d*sizeof(double));
  double* X=(double*)malloc(n*d*sizeof(double));
  for(int i=0; i<m*d; i++)
    Y[i]=i+1;
  for(int i=0; i<3; i++)
    X[i]=i+1;
  X[3]=7;
  X[4]=8;
  X[5]=4;
  int k=1;
  knnresult K=kNN(X,Y,n,m,d,k);
  printf("\n");
  for(int i=0; i<m*k; i++)
    printf("%d ",K.nidx[i]);
  printf("\n");
  return 0;
}
