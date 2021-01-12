/*
                PARALLEL AND DISTRIBUTED SYSTEMS
                    ELEFTHERIOS MOURELATOS

          In V0.c the process of finding the k nearest neighbors
          of each element in an array X is implemented using knnsearch.
          Knnsearch uses cblas functions to calculate the matrix :
          D=(X.*X)*e*inverse(e) -2*X*inverse(Y) + e*inverse(e)*(Y.*Y).
          Where Dij is the distance between elements Xi and Yj.
          The results of kmin search are later on being passed in a
          struct knnresult.


*/



#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cblas.h>
#include <time.h>
#include <math.h>

#define BLOCK_SIZE 500
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


//generating random points
void generate_random_points(double* matrix,int n,int d){
    for (int i=0; i<(d*n); i++)
      matrix[i]=(rand()%100)+((double)rand())/10000;
}


//! Compute k nearest neighbors of each point in X [n-by-d]
/*!

  \param  X      Corpus data points              [n-by-d]
  \param  Y      Query data points               [m-by-d]
  \param  n      Number of corpus points         [scalar]
  \param  m      Number of query points          [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/
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
      D[j]=sqrt(fabs(A[i]+B[i]+C[i]));

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
    free(A);
    free(B);
    free(C);
    free(D);
    free(Z);
    free(Z_had);
    free(e_z);
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
      D[i]=sqrt(fabs(A[i]+B[i]+C[i]));


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
      free(A);
      free(B);
      free(C);
      free(D);
      free(Z);
      free(Z_had);
      free(e_z);
  }
  k1.m=m;
  k1.k=k;
  return k1;
}



//randomly created matrixes of nxd and mxd
int main(int argc,char *argv[]){
  //initialize variables needed for the arrays
  //user's input
  //ARRAY X WILL HAVE n POINTS
  //ARRAY Y WILL HAVE m POINTS
  srand(time(NULL));
  int n,m,d,k;
  //printf("\nEnter desired dimensions: ");
  //scanf("%d", &d);
  //printf("\nEnter desired number of points for X: ");
  //scanf("%d", &n);
  //printf("\nEnter desired number of points for Y: ");
  //scanf("%d", &m);
  //printf("\nEnter the number of desired closest neighbors: ");
  //scanf("%d", &k);
  //printf("\nAll set!!! \nRandom matrixes will be generated...\n");


  // Initialize for TEST
  n=3;
  m=2;
  d=3;
  k=2;


  // Initialize random matrixes
  double* X=(double*)malloc((n*d)*(sizeof(double)));
  double* Y=(double*)malloc((m*d)*(sizeof(double)));



  /********************************************/
  //                TEST HERE

  X[0]=1;
  X[1]=3;
  X[2]=4;
  X[3]=5;
  X[4]=6;
  X[5]=7;
  X[6]=2;
  X[7]=4;
  X[8]=7;
  Y[0]=1;
  Y[1]=3;
  Y[2]=4;
  Y[3]=7;
  Y[4]=8;
  Y[5]=4;

  /********************************************/
  //              MATRIX READING





  /********************************************/
  //generate_random_points(X,n,d);
  //generate_random_points(Y,m,d);
  knnresult x=kNN(X,Y,n,m,d,k);

  printf("\nThe indices are: ");
  for(int i=0; i<x.m*x.k; i++)
    printf("%d ",x.nidx[i]);
  printf("\n");









  /********************************************/
  //             CLEARING MEMORY


  free(X);
  free(Y);
  return 0;
}
