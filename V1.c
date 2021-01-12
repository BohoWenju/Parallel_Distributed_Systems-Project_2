/*
                PARALLEL AND DISTRIBUTED SYSTEMS
                    ELEFTHERIOS MOURELATOS

          In V1.c the process of finding the k nearest neighbors
          of each element in an array X is being implemented with
          the MPI INTERFACE.With Mpi_scatter initiallly each process
          takes a subset of X.After calculating the k nearest neighbors
          of each point(excluding itself) the subset of X is being distributed
          while a copy of it remains for all the duration of the programm
          in the process.Knnsearch is being used again for p-1 time where
          p is the number of processes;


*/






#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cblas.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"

#define BLOCK_SIZE 500
#define MAX_CO 100

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


//search for index in used_indexes array
bool check_index(int j,int counter,int* used_indexes){
  bool is_true=false;
  for(int i=0; i<counter; i++)
    if (used_indexes[i]==j){
      is_true=true;
      break;
    }
  return is_true;
}

// function to shift an arr one place to the right
// circular motion is implemented
void shift_right_double(double* arr,int size){
  double temp=arr[0];
  for(int i=0;i<size; i++){
    double temp1=arr[(i+1)%size];
    arr[(i+1)%size]=temp;
    temp=temp1;
  }
}
void shift_right_int(int* arr,int size){
  int temp=arr[0];
  for(int i=0;i<size; i++){
    int temp1=arr[(i+1)%size];
    arr[(i+1)%size]=temp;
    temp=temp1;
  }
}

// function to shift an array one place to the right from
// a given index.The elelement that pops out does not go
// to the beggining of the array
void shift_nc_double(double* arr,int size,int index,double element){
  double temp=arr[index];
  for (int i=index+1; i<size; i++){
    double temp1=arr[i];
    arr[i]=temp;
    temp=temp1;
  }
  arr[index]=element;
}
void shift_nc_int(int* arr,int size,int index,int element){
  int temp=arr[index];
  for (int i=index+1; i<size; i++){
    int temp1=arr[i];
    arr[i]=temp;
    temp=temp1;
  }
  arr[index]=element;
}


// Function to update K
void update_kNN(knnresult* k1,knnresult* k2){
  int k=k1->k;
  for(int z=0; z<k1->m; z++)
    for (int j=0; j<k; j++)
      for (int i=0; i<k; i++)
        if (k2->ndist[z*k+j]<k1->ndist[z*k+i]){
          shift_nc_double(k1->ndist,k1->k,z*k+i,k2->ndist[z*k+j]);
          shift_nc_int(k1->nidx,k1->k,z*k+i,k2->nidx[z*k+j]);
          break;
        }
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
  //#pragma omp parallel for
    for(int i=0; i<(n*d); i++)
      e_x[i]=1;

  double* A=(double*)malloc((n*BLOCK_SIZE)*sizeof(double));
  double* B=(double*)malloc((n*BLOCK_SIZE)*sizeof(double));
  double* C=(double*)malloc((n*BLOCK_SIZE)*sizeof(double));
  double* D=(double*)malloc((n*BLOCK_SIZE)*sizeof(double));
  double* Z=(double*)malloc(BLOCK_SIZE*d*sizeof(double));
  double* Z_had=(double*)malloc((BLOCK_SIZE*d)*sizeof(double));
  double* e_z=(double*)malloc((BLOCK_SIZE*d)*(sizeof(double)));

  knnresult k1;
  k1.nidx=(int*)malloc(m*k*sizeof(int));
  k1.ndist=(double*)malloc(m*k*sizeof(double));

  for(int i=0; i<iterations; i++){

    //#pragma omp parallel for
      for(int i=0; i<(BLOCK_SIZE*d); i++)
        e_z[i]=1;

    //#pragma omp prallel for
      for (int j=0; j<(BLOCK_SIZE*d); j++)
        Z[j]=Y[(BLOCK_SIZE*i)*d +j];
    //Hadamard implementation in row major format
    //Assuming u have two arrays X,Y in row major format
    //The element aij of the X.*Y is xij*yij
    //The dimensions of the product array remain the same

    hadamard(Z,Z,BLOCK_SIZE,d,Z_had);

    // A = (X.*X)*e*(e')
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,BLOCK_SIZE,d,1,X_had,d,e_z,BLOCK_SIZE,0,A,BLOCK_SIZE);

    // B = (-2*X*(Y'))
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,BLOCK_SIZE,d,-2,X,d,Z,d,0,B,BLOCK_SIZE);

    // C = e*(e')*((Y.*Y)')
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,BLOCK_SIZE,d,1,e_x,d,Z_had,d,0,C,BLOCK_SIZE);

    // D = sqrt(A + B + C)
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
      free(temp_array);
      free(temp_dist);
    }

  }
  // now for the remainder
  if (remainder!=0){

    e_z=(double*)realloc(e_z,(remainder*d)*(sizeof(double)));
    //#pragma omp parallel for
      for(int i=0; i<(remainder*d); i++)
        e_z[i]=1;

    Z=(double*)realloc(Z,(remainder*d)*sizeof(double));
    //#pragma omp prallel for
    for (int j=0; j<(remainder*d); j++)
      Z[j]=Y[(m-remainder)*d + j];

    Z_had=(double*)realloc(Z_had,(remainder*d)*sizeof(double));
    hadamard(Z,Z,remainder,d,Z_had);

    // A = (X.*X)*e*(e')
    A=(double*)realloc(A,(n*remainder)*sizeof(double));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,remainder,d,1,X_had,d,e_z,remainder,0,A,remainder);

    // B = (-2*X*(Y'))
    B=(double*)realloc(B,(n*remainder)*sizeof(double));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,remainder,d,-2,X,d,Z,d,0,B,remainder);

    // C = e*(e')*((Y.*Y)')
    C=(double*)realloc(C,(n*remainder)*sizeof(double));
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,remainder,d,1,e_x,d,Z_had,d,0,C,remainder);

    // D = sqrt(A + B + C)
    D=(double*)realloc(D,(n*remainder)*sizeof(double));
    //#pragma omp parallel for
    for (int i=0; i<(n*remainder); i++)
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
        free(temp_array);
        free(temp_dist);
      }
  }
  free(A);
  free(B);
  free(C);
  free(D);
  free(Z);
  free(Z_had);
  free(e_z);
  free(e_x);
  free(X_had);
  k1.m=m;
  k1.k=k;
  return k1;
}



//function to separate the input matrix X to matrixes with n/p
//elements in order to use those elements in separate processes
//using mpi
//! Compute distributed all-kNN of points in X
/*!

  \param  X      Data points                     [n-by-d]
  \param  n      Number of data points           [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/
knnresult distrAllkNN(double * X, int n, int d, int k){
  // function that distributes the array X to p processes
  // then by calling the knnresult implemented above
  // each process calculates the knn result
  // In the end of each process the nidx and ndist array
  // are updated to find the k nearest points
  // Initializing MPI
  MPI_Init(NULL,NULL);


  // getting the rank of every process
  // each process will take the (rank+1)*(n/p) part of matrix X
  // **rank starts at 0
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  // size=p
  int size;
  MPI_Comm_size(MPI_COMM_WORLD,&size);



  if(rank==0){
    X=(double*)malloc(n*d*sizeof(double));
    for(int i=0; i<(n*d); i++)
      X[i]=(double)(rand()%MAX_CO);
    }
    struct timespec ts_start,ts_end;
    clock_gettime(CLOCK_MONOTONIC,&ts_start);
  // initializing displ, sendcount arrays
  int remainder=n%size;
  int* sendcount=(int*)malloc(size*sizeof(int));
  int* displ=(int*)malloc(size*sizeof(int));
  int sum=0;
  for (int i=0; i<size; i++){
    sendcount[i]=(n/size)*d;
    if (remainder>0){
      sendcount[i]+=d;
      remainder--;
    }
    displ[i]=sum;
    sum+=sendcount[i];
  }
  remainder=n%size;
  double* Y_p=(double*)malloc(sendcount[rank]*sizeof(double));

  // distributing the array X
  MPI_Scatterv(X,sendcount,displ,MPI_DOUBLE,Y_p,sendcount[rank],MPI_DOUBLE,0,MPI_COMM_WORLD);

  // first iteration has Y=X
  double* X_p=(double*)malloc(sendcount[rank]*sizeof(double));
  X_p=Y_p;
  int m=sendcount[rank]/d;
  knnresult k1=kNN(X_p,Y_p,m,m,d,k);

  // making sure the indices have the indices that correspond to the
  // "big" array
  if(size>1)
  #pragma omp parallel for
    for(int i=0; i<m*k; i++)
      k1.nidx[i]=k1.nidx[i]+displ[rank]/d;

  knnresult k2;
  for (int i=1; i<size; i++){
    // in need to update the displ/sendcount array every time
    // all elements of the array must be shifted one place to the right
    // that way the Y_p of each iteration is the Y_p the previous process had
    // in the previous itertion

    shift_right_int(displ,size);
    shift_right_int(sendcount,size);
    // reallocating array X_p in order to receive the next elements

    X_p=(double*)realloc(X_p,sendcount[rank]*sizeof(double));
    // scatter again to get new X_p

    MPI_Scatterv(X,sendcount,displ,MPI_DOUBLE,X_p,sendcount[rank],MPI_DOUBLE,0,MPI_COMM_WORLD);

    // new n is the number of elements received
    int l=sendcount[rank]/d;

    k2=kNN(X_p,Y_p,l,m,d,k);

    #pragma omp parallel for
    for(int j=0; j<m*k; j++)
      k2.nidx[j]=k2.nidx[j]+(displ[rank]/d);

    /*  update k */
    update_kNN(&k1,&k2);

  }


  // return the arrays to their original form
  // in order to extract number of elements
  shift_right_int(displ,size);
  shift_right_int(sendcount,size);

  // All the ks for all the subparts of X have been calculated
  // Now rank=0 will gather the data and update each time the k_array


  knnresult kNN;
  // sending/receiving the fully updated for each subpart of X knnresult
  // gathering all knnresults to an array in process 0
  // and finally updating the knnresult of 0 for each knnresult
  // of other processes
  if (rank!=0){
    // k1 to process wwith rank=0.A problem occurs with dynamic memory
    // so in order to avoid that multiple messages are being send
    // in order to form the new k in process with rank=0

    MPI_Send(k1.nidx,k*m,MPI_INT,0,rank,MPI_COMM_WORLD);
    MPI_Send(k1.ndist,k*m,MPI_DOUBLE,0,rank,MPI_COMM_WORLD);
  }
  else{
    MPI_Status stat;
    kNN.m=n;
    kNN.k=k;
    kNN.nidx=(int*)malloc(k*n*sizeof(int));
    kNN.ndist=(double*)malloc(k*n*sizeof(double));

    int temp=0;
    for(int i=0; i<m*k; i++){
      kNN.nidx[temp]=k1.nidx[i];
      kNN.ndist[temp]=k1.ndist[i];
      temp++;
    }

    for (int i=0; i<size; i++){
      if (i==0){
        continue;
      }
      int m1=(sendcount[i]/d);
      int* nidx=(int*)malloc(k*m1*sizeof(int));
      double* ndist=(double*)malloc(k*m1*sizeof(double));

      MPI_Recv(nidx,k*m1,MPI_INT,i,i,MPI_COMM_WORLD,&stat);

      MPI_Recv(ndist,k*m1,MPI_DOUBLE,i,i,MPI_COMM_WORLD,&stat);

      for(int j=0; j<m1*k; j++){
        kNN.nidx[temp]=nidx[j];
        kNN.ndist[temp]=ndist[j];
        temp++;
      }

      free(nidx);
      free(ndist);
    }


    clock_gettime(CLOCK_MONOTONIC,&ts_end);

    printf("Time for V1 imlpementation in secs : %lf \n",( (double)ts_end.tv_sec+(double)ts_end.tv_nsec*pow(10,-9) - (double)ts_start.tv_sec -(double)ts_start.tv_nsec*pow(10,-9) ));

  }
  free(X_p);
  free(k1.nidx);
  free(k1.ndist);
  free(sendcount);
  free(displ);
  MPI_Finalize();
  return kNN;
}





// randomly created matrixes of nxd and mxd
int main(int argc,char *argv[]){
  // initialize variables needed for the arrays
  // user's input
  // ARRAY X WILL HAVE n POINTS
  // ARRAY Y WILL HAVE m POINTS
  srand(time(NULL));
  int n,d,k;


  n=10000;
  d=50;
  k=400;
  // initialize random matrixes
  double* X=NULL;



  /********************************************/
  /*              MATRIX READING              */





  /********************************************/
  /*              V1 STARTS HERE              */


  knnresult x=distrAllkNN(X,n,d,k);







  /********************************************/
  /*             CLEARING MEMORY              */
  return 0;
}
