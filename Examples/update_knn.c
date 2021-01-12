#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>




// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

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
void update_kNN(knnresult* k1,knnresult* k2,int k){
  for(int z=0; z<k1->m; z++)
    for (int j=0; j<k2->k; j++)
      for (int i=0; i<k1->k; i++)
        if (k2->ndist[z*k+j]<k1->ndist[z*k+i]){
          shift_nc_double(k1->ndist,k1->k,z*k+i,k2->ndist[z*k+j]);
          shift_nc_int(k1->nidx,k1->k,z*k+i,k2->nidx[z*k+j]);
          break;
        }
}

int main() {
  knnresult k1;
  knnresult k2;
  int k=3;
  k1.k=3;
  k2.k=3;
  int m=3;
  int n=3;
  k1.m=m;
  k2.m=n;
  k1.nidx=(int*)malloc(m*k*sizeof(int));
  k1.ndist=(double*)malloc(m*k*sizeof(double));
  k2.ndist=(double*)malloc(n*k*sizeof(double));
  k2.nidx=(int*)malloc(n*k*sizeof(int));
  printf("\nFirst array is: ");
  for(int i=0; i<(m*k); i++){
    k1.nidx[i]=i;
    k1.ndist[i]=i;
    printf("%lf ", k1.ndist[i]);
  }
  printf("\nSecond array is: ");
  for(int j=0; j<(n*k); j++){
    k2.nidx[j]=j-1;
    k2.ndist[j]=j-1;
    printf("%lf ", k2.ndist[j]);
  }
  printf("\n");
  update_kNN(&k1,&k2,k);

  printf("\nFirst array now is: ");
  for(int i=0; i<(m*k); i++)
    printf("%lf ", k1.ndist[i]);
  printf("\n");

  double* arr=(double*)malloc(10*sizeof(double));
  for(int i=0; i<10; i++)
    arr[i]=i;
  shift_nc_double(arr,10,4,8);
  printf("\n");
  for(int i=0; i<10; i++)
    printf("%lf ",arr[i]);
  printf("\n");

  return 0;
}
