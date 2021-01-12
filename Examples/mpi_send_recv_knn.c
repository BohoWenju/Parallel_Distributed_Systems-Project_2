#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>
#include "mpi.h"


// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;



int main() {


  MPI_Init(NULL,NULL);
  int k=3;
  knnresult k1;
  k1.nidx=(int*)malloc(k*sizeof(int));
  int rank,size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Datatype knnresult_mpi;
  int lengths[4]={k,k,1,1};
  const MPI_Aint displacements[4]={0,k*sizeof(int),k*sizeof(int)+k*sizeof(double),k*(sizeof(int)+sizeof(double))+sizeof(int)};
  MPI_Datatype types[4]={MPI_INT,MPI_DOUBLE,MPI_INT,MPI_INT};
  MPI_Type_create_struct(4,lengths,displacements,types,&knnresult_mpi);
  MPI_Type_commit(&knnresult_mpi);
  if (rank!=0) {
    k1.m=3;
    for(int i=0; i<k; i++)
      k1.nidx[i]=i;
    printf("i am in send... \n");
    MPI_Send(&k1,1,knnresult_mpi,0,rank,MPI_COMM_WORLD);
    MPI_Send(k1.nidx,k,MPI_INT,0,rank,MPI_COMM_WORLD);
    printf("exiting send...\n");
  }
  else{
    MPI_Status stat;
    knnresult new_k;
    for (int i=0; i<(size-1); i++){
      int* recv=(int *)malloc(k*sizeof(int));
      printf("i am in recv...\n");
      MPI_Recv(&new_k,1,knnresult_mpi,i+1,i+1,MPI_COMM_WORLD,&stat);
      printf("all good \n");
      MPI_Recv(recv,k,MPI_INT,i+1,i+1,MPI_COMM_WORLD,&stat);
      printf("exiting recv...\n");
      new_k.nidx=recv;
    }
    printf("%d \n",new_k.m);
    for(int i=0; i<k; i++){
      printf("%d ",new_k.nidx[i]);
    }
  }
  MPI_Finalize();
  printf("\n");
  return 0;
}
