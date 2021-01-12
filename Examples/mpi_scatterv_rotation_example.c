#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(){
  double* X=(double*)malloc(8*sizeof(double));
  for(int i=0; i<8; i++)
    X[i]=i;
  MPI_Init(NULL,NULL);
  int rank,size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  if(size==4){
    int distr[4]={0,2,4,6};
    int sendcount[4]={2,2,2,2};
    int recv_data=2;
    double* X_p=(double*)malloc(2*sizeof(double));
    MPI_Scatterv(X,sendcount,distr,MPI_DOUBLE,X_p,recv_data,MPI_DOUBLE,0,MPI_COMM_WORLD);
    printf("\nI am in process %d and the array that was scattered is: ",rank);
    for(int i=0; i<2; i++)
      printf("%lf ", X_p[i]);
    printf("\n");
    for(int i=0; i<4; i++){
      distr[i]=(distr[i]+2)%8;
    }
    MPI_Scatterv(X,sendcount,distr,MPI_DOUBLE,X_p,recv_data,MPI_DOUBLE,0,MPI_COMM_WORLD);
    printf("\nI am in process %d and the array that was scattered is: ",rank);
    for(int i=0; i<2; i++){
      printf("%lf ", X_p[i]);
      X_p[i]++;
    }
    printf("\n");

    double* recv_array=NULL;
    if (rank==0){
      recv_array=(double*)malloc(2*size*sizeof(double));
    }
    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(X_p,2,MPI_DOUBLE,recv_array,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0){
      printf("\nAll elements have been gathered and the new array must contain all the elements with values+1\n");
      for(int i=0; i<8;i++)
        printf("%lf ",recv_array[i]);
      printf("\n All good...\n");
    }
  }
  else
    printf("processes were not 4...\n");

  MPI_Finalize();

  return 0;
}
