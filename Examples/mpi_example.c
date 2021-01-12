#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"


int main(){
  printf("Initializing the mpi environment\n");
  //test to see whether the data is passed to each process
  //a copy of the data should be at least passed
  //also trying with pointers
  int a=10;
  int* p=(int*)malloc(2*sizeof(int));
  p[0]=1;
  p[1]=2;

  //dont need the command line arguments
  MPI_Init(NULL,NULL);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank ==0 ){
    int size;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    printf("The number of processes available on this machine is %d \n",size);
    printf("I am within process %d and a=%d \n",rank,p[0]);
    a++;
    p=p+sizeof(int);
  }
  else if(rank==1){
    printf("I am within process %d and a=%d \n",rank,p[0]);
    p[0]++;
  }
  else
    printf("I am within process %d and a=%d \n",rank,a);
  MPI_Finalize();


}
