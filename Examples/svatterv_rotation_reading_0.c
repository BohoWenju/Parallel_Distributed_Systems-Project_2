
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZE 4


void shift_right_int(int* arr,int size){
  int temp=arr[0];
  for(int i=0;i<size; i++){
    int temp1=arr[(i+1)%size];
    arr[(i+1)%size]=temp;
    temp=temp1;
  }
}

int main(int argc, char *argv[])
{
    int rank, size;     // for storing this process' rank, and the number of processes
        // array describing the displacements where each segment begins

    int sum = 0;                // Sum of counts. Used to calculate displacements

    // the data to be distributed


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rem = (SIZE*SIZE)%size; // elements remaining after division among processes

    double* X=NULL;
    if (rank==0){
      X=(double*)malloc(SIZE*SIZE*sizeof(double));
      for(int i=0; i<SIZE*SIZE; i++){
        X[i]=i;
        printf("%f\t",X[i]);
      }
      printf("\n");
    }


    int *sendcounts =(int*) malloc(sizeof(int)*size);
    int *displs =(int*) malloc(sizeof(int)*size);

    // calculate send counts and displacements

    for (int i = 0; i < size; i++) {
        sendcounts[i] = (SIZE*SIZE)/size;
        if (rem > 0) {
            sendcounts[i]++;
            rem--;
        }

        displs[i] = sum;
        sum += sendcounts[i];
    }


    // print calculated send counts and displacements for each process


    // divide the data among processes as described by sendcounts and displs




    for(int i=0; i<size; i++){
      double* X_p=(double*)malloc(sendcounts[rank]*sizeof(double));
      if (0 == rank) {
          for (int i = 0; i < size; i++) {
              printf("sendcounts[%d] = %d\tdispls[%d] = %d\n", i, sendcounts[i], i, displs[i]);
          }
      }
      MPI_Scatterv(X, sendcounts, displs, MPI_DOUBLE, X_p, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
      shift_right_int(displs,size);
      shift_right_int(sendcounts,size);
      // print what each process received
      printf("%d: ", rank);
      for (int i = 0; i < sendcounts[rank]; i++) {
          printf("%lf\t", X_p[i]);
      }
      printf("\n");

    }
    //free(X_p);
    free(X);
    MPI_Finalize();



    free(sendcounts);
    free(displs);

    return 0;
}
