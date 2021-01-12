#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

double distance_calc(double* X,int index_p,int index_d,int d){
 double distance=0;
 for(int i=0; i<d; i++)
   distance+=(X[index_p*d+i]-X[index_d*d+i])*(X[index_p*d+i]-X[index_d*d+i]);
 return sqrt(distance);
}


double median(double* X,int index_p,int index_d,int size_d,int d){

  double* dist_array=(double*)malloc(size_d*sizeof(double));
  //size of array in a current time
  //creating a sorted array right away
  int temp=0;
  for (int i=0; i<size_d; i++){
    double temp_dist=distance_calc(X,index_p,index_d+i,d);
    printf("Temorary dist is: %lf\n",temp_dist);
    int index_2_store=temp;
    for (int j=0; j<temp; j++)
      if (temp_dist<=dist_array[j]){
        index_2_store=j;
        // shifting array one place to the right
        double temp_dist1=dist_array[j];
        for (int k=j; k<temp; k++){
          double temp_dist2=temp_dist1;
          temp_dist1=dist_array[k+1];
          dist_array[k+1]=temp_dist2;
          temp_dist1=temp_dist2;
        }
      }
    dist_array[index_2_store]=temp_dist;
    temp++;
  }
  printf("Dist array is: ");
  for(int i=0; i<temp; i++)
    printf("%lf ",dist_array[i]);
  printf("\n");
  double median=dist_array[size_d/2];
  free(dist_array);
  return median;
}


void find_random(int n,int* index,int* size){
  int t_index,t_size;
  while (true) {
    t_index=rand()%n;
    if ((n-t_index)< n/4)
      continue;
    t_size=rand()%(n-t_index);
    if (t_size>(n/4))
      break;
  }
  (*index)=t_index;
  (*size)=t_size;
}

int main(){
  srand(time(NULL));
  int d=2;
  double* X=(double*)malloc(4*d*sizeof(double));
  X[0]=0;
  X[1]=0;
  X[2]=1;
  X[3]=1;
  X[4]=2;
  X[5]=2;
  X[6]=3;
  X[7]=3;
  printf("Distance between X(0) and X(1) is: %lf\n",distance_calc(X,0,1,d));
  printf("Distance between X(0) and X(2) is: %lf\n",distance_calc(X,0,2,d));
  printf("Distance between X(0) and X(3) is: %lf\n",distance_calc(X,0,2,d));
  double mu=median(X,0,1,3,d);
  printf("Median is: %lf\n",mu);
  printf("\nChecking find random\n");
  int n=16;
  int index,size;
  find_random(n,&index,&size);
  printf("The index is: %d \n",index);
  printf("The size is: %d \n",size);
  printf("The total size is: %d \n",n);
  printf("Gets elements between indexes: %d - %d \n",index,index+size);

  return 0;
}
