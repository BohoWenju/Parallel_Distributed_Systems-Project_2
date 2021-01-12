#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <time.h>

int main(){
  double A[6]={1,1,1,1,1,1};
  double E[6]={1,2,1,2,1,2};
  double B[9]={1,0,0,0,0,0,0,0,0};
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,3,3,2,1,A,2,E,3,0,B,3);
  for(int i=0; i<9; i++)
    printf("%lf ",B[i]);
  printf("\n");

}
