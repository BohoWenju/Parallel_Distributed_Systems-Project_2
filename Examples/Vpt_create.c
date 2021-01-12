
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

typedef struct node_t {
  int price;
  struct node_t* left_child;
  struct node_t* right_child;
}node_t;

int pow(int x){
  int result=1;
  for(int i=0; i<x; i++)
    result*=2;
  return result;
}

void fill_send_Arrays(node_t* root,int* index,
  int level,int point_2_store,int d,bool root_flag){

  if(point_2_store>=7)
    return;
  if (root_flag==true){
    index[point_2_store]=root->price;
    fill_send_Arrays(root->left_child,index,level+1,1,d,false);
    fill_send_Arrays(root->right_child,index,level+1,2,d,false);
  }
  else {
    index[point_2_store]=root->price;
    int stp=1;
    for (int i=1; i<level; i++)
      stp+=pow(i);
    int lvl_id=point_2_store-stp;
    point_2_store=lvl_id*2+(stp+2*(level));
    fill_send_Arrays(root->left_child,index,level+1,point_2_store,d,false);
    fill_send_Arrays(root->right_child,index,level+1,point_2_store+1,d,false);
  }
}




int main() {
  //srand(time(NULL));
  //node_t *root=(node_t*)malloc(sizeof(node_t));
  //int n,d,k;
  //n=10;
  //d=2;
  //k=1;
  //int B=2;
  //int a=0;
  //int* iteration=&a;
  //double* X=(double*)malloc((n*d)*sizeof(double));
  //for(int i=0; i<(n*d); i++)
    //X[i]=i;
  //int* indexes=(int*)malloc(n*sizeof(int));
  //for(int i=0; i<n; i++)
    //indexes[i]=i;
  //create_VPtree(X,n,B,d,root,indexes);
  //printf_tree(root,0,0);
  //delete_node(root);
  int b=10;
  int *a=&b;
  (*a)++;
  printf("%d\n",*a);



  int
  node_t* root0=(node_t*)malloc(sizeof(node_t));
  root0->price=0;
  node_t* root1=(node_t*)malloc(sizeof(node_t));
  root1->price=1;
  node_t* root2=(node_t*)malloc(sizeof(node_t));
  root2->price=2;
  node_t* root3=(node_t*)malloc(sizeof(node_t));
  root3->price=3;
  node_t* root4=(node_t*)malloc(sizeof(node_t));
  root4->price=4;
  node_t* root5=(node_t*)malloc(sizeof(node_t));
  root5->price=5;
  node_t* root6=(node_t*)malloc(sizeof(node_t));
  root6->price=6;

  root0->left_child=root1;
  root0->right_child=root2;
  root1->left_child=root3;
  root1->right_child=root4;
  root2->left_child=root5;
  root4->left_child=NULL;
  root4->right_child=NULL;
  root5->left_child=NULL;
  root5->right_child=NULL;
  root6->left_child=NULL;
  root6->right_child=NULL;


  root2->right_child=root6;
  int* indexesa=(int*)malloc(7*sizeof(int));
  fill_send_Arrays(root0,indexesa,0,0,4,true);
  for(int i=0; i<7; i++)
    printf("%d\t",indexesa[i]);
  printf("\n");

  return 0;



}
