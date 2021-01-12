/*
                PARALLEL AND DISTRIBUTED SYSTEMS
                    ELEFTHERIOS MOURELATOS

          In V2.c the process of finding the k nearest neighbors
          of each element in an array X is being divided in to two
          steps: Creating a Vantage Point tree with a given subset
          of X , using an extended form of knnsearch,which takes
          advantage of the previously created tree , to find the
          k nearest neighbors of each point using tha tree and final
          passing that tree to the other processes in order to be
          recreated and be uses by other subsets.


*/



#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include "mpi.h"

#define BLOCK_SIZE 500

// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;



// Definition of nodes gor the VP tree
typedef struct node_t{
  int index_p; // index of element in X array,index in a process will have
               // the index of the X_p part of X and it will change when the
               // VP tree created in that process is being sent
  double* vector; //coordinates of each element size==d
  double mu;   // median of distances between the other elements and this one
  int B_index_r; // the index of which the branch of this element starts in B array
  int B_size_r;  // the size of elements in this node's leaf,left side
  int B_index_l;
  int B_size_l;
  struct node_t *left_child;  // left child in X array
  struct node_t *right_child; // right child in X array
} node_t;

int b_size=0;
int k_size=0;
int b_hold=0;
int i_size=0;


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

void shift_k(int size,int index,int element,double element_d,knnresult* k1){
  int temp=k1->nidx[index];
  double temp1=k1->ndist[index];
  for (int i=index+1; i<size; i++){
    int temp2=k1->nidx[i];
    k1->nidx[i]=temp;
    temp=temp2;
    double temp3=k1->ndist[i];
    k1->ndist[i]=temp1;
    temp1=temp3;
  }
  k1->nidx[index]=element;
  k1->ndist[index]=element_d;
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




 /*               VP_TREE               */



// function to delete a node_t object
void delete_node(node_t* root){
  if(root->B_index_l!=-2)
    delete_node(root->left_child);
  if(root->B_index_r!=-2)
    delete_node(root->left_child);
  free(root);
}
 //simple algorithm to find a random index and a random size
 //to use in function select_vp
 //size > n/4 was chosen in order for size to be considerable
void find_random(int n,int* index,int* size){
  int t_index,t_size;
  while (true) {
    t_index=rand()%n;
    t_size=rand()%(n-(n/4))+n/4;
    if(t_index+t_size<=n)
      break;
  }
  (*index)=t_index;
  (*size)=t_size;
}

//function to calculate distance between two points of dimensions d
//in row major format
double distance_calc(double* X,int index_p,int index_d,int d){
 double distance=0;
 for(int i=0; i<d; i++)
   distance+=(X[index_p*d+i]-X[index_d*d+i])*(X[index_p*d+i]-X[index_d*d+i]);
 return sqrt(distance);
}

double distance_calc1(double* X,double* Y,int d){
  double distance=0;
  for(int i=0; i<d; i++)
    distance+=(X[i]-Y[i])*(X[i]-Y[i]);
  return sqrt(distance);
}
//find median of some distances
double median(double* X,int* indexes,int index_p,int index_d,int size_d,int d){

  double* dist_array=(double*)malloc(size_d*sizeof(double));
  //size of array in a current time
  //creating a sorted array right away
  int temp=0;
  for (int i=0; i<size_d; i++){
    double temp_dist=distance_calc(X,index_p,indexes[index_d+i],d);
    int index_2_store=temp;
    for (int j=0; j<temp; j++)
      if (temp_dist<=dist_array[j]){
        index_2_store=j;
        // doesnt matter the size of the array is exactly size_d
        // so no element is being throwed out
        shift_nc_double(dist_array,size_d,j,temp_dist);
        break;
      }
    dist_array[index_2_store]=temp_dist;
    temp++;
  }
  double median=dist_array[size_d/2];
  free(dist_array);
  return median;
}




 //algorithm to find the best vantage point for a given space
int select_vp(double* X,int n,int d,int* indexes){
  int index_p,size_p;
  find_random(n,&index_p,&size_p);
  double best_spread=0;
  int best_p=0;
  for (int i=0; i<size_p; i++){
    int index_d,size_d;
    find_random(n,&index_d,&size_d);
    double mu=median(X,indexes,indexes[index_p],index_d,size_d,d);
    double spread=0;
    for (int j=0; j<size_d; j++)
      spread+=((distance_calc(X,indexes[index_p],indexes[index_d+j],d)-mu))*((distance_calc(X,indexes[index_p],indexes[index_d+j],d)-mu));
    if (spread>best_spread){
      best_spread=spread;
      best_p=indexes[index_p+i];
    }
  }
  return best_p;
}


// extended kNN search given as input a vantage point TREE and an element Y_i
// Output is the k nearest neighbours of every point of a point Y_i


double extended_kNN(node_t* root,double* Y_i,int index_y,int d,int k,
  knnresult* k1,double* B_vector,int* B_indexes){



  // calculate distance with root of TREE
  double max_dist=0;
  double temp_dist=distance_calc1(root->vector,Y_i+index_y*d,d);
  for (int i=0; i<k; i++){
    if (temp_dist<k1->ndist[i+index_y*k]){
      shift_k(k1->k,i+index_y*k,root->index_p,temp_dist,k1);
      if(k_size==k)
        break;
      (k_size)++;
      break;
    }
  }
  if ((temp_dist>=root->mu)&&(root->B_index_r!=-2)){
    // if i am in branch
    if (root->B_size_r>0)
      for (int i=0; i<root->B_size_r; i++){
        double dist=distance_calc1(B_vector+(root->B_index_r)*d+i*d,Y_i+index_y*d,d);
        for (int j=0; j<k; j++)
            if (dist<k1->ndist[j+index_y*k]){
                shift_k(k1->k,j+index_y*k,B_indexes[root->B_index_r+i],dist,k1);
                if(k_size==k)
                    break;
                (k_size)++;
                break;
            }
      }
    else
      max_dist=extended_kNN(root->right_child,Y_i,index_y,d,k,k1,B_vector,B_indexes);
    if ((fabs(temp_dist-max_dist)<root->mu)&&(root->B_index_l!=-2))
      max_dist=extended_kNN(root->left_child,Y_i,index_y,d,k,k1,B_vector,B_indexes);
  }
  else if ((temp_dist<root->mu)&&(root->B_index_l!=-2)){
    // if i am in branch
    if(root->B_size_l>0)
      for (int i=0; i<root->B_size_l; i++){
        double dist=distance_calc1(B_vector+(root->B_index_l)*d+i*d,Y_i+index_y*d,d);
        for (int j=0; j<k; j++)
            if (dist<k1->ndist[j+index_y*k]){
                shift_k(k1->k,j+index_y*k,B_indexes[root->B_index_l+i],dist,k1);
                if(k_size==k)
                    break;
                (k_size)++;
                break;
            }
      }
      // get max distance
    else
      max_dist=extended_kNN(root->left_child,Y_i,index_y,d,k,k1,B_vector,B_indexes);
    if ((fabs(temp_dist-max_dist)>=root->mu)&&(root->B_index_r!=-2))
      max_dist=extended_kNN(root->right_child,Y_i,index_y,d,k,k1,B_vector,B_indexes);
  }
  return k1->ndist[index_y*k+(k_size)-1];
}

// creating the vantage point tree
void create_VPtree(double *X,int size,int B,int d,node_t* root,
  int* indexes,int* B_indexes,double* B_vector,int* B_holder,int m){


  // in case the size of the leftover nodes are B make node
  // be simillar to a NULL object
  root->B_size_r=0;
  root->B_size_l=0;
  // -2 this node has no child from either the right or the left side
  root->B_index_r=-2;
  root->B_index_l=-2;
  root->index_p=select_vp(X,size,d,indexes);
  root->vector=(double*)malloc(d*sizeof(double));
  for(int i=0; i<d; i++)
    root->vector[i]=X[root->index_p*d+i];
  root->mu=median(X,indexes,root->index_p,0,size,d);
  int* L_indexes=(int*)malloc(size*sizeof(int));
  int* R_indexes=(int*)malloc(size*sizeof(int));
  // indices to reduce the size of arrays L,R
  // creating the arrays R,L which are basically
  // sub arrays of X
  int temp_L=0,temp_R=0;
  for(int i=0; i<size; i++){
    if (indexes[i]==root->index_p)
      continue;
    double distance=distance_calc(X,root->index_p,indexes[i],d);
    if (distance>=root->mu){
      R_indexes[temp_R]=indexes[i];
      temp_R++;
    }
    else{
      L_indexes[temp_L]=indexes[i];
      temp_L++;
    }
  }

  // reallocating memory for L,R
  // making them have the right size
  L_indexes=(int*)realloc(L_indexes,sizeof(int)*temp_L);
  R_indexes=(int*)realloc(R_indexes,sizeof(int)*temp_R);

  if ((temp_L>0)&&(temp_L<=B)){
    root->B_index_l=(b_size);
    B_holder[b_hold]=root->index_p;
    b_hold++;
    root->B_size_l=temp_L;
    root->left_child=NULL;
    for(int i=0; i<temp_L; i++){
      B_indexes[b_size]=L_indexes[i];
      for (int j=0; j<d; j++)
        B_vector[b_size*d +j]=X[L_indexes[i]*d+j];
      b_size++;
    }
  }
  else if (temp_L>B){
    // -1 node has child but its not a leaf
    root->B_index_l=-1;
    root->left_child=(node_t*)malloc(sizeof(node_t));
    create_VPtree(X,temp_L,B,d,root->left_child,L_indexes,B_indexes,B_vector,B_holder,m);
  }
  if ((temp_R>0)&&(temp_R<=B)){
    root->B_index_r=(b_size);
    // - index to signify the right side
    B_holder[b_hold]=-(root->index_p);
    // 0element cant have -0 so m
    if (root->index_p==0)
        B_holder[b_hold]=m;
    b_hold++;
    root->B_size_r=temp_R;
    root->right_child=NULL;
    for(int i=0; i<temp_R; i++){
      B_indexes[b_size]=R_indexes[i];
      for (int j=0; j<d; j++)
        B_vector[b_size*d +j]=X[R_indexes[i]*d+j];
      b_size++;
    }
  }
  else if(temp_R>B){
    root->B_index_r=-1;
    root->right_child=(node_t*)malloc(sizeof(node_t));
    create_VPtree(X,temp_R,B,d,root->right_child,R_indexes,B_indexes,B_vector,B_holder,m);
  }
  free(L_indexes);
  free(R_indexes);
}

// function to store the tree into arrays
// in order to send it to another process
void fill_send_Arrays(node_t* root,int** index,double** dim,
  double** mu,int* size_branches,int level,int point_2_store,int d,
  bool root_flag){

  int size_l=0;
  int size_r=0;
  if (root_flag==true){
    int x=0;
    for(int i=0; i<=level; i++)
        x+=pow(2,i);
    if (i_size<=point_2_store){
      (*index)=(int*)realloc((*index),x*sizeof(int));
      for (int i=i_size; i<x; i++)
          (*index[i])=-1;
      (*dim)=(double*)realloc((*dim),x*d*sizeof(double));
      (*mu)=(double*)realloc((*mu),x*sizeof(double));
      i_size=x;
    }
    (*index)[point_2_store]=root->index_p;
    (*mu)[point_2_store]=root->mu;
    for(int i=0; i<d; i++)
      (*dim)[point_2_store*d +i]=root->vector[i];
    if (root->B_index_l==-1)
        fill_send_Arrays(root->left_child,index,dim,mu,size_branches,level+1,1,d,false);
    else if(root->B_index_l>=0){
        if (root->B_size_l>0){
          size_branches[b_size]=root->B_size_l;
          (b_size)++;
          size_l=point_2_store+1;
        }
    }
    if(root->B_index_r==-1)
        fill_send_Arrays(root->right_child,index,dim,mu,size_branches,level+1,2,d,false);
    else if(root->B_index_r>=0){
        if (root->B_size_r>0){
          size_branches[b_size]=root->B_size_r;
          (b_size)++;
          size_l=point_2_store+1;
        }
    }
  }
  else {
    int x=0;
    for(int i=0; i<=level; i++)
        x+=pow(2,i);
    if (i_size<=point_2_store){
      (*index)=(int*)realloc((*index),x*sizeof(int));
      for (int i=i_size; i<x; i++)
          (*index[i])=-1;
      (*dim)=(double*)realloc((*dim),x*d*sizeof(double));
      (*mu)=(double*)realloc((*mu),x*sizeof(double));
      i_size=x;
    }
    (*index)[point_2_store]=root->index_p;
    (*mu)[point_2_store]=root->mu;
    for(int i=0; i<d; i++)
      (*dim)[point_2_store*d +i]=root->vector[i];

    // array thah holds the indexes of the elements with branches
    bool flag1=true;
    if (root->B_index_l==-2)
        flag1=false;
    bool flag2=true;
    if (root->B_index_r==-2)
        flag2=false;
    if ((root->B_index_l!=-2)&&(root->B_size_l>0)){
      size_branches[b_size]=root->B_size_l;
      (b_size)++;
      // dont go to left child
      flag1=false;
    }
    else if ((root->B_index_r!=-2)&&(root->B_size_r>0)){
      size_branches[b_size]=root->B_size_r;
      (b_size)++;
      // dont go to right child
      flag2=false;
    }
    int stp=0;
    for (int i=0; i<level; i++)
      stp+=pow(2,i);
    int n_stp=stp+pow(2,level);
    int lvl_id=point_2_store-stp;
    point_2_store=n_stp+2*lvl_id;
    if(root->B_index_l==-1)
      fill_send_Arrays(root->left_child,index,dim,mu,size_branches,level+1,point_2_store,d,false);
    if(root->B_index_r==-1)
      fill_send_Arrays(root->right_child,index,dim,mu,size_branches,level+1,point_2_store+1,d,false);
  }
}

// function to retrieve a tree from an MPI recv
void empty_recv_Arrays(node_t** root,int* index,double* dim,
  double* mu,int* B_holder,int* B_indexes,int index_t,int d,int level,
  int* size_branches,bool root_flag,int m,int size){


  if (root_flag==true){
    (*root)->index_p=index[index_t];
    for(int j=0; j<d; j++)
      (*root)->vector[j]=dim[(index_t)*d+j];
    (*root)->mu=mu[index_t];
    (*root)->B_index_l=-1;
    (*root)->B_index_r=-1;
    bool flag1=true;
    bool flag2=true;
    for (int i=0; i<b_hold; i++){
        if ((index[index_t]==fabs(B_holder[i]))||((index[index_t]==0)&&(B_holder[i]==m))){
            if ((B_holder[i]>=0)&&(B_holder[i]!=m)){
                (*root)->B_size_l=size_branches[i];
                int sum=0;
                for (int j=0; j<i; j++)
                    sum+=size_branches[j];
                (*root)->B_index_l=sum;
                flag1=false;
            }
            else {
                (*root)->B_size_r=size_branches[i];
                int sum=0;
                for (int j=0; j<i; j++)
                    sum+=size_branches[j];
                (*root)->B_index_r=sum;
                flag2=false;
            }
            if ((!flag1)&&(!flag2))
                break;
        }
        if ((i==b_hold-1)&&(flag1)&&(index[index_t+1]==-1)||(index_t+1>size)){
            flag1=false;
            (*root)->B_index_l=-2;
        }
        if ((i==b_hold-1)&&(flag2)&&(index[index_t+2]==-1)||(index_t+2>size)){
            flag2=false;
            (*root)->B_index_r=-2;
        }
    }
    if ((flag1)&&(index[index_t+1]==-1)){
        (*root)->B_index_l=-2;
        flag1=false;
    }
    if (flag1){
        (*root)->left_child=(node_t*)malloc(sizeof(node_t));
        (*root)->left_child->vector=(double*)malloc(d*sizeof(double));
        empty_recv_Arrays(&(*root)->left_child,index,dim,mu,B_holder,B_indexes,1,d,level+1,size_branches,false,m,size);
    }
    if ((flag2)&&(index[index_t+2]==-1)){
        (*root)->B_index_r=-2;
        flag2=false;
    }
    if (flag2){
        (*root)->right_child=(node_t*)malloc(sizeof(node_t));
        (*root)->right_child->vector=(double*)malloc(d*sizeof(double));
        empty_recv_Arrays(&(*root)->right_child,index,dim,mu,B_holder,B_indexes,2,d,level+1,size_branches,false,m,size);
    }
  }
  else {
    (*root)->index_p=index[index_t];
    for(int j=0; j<d; j++)
        (*root)->vector[j]=dim[(index_t)*d+j];
    (*root)->mu=mu[index_t];
    (*root)->B_index_l=-1;
    (*root)->B_index_r=-1;
    // index for left,right_child
    int stp=0;
    for (int i=0; i<level; i++)
        stp+=pow(2,i);
    int n_stp=stp+pow(2,level);
    int lvl_id=index_t-stp;
    int n_index=n_stp+2*lvl_id;
    bool flag1=true;
    bool flag2=true;
    for (int i=0; i<b_hold; i++){
        if ((index[index_t]==fabs(B_holder[i]))||((index[index_t]==0)&&(B_holder[i]==m))){
            if ((B_holder[i]>=0)&&(B_holder[i]!=m)){
                (*root)->B_size_l=size_branches[i];
                int sum=0;
                for (int j=0; j<i; j++)
                    sum+=size_branches[j];
                (*root)->B_index_l=sum;
                flag1=false;
            }
            else {
                (*root)->B_size_r=size_branches[i];
                int sum=0;
                for (int j=0; j<i; j++)
                    sum+=size_branches[j];
                (*root)->B_index_r=sum;
                flag2=false;
            }
            if ((!flag1)&&(!flag2))
                break;
            }
        if ((i==b_hold-1)&&(flag1)&&(index[n_index]==-1)||(n_index>size)){
            flag1=false;
            (*root)->B_index_l=-2;
        }
        if ((i==b_hold-1)&&(flag2)&&(index[n_index+1]==-1)||(n_index>size)){
            flag2=false;
            (*root)->B_index_r=-2;
        }
    }
    if ((flag1)&&(index[n_index]==-1)){
        (*root)->B_index_l=-2;
        flag1=false;
    }
    if (flag1){
        (*root)->left_child=(node_t*)malloc(sizeof(node_t));
        (*root)->left_child->vector=(double*)malloc(d*sizeof(double));
        empty_recv_Arrays(&(*root)->left_child,index,dim,mu,B_holder,B_indexes,n_index,d,level+1,size_branches,false,m,size);
    }
    if ((flag2)&&(index[n_index+1]==-1)){
        (*root)->B_index_r=-2;
        flag2=false;
    }
    if (flag2){
        (*root)->right_child=(node_t*)malloc(sizeof(node_t));
        (*root)->right_child->vector=(double*)malloc(d*sizeof(double));
        empty_recv_Arrays(&(*root)->right_child,index,dim,mu,B_holder,B_indexes,n_index+1,d,level+1,size_branches,false,m,size);
    }
  }
}

/*
   Extended distrAllkNN to work with Vantage point tree from
   other processes.
   ~Each process will have a selected number of points from array X.
   ~Then for those points the VPT will be created.
   ~Afterwards,the k nearest neighbours will be found for that particular
   set of points with the algorithm VPT_search
   ~When all of this is done along the ring will be parsed the
   VPT in order for the other processes to do the same.
   ~Finally,all knnresults will be gathered to main process (rank==0)
   and the complete knnresult will be returned
*/

knnresult distrAllkNN_extended(double*  X,int n, int d, int k){


  // Initializing MPI
  MPI_Init(NULL,NULL);

  // getting rank,size
  int rank,size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  if (rank==0){
    X=(double*)malloc(n*d*sizeof(double));
    for(int i=0; i<(n*d); i++)
      X[i]=i;
  }
  struct timespec ts_start,ts_end;
  clock_gettime(CLOCK_MONOTONIC,&ts_start);
  // initializing displ,sendcount arrays
  // same as distrAllkNN of V1
  int remainder=n%size;
  int* displ=(int*)malloc(size*sizeof(int));
  int* sendcount=(int*)malloc(size*sizeof(int));
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

  // X_p will have the (n/p ~+ 1) elements
  // as will have the Y_p because at first Y_p=X_p
  double* Y_p=(double*)malloc(sendcount[rank]*sizeof(double));

  //distributing array X to all processes
  MPI_Scatterv(X,sendcount,displ,MPI_DOUBLE,Y_p,sendcount[rank],MPI_DOUBLE,0,MPI_COMM_WORLD);

  int m=sendcount[rank]/d;
  // Decide B
  int B=k+1;

  // creating a root for output of VPT_create
  node_t* root=(node_t*)malloc(sizeof(node_t));

  int* indexes=(int*)malloc(m*sizeof(int));
  for(int i=0; i<m; i++)
    indexes[i]=i+displ[rank]/d;

  int* B_holder=(int*)malloc(m*sizeof(int));
  int* B_indexes=(int*)malloc(m*sizeof(int));
  double* B_vector=(double*)malloc(m*d*sizeof(double));
  create_VPtree(Y_p,m,B,d,root,indexes,B_indexes,B_vector,B_holder,m);
  B_indexes=(int*)realloc(B_indexes,b_size*sizeof(int));
  B_vector=(double*)realloc(B_vector,b_size*d*sizeof(double));
  B_holder=(int*)realloc(B_holder,b_hold*sizeof(int));
  free(indexes);
  knnresult* k1=(knnresult*)malloc(sizeof(knnresult));
  k1->nidx=(int*)malloc(m*k*sizeof(int));
  k1->ndist=(double*)malloc(m*k*sizeof(double));
  for(int i=0; i<m*k; i++)
    k1->ndist[i]=INFINITY;
  k1->k=k;
  k1->m=m;
  // calculate the knn of each element in Y_p given the VP tree
  for (int i=0; i<m; i++){
    extended_kNN(root,Y_p,i,d,k,k1,B_vector,B_indexes);
    k_size=0;
  }
  // update the indices of Y_p in order to match those of array X
  node_t* root1=root;
  MPI_Status stat;
  for (int i=1; i<size; i++){

    // sending the vp tree
    // one array to hold the indexes of each point
    // if it is -1 it is a leaf
    int* index_snd=(int*)malloc(sizeof(int));
    index_snd[0]=-1;
    i_size=1;
    // one array to hold the dimensions of each point
    double* dim_snd=(double*)malloc(d*sizeof(double));
    // one array to hold the mu of each element
    double* mu_snd=(double*)malloc(sizeof(double));
    // one array that holds which elements have leaves
    b_size=0;
    int* size_branches=(int*)malloc(m*sizeof(int));
    // size of branch array
    // filling the arrays with a recursive Function
    fill_send_Arrays(root1,&index_snd,&dim_snd,&mu_snd,size_branches,0,0,d,true);
    size_branches=(int*)realloc(size_branches,b_size*sizeof(int));
    // sending the arrays to other processes
    // then they will recreate the tree and perform
    // extended_kNN for their set on that TREE
    MPI_Send(&i_size,1,MPI_INT,(rank+i)%size,rank,MPI_COMM_WORLD);
    MPI_Send(index_snd,i_size,MPI_INT,(rank+i)%size,rank,MPI_COMM_WORLD);
    MPI_Send(dim_snd,i_size*d,MPI_DOUBLE,(rank+i)%size,rank,MPI_COMM_WORLD);
    MPI_Send(mu_snd,i_size,MPI_DOUBLE,(rank+i)%size,rank,MPI_COMM_WORLD);
    MPI_Send(&b_size,1,MPI_INT,(rank+i)%size,rank,MPI_COMM_WORLD);
    MPI_Send(size_branches,b_size,MPI_INT,(rank+i)%size,rank,MPI_COMM_WORLD);
    MPI_Send(B_indexes,b_size,MPI_INT,(rank+i)%size,rank,MPI_COMM_WORLD);
    MPI_Send(B_vector,b_size*d,MPI_DOUBLE,(rank+i)%size,rank,MPI_COMM_WORLD);
    MPI_Send(&b_hold,1,MPI_INT,(rank+i)%size,rank,MPI_COMM_WORLD);
    MPI_Send(B_holder,b_hold,MPI_INT,(rank+i)%size,rank,MPI_COMM_WORLD);

    free(index_snd);
    free(dim_snd);
    free(mu_snd);
    free(size_branches);
    free(B_indexes);
    free(B_vector);
    free(B_holder);
    // Receiving the arrays from the other processes
    MPI_Recv(&i_size,1,MPI_INT,(rank-i+size)%size,(rank-i+size)%size,MPI_COMM_WORLD,&stat);

    int* index_rcv=(int*)malloc(i_size*sizeof(int));
    MPI_Recv(index_rcv,i_size,MPI_INT,(rank-i+size)%size,(rank-i+size)%size,MPI_COMM_WORLD,&stat);

    double* dim_rcv=(double*)malloc(i_size*d*sizeof(double));
    MPI_Recv(dim_rcv,i_size*d,MPI_DOUBLE,(rank-i+size)%size,(rank-i+size)%size,MPI_COMM_WORLD,&stat);

    double* mu_rcv=(double*)malloc(i_size*sizeof(double));
    MPI_Recv(mu_rcv,i_size,MPI_DOUBLE,(rank-i+size)%size,(rank-i+size)%size,MPI_COMM_WORLD,&stat);

    MPI_Recv(&b_size,1,MPI_INT,(rank-i+size)%size,(rank-i+size)%size,MPI_COMM_WORLD,&stat);

    int *size_branches_n=(int*)malloc(b_size*sizeof(int));
    MPI_Recv(size_branches_n,b_size,MPI_INT,(rank-i+size)%size,(rank-i+size)%size,MPI_COMM_WORLD,&stat);

    int* B_indexes=(int*)malloc(b_size*sizeof(int));
    MPI_Recv(B_indexes,b_size,MPI_INT,(rank-i+size)%size,(rank-i+size)%size,MPI_COMM_WORLD,&stat);

    double* B_vector=(double*)malloc(b_size*d*sizeof(double));
    MPI_Recv(B_vector,b_size*d,MPI_DOUBLE,(rank-i+size)%size,(rank-i+size)%size,MPI_COMM_WORLD,&stat);

    MPI_Recv(&b_hold,1,MPI_INT,(rank-i+size)%size,(rank-i+size)%size,MPI_COMM_WORLD,&stat);

    int* B_holder=(int*)malloc(b_hold*sizeof(int));
    MPI_Recv(B_holder,b_hold,MPI_INT,(rank-i+size)%size,(rank-i+size)%size,MPI_COMM_WORLD,&stat);

    int mpi_size=sendcount[(rank-1+size)%size];
    node_t* root2=(node_t*)malloc(sizeof(node_t));
    root2->vector=(double*)malloc(d*sizeof(double));
    // recreate the tree
    empty_recv_Arrays(&root2,index_snd,dim_snd,mu_snd,B_holder,B_indexes,0,d,0,size_branches_n,true,mpi_size,mpi_size);
    root1=root2;
    // performing knn_extended for the new tree
    for (int i=0; i<m; i++){
      extended_kNN(root2,Y_p,i,d,k,k1,B_vector,B_indexes);
      k_size=0;
    }
    }
    /*
       Creating MPI_datatype knnresult in order to send to
       process with rank=0
    */
    knnresult kNN;
    // sending/receiving the fully updated for each subpart of X knnresult
    // gathering all knnresults to an array in process 0
    // and finally updating the knnresult of 0 for each knnresult
    // of other processes
    if (rank!=0){
      // k1 to process wwith rank=0.A problem occurs with dynamic memory
      // so in order to avoid that multiple messages are being send
      // in order to form the new k in process with rank=0

      MPI_Send(k1->nidx,k*m,MPI_INT,0,rank,MPI_COMM_WORLD);
      MPI_Send(k1->ndist,k*m,MPI_DOUBLE,0,rank,MPI_COMM_WORLD);
    }
    else{
      MPI_Status stat;
      kNN.m=n;
      kNN.k=k;
      kNN.nidx=(int*)malloc(k*n*sizeof(int));
      kNN.ndist=(double*)malloc(k*n*sizeof(double));

      int temp=0;
      for(int i=0; i<m*k; i++){
        kNN.nidx[temp]=k1->nidx[i];
        kNN.ndist[temp]=k1->ndist[i];
        temp++;
      }

      for (int i=0; i<size; i++){
        if (i==0)
          continue;
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

      printf("Time for V2 imlpementation in secs : %lf \n",( (double)ts_end.tv_sec+(double)ts_end.tv_nsec*pow(10,-9) - (double)ts_start.tv_sec -(double)ts_start.tv_nsec*pow(10,-9) ));

    }
    free(B_holder);
    free(B_indexes);
    free(k1->nidx);
    free(k1->ndist);
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
  /*              V2 STARTS HERE              */
  knnresult x=distrAllkNN_extended(X,n,d,k);










  /********************************************/
  /*             CLEARING MEMORY              */

  return 0;
}
