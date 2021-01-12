# Parallel_Distributed_Systems-Project_2

In this project an array of random elements,that form a matrix in row major format , is being treted as input and the output is the k-nearest neighbours of each vector.(A vector represents an element in a d-dimensional space.Every d elements of the array are the coordinates of a vector).The result is stored in a struct calles knnresult which consists of the number of query elements m,the number k and two matrixes,again in row major format,size m*k called nidx and ndist.Nidx has the indexes of the k-nearest neighbours and ndist has the k-minimum distances.

**V0.c** :
In V0.c the k nearest neighbours for each point in space d are calculated by taking advantage of CBLAS routines in order to calculate the matrix 
D=sqrt((X.*X)*e*inverse(e) -2*X*inverse(Y) + e*inverse(e)*(Y.*Y)) where Dij element is the distance between Xi and Yj element.Afterwards,a simple kmin search algortihm is being implemented to find the k-nearest neighbours of each point in set Y.

**V1.c** :
In V1.c the k-nearest neigbours are being found using the same method as V0.c .In this case though,the program is being divided in multiple processes using the MPI interface.Each process holds a piece of the Array X then performs a knnsearch in this set using the method implemented in V0.c .Afterwards,with the use of the function MPI scatterv and by holding the initital array a process can again perform knnresult but with a different subset of X,thus updating the previous result.This is implemented for all the processes p-1 times where p is the number of processes.

**V2.c** :
In V2.c the process of finding the k-nearest neighbours is being implemented with the use of a Vantage Point Tree(VPT).By holding a subset of the input array X,each process forms a VPT and performs an extended version of knnresult.This is done by initializing all the k-distances of a point Y_i with infinity and later compairing the array ndist with the distance of each node of the VPT and updating it.Then,the tree is sent to the other processes to perform their knnresult and update their arrays.
