CC= gcc
MPICC= mpicc
CFLAGS = -lopenblas -lpthread -lm
OPTFLAGS= -O3


all: mpi_scatter_example.c mpi_scatterv_rotation_example.c mpi_send_recv_knn.c update_knn.c median_findRandom_example.c

	$(MPICC) $(OPTFLAGS) mpi_scatter_example.c -o mpi_scatter_example $(CFLAGS)

	$(MPICC) $(OPTFLAGS) mpi_scatterv_rotation_example.c -o mpi_scatterv_rotation_example $(CFLAGS)

	$(MPICC) $(OPTFLAGS) mpi_send_recv_knn.c -o mpi_send_recv_knn $(CFLAGS)

	$(CC) $(OPTFLAGS) update_knn.c -o update_knn $(CFLAGS)

	$(CC) $(OPTFLAGS) median_findRandom_example.c -o median_findRandom_example $(CFLAGS)



clean:
	rm -f mpi_scatter_example mpi_scatterv_rotation_example mpi_send_recv_knn update_knn median_findRandom_example
