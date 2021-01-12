CC= gcc
MPICC= mpicc
CFLAGS = -lopenblas -lpthread -lm
OPTFLAGS= -O3


all: V0.c V1.c V2.c
	$(CC) $(OPTFLAGS) V0.c -o V0 $(CFLAGS)

	$(MPICC) $(OPTFLAGS) V1.c -o V1 $(CFLAGS)

	$(MPICC) $(OPTFLAGS) V2.c -o V2 $(CFLAGS)


clean:
	rm -f V0 V1 V2
