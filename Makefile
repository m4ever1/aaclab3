# C compiler
HOST_CC = nvcc
HOST_CC_FLAGS = -g -O2

GEM5_CC = aarch64-linux-gcc
GEM5_CC_FLAGS = -g -O2 -static 
 
DATA_DIR = $(HOME)/apps/kmeans/data
GEM5_DIR = $(HOME)/gem5

compile_all: kmeans_gem5 kmeans_host

compile_host: kmeans_host

compile_gem5: kmeans_gem5

run_host: kmeans_host
	@echo "======================================================================"
	@echo "Running with kdd_cup... you may change the dataset"
	@echo "======================================================================"
	./kmeans_host -i $(DATA_DIR)/kdd_cup

run_gem5: kmeans_gem5
	@echo "======================================================================"
	@echo "WARNING: Running with a small data set (1000_34.txt)"
	@echo "change the makefile to use a larger dataset for the final results"
	@echo "======================================================================"
	$(GEM5_DIR)/build/ARM/gem5.opt $(GEM5_DIR)/configs/example/se.py --cpu-type=ex5_LITTLE --caches --svevl=1 -c kmeans_gem5 -o '-i $(DATA_DIR)/1000_34.txt'

########### HOST SIDE (X86) COMPILATION ###########
kmeans_host: cluster_host.o getopt_host.o kmeans_host.o kmeans_clustering_host.o 
	$(HOST_CC) $(HOST_CC_FLAGS) cluster_host.o getopt_host.o kmeans_host.o kmeans_clustering_host.o  -o kmeans_host

#%.o: %.[ch]
#	$(HOST_CC) $(HOST_CC_FLAGS) $< -c

cluster_host.o: cluster.cu 
	$(HOST_CC) $(HOST_CC_FLAGS) -c cluster.cu -o cluster_host.o
	
getopt_host.o: getopt.c 
	$(HOST_CC) $(HOST_CC_FLAGS) -c getopt.c -o getopt_host.o
	
kmeans_host.o: kmeans.cu 
	$(HOST_CC) $(HOST_CC_FLAGS) -c kmeans.cu -o kmeans_host.o

kmeans_clustering_host.o: kmeans_clustering.cu kmeans.h
	$(HOST_CC) $(HOST_CC_FLAGS) -c kmeans_clustering.cu -o kmeans_clustering_host.o

########### GEM5 SIDE (ARM) COMPILATION ###########
kmeans_gem5: cluster_gem5.o getopt_gem5.o kmeans_gem5.o kmeans_clustering_gem5.o 
	$(GEM5_CC) $(GEM5_CC_FLAGS) cluster_gem5.o getopt_gem5.o kmeans_gem5.o kmeans_clustering_gem5.o  -o kmeans_gem5

#%_gem5.o: %_gem5.[ch]
#	$(GEM5_CC) $(GEM5_CC_FLAGS) -static $< -c

cluster_gem5.o: cluster.c 
	$(GEM5_CC) $(GEM5_CC_FLAGS) -c cluster.c -o cluster_gem5.o
	
getopt_gem5.o: getopt.c 
	$(GEM5_CC) $(GEM5_CC_FLAGS) -c getopt.c -o getopt_gem5.o
	
kmeans_gem5.o: kmeans.c 
	$(GEM5_CC) $(GEM5_CC_FLAGS) -c kmeans.c -o kmeans_gem5.o

kmeans_clustering_gem5.o: kmeans_clustering.c kmeans.h
	$(GEM5_CC) $(GEM5_CC_FLAGS) -c kmeans_clustering.c -o kmeans_clustering_gem5.o

clean:
	rm -f *.o *~ kmeans_host kmeans_gem5
