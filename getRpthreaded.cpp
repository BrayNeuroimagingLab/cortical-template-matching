#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <sys/sysinfo.h>

using namespace std;

// Shared data!
double * T;
float * R;
size_t VOXELCOUNT;
size_t COLUMNS;

// Each thread given a struct
struct Params{
    size_t thread_N;
    size_t start_gmv;
    size_t end_gmv;
};

static inline float corr(size_t gmv1, size_t gmv2){
    double sum_X = 0;
    double sum_Y = 0;
    double squareSum_X = 0;
    double squareSum_Y = 0;
    double sum_XY = 0;
    for(size_t j = 0; j < COLUMNS; j++){
        sum_Y += *(T + (gmv1*COLUMNS) + j);
        sum_X += *(T + (gmv2*COLUMNS) + j);
        squareSum_X += *(T + (gmv2*COLUMNS) + j) * *(T + (gmv2*COLUMNS) + j);
        squareSum_Y += *(T + (gmv1*COLUMNS) + j) * *(T + (gmv1*COLUMNS) + j);
        sum_XY += *(T + (gmv1*COLUMNS) + j) * *(T + (gmv2*COLUMNS) + j);
    }
    return (float) (COLUMNS * sum_XY - sum_X * sum_Y) / sqrt((COLUMNS * squareSum_X - sum_X * sum_X) * (COLUMNS * squareSum_Y - sum_Y * sum_Y));
}

void * biteR(void * params){
    struct Params * P = ((struct Params *) params);
    for (size_t gmv = P->start_gmv; gmv <= P->end_gmv; gmv++){
        for (size_t j = 0; j < VOXELCOUNT; j++){
            float r = corr(gmv, j);
            *(R + (gmv*VOXELCOUNT) + j) = r;
        }
    }
    pthread_exit(NULL);
}

void initThreads(double * t, float * r, size_t voxelCount, size_t time){
    T = t;
    R = r;
    VOXELCOUNT = voxelCount;
    COLUMNS = time;
    int n_threads = get_nprocs();
    cout << "Number of threads found for use:  " << n_threads << "\n";

    const size_t thread_load = VOXELCOUNT / n_threads;
    cout << "Thread load:  " << thread_load << "  voxels \n";

    Params threadParams[n_threads];
    // Populate the parameter structs
    for(size_t i = 0; i < n_threads; i++){
        threadParams[i].thread_N = i;
        threadParams[i].start_gmv = i * thread_load;
        threadParams[i].end_gmv = threadParams[i].start_gmv + thread_load - 1;
        if (i == n_threads - 1)
            threadParams[i].end_gmv = VOXELCOUNT - 1;
    }

    // Execute the threads
    for(int i = 0; i < n_threads; i++){
        pthread_create(&threads[i], NULL, biteR, &threadParams[i]);
    }

    // wait for the threads to finish
    for (int i = 0; i < n_threads; i++){
        pthread_join(threads[i], NULL);
    }
    cout << "FINISHED C++ multithreading \n";

    return;
}
