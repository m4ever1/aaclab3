/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/
/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Sang-Ha  Lee											**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					Simplified for main functionality: regular k-means	**/
/**					clustering.											**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "kmeans.h"
#include <omp.h>

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

extern double wtime(void);

// __device__ void warpReduce(volatile int* shared_data, int tid) 
// {
//     shared_data[tid] += shared_data[tid + 32];
//     shared_data[tid] += shared_data[tid + 16];
//     shared_data[tid] += shared_data[tid + 8];
//     shared_data[tid] += shared_data[tid + 4];
//     shared_data[tid] += shared_data[tid + 2];
//     shared_data[tid] += shared_data[tid + 1];
// }

__device__ int reduce(volatile float* dists, int* ind_out)
{
    float min_dist = FLT_MAX;
    unsigned int i = threadIdx.y;
    // printf("id = %d\n", i);
    int ind=3;
    if(i < 2 && threadIdx.x == 0)
    {
        if(dists[i] < dists[i+2])
        {
            atomicExch(ind_out, i);
            // *ind_out = i;
        }
        else
        {
            dists[i] = dists[i+2];
            atomicExch(ind_out, i+2);
            // *ind_out = i+2;
        }
        __syncthreads();
        if(dists[i] < dists[!i])
        {
            if(dists[i] < dists[4])
            {
                return(ind);
            }
            else
            {
                atomicExch(ind_out, 4);
                // *ind_out = 4;
                return(4);
            }
        }
    }   
    return(ind);
}
    

__global__ void dowhileloop(float **feature, 
                            float **clusters,
                            int     nfeatures,
                            int     npoints,
                            int     nclusters,
                            float   threshold,
                            int*    membership,
                            int* new_centers_len,
                            float** new_centers,
                            float* delta)
{
    // printf("bruh\n");
    unsigned int i = blockIdx.x, j, k;
    extern __shared__ int ind[1];
    // int* ind = (int*) malloc(sizeof(int));
    j = threadIdx.x;
    k = threadIdx.y;
    extern __shared__ float dists_device[5];    
    /* find the ind of nestest cluster centers */
    //min_dist=FLT_MAX;

    // extern __device__ float delta;

    find_nearest_point(feature[i], nfeatures, clusters, nclusters, dists_device);
    __syncthreads();
    reduce(dists_device, ind);
    // ;
    
    __syncthreads();
    // *ind = (int) dists_device[0];
    // printf("point = %d, cluster = %d , feature = %d, ind = %d\n",i , k, j, ind);
    // cudaMemcpy(dists_host, dists_device, sizeof(float)*nclusters, cudaMemcpyDeviceToHost);
    // if(j == 0 && k == 0) 
    //     printf("POINT = %d, ind = %d\n", i,*ind);

    // printf("point = %d, cluster = %d , feature = %d, ind = %d\n",i , k, j, ind);
    /* if membership changes, increase delta by 1 */
    if (membership[i] != *ind)
    {
        atomicAdd(delta, 1.0);
        
    }
    /* assign the membership to object i */
    membership[i] = *ind;
    // printf("new_centers[ind][j] = %f\n", new_centers[ind][j]);
    /* update new cluster centers : sum of objects located within */
    new_centers_len[*ind]++;

    // __syncthreads();
    
    atomicAdd(&(new_centers[*ind][j]), feature[i][j]);
    // 

    __syncthreads();
    printf("delta inside = %f\n", *delta);
}

__device__ void find_nearest_point(float  *pt,          /* [nfeatures] */
                       int     nfeatures,
                       float **pts,         /* [npts][nfeatures] */ 
                       int     npts,
                       float*    dists)
{

    float min_dist=FLT_MAX;
    float* dist;
    int i = threadIdx.y;
    // if (i<npts) {
    dists[i] = 0;
    __syncthreads();
    euclid_dist_2(pt, pts[i], nfeatures, &(dists[i]));  /* no need square root */
    __syncthreads();
    if(blockIdx.x == 0 && threadIdx.x == 0)
        printf("point = %d, cluster = %d | dist = %f\n", blockIdx.x,i, dists[i]);
    // }
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
__device__ void euclid_dist_2(float *pt1,
                    float *pt2,
                    int    numdims, float* dist)
{
    int i = threadIdx.x;
    float ans=0.0;
    // printf("i = %f\n", dist);

    atomicAdd(dist, (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]));
    // *dist += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);
        // float ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);
    // *dist = ans;

}


/*----< kmeans_clustering() >---------------------------------------------*/
float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *membership) /* out: [npoints] */
{

    int      i, j, n=0, loop=0;
    int     *new_centers_len; /* [nclusters]: no. of points in each cluster */
    float   *delta;
    float  **clusters;   /* out: [nclusters][nfeatures] */
    float  **new_centers_host;     /* [nclusters][nfeatures] */
    float  **new_centers_device;
  

    /* allocate space for returning variable clusters[] */
    cudaMallocManaged(&clusters, nclusters *             sizeof(float*));
    cudaMallocManaged(clusters, nclusters * nfeatures * sizeof(float));
    for (i=1; i<nclusters; i++)
        clusters[i] = clusters[i-1] + nfeatures;

    /* randomly pick cluster centers */
    for (i=0; i<nclusters; i++) {
        //n = (int)rand() % npoints;
        for (j=0; j<nfeatures; j++)
            clusters[i][j] = feature[n][j];
		n++;
    }

    for (i=0; i<npoints; i++)
		membership[i] = -1;


    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    cudaMallocManaged(&new_centers_len, sizeof(int)*nclusters);
    cudaMemset(new_centers_len, 0,sizeof(int)*nclusters);
    // new_centers_len = (int*) calloc(nclusters, sizeof(int));

    cudaMallocManaged(&new_centers_device, nclusters *sizeof(float*));
    cudaMallocManaged(new_centers_device, nclusters * nfeatures * sizeof(float));
    // for (i=0; i<nclusters; i++)
    // {
    //     new_centers_device[i]
    // }

    // new_centers_host    = (float**) malloc(nclusters *            sizeof(float*));
    // new_centers_host[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
    cudaMemset(new_centers_device[0], 0,sizeof(float)*nclusters*nfeatures);
    for (i=1; i<nclusters; i++)
        new_centers_device[i] = new_centers_device[i-1] + nfeatures;

    // for (i = 0; i <nclusters; i++)
    // {

    //     cudaMemcpy(new_centers_device[i], new_centers_host[i], nclusters * nfeatures * sizeof(float), cudaMemcpyHostToDevice);
    // }    
    // cudaMemcpy(new_centers_device, new_centers_host, nclusters*sizeof(float*), cudaMemcpyHostToDevice);

 
    float * dists_device;
    float * dists_host;
    cudaMalloc(&dists_device, sizeof(float)*nclusters);
    dists_host = (float*) malloc(sizeof(float)*nclusters);
    int ind;
    float min_dist=FLT_MAX;
    FILE* fp = fopen("./delta_cuda.txt", "a");
    // printf("nclusters = %d, npoints = %d, nfeatures = %d\n", nclusters, npoints, nfeatures);
    dim3 blocks(npoints, nclusters);
    dim3 threads(nfeatures, nclusters);
    // float *delta_host, *delta_device;
    cudaMallocManaged(&delta, sizeof(float));

    // cudaMalloc(&delta_device, sizeof(float));
    // delta_host = (float*) malloc(sizeof(float));
    do{
        *delta = 0.0;
        dowhileloop<<<npoints, threads>>>(feature, clusters, nfeatures, npoints, nclusters, threshold, membership, new_centers_len, new_centers_device, delta);
        cudaDeviceSynchronize();
        for (i=0; i<nclusters; i++) {
            for (j=0; j<nfeatures; j++) {
                if (new_centers_len[i] > 0)
					clusters[i][j] = new_centers_device[i][j] / new_centers_len[i];
				new_centers_device[i][j] = 0.0;   /* set back to 0 */
			}
			new_centers_len[i] = 0;   /* set back to 0 */
        }
        printf("delta = %f\n", *delta);
    } while(*delta>threshold);

    // cudaMemcpy(delta_host, delta_device, sizeof(float), cudaMemcpyDeviceToHost);
    // fprintf(fp, "delta = %d\n", *delta_host);
    // printf("delta = %f\n", *delta_host);
    fclose(fp);
    cudaFree(dists_device);
    cudaFree(delta);
    free(dists_host);
    cudaFree(new_centers_device[0]);
    cudaFree(new_centers_device);
    cudaFree(new_centers_len);

    return clusters;
}

