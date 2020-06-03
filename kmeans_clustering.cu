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

__global__ void find_nearest_point(float  *pt,          /* [nfeatures] */
                       int     nfeatures,
                       float **pts,         /* [npts][nfeatures] */ 
                       int     npts,
                       float*    dists)
{
    //  fp;
    // fp = fopen("./ans_cuda.txt", "a");
    float min_dist=FLT_MAX;
    float* dist;
    // dist = (float*) malloc(sizeof(float)*npts);
    // cudaMallocManaged(&dist, sizeof(float));
    /* find the cluster center id with min distance to pt */
    int i = blockIdx.x;
    // if (i<npts) {
        dists[i] = 0;
        euclid_dist_2(pt, pts[i], nfeatures, &(dists[i]));  /* no need square root */
        
        // fprintf(fp, "ans = %f\n", *dist);
        // atomicMax(&(dist[i]), min_dist);
        // atomicExch(&min_dist, dist[i]);
        // *index = 2;
        // if (dist[i] != ) 
        // {
        //     min_dist = dist[j];
        //     *index    = j;
        // }

    // }
    // ;
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
    for (i=0; i<numdims; i++)
    {
        // atomicAdd(dist, (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]));
        *dist += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);
    }
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
    float    delta;
    float  **clusters;   /* out: [nclusters][nfeatures] */
    float  **new_centers;     /* [nclusters][nfeatures] */
  

    /* allocate space for returning variable clusters[] */
    cudaMallocManaged(&clusters, nclusters *             sizeof(float*));
    cudaMallocManaged(clusters, nclusters * nfeatures * sizeof(float));
    // clusters    = (float**) malloc(nclusters *             sizeof(float*));
    // clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
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
    new_centers_len = (int*) calloc(nclusters, sizeof(int));

    new_centers    = (float**) malloc(nclusters *            sizeof(float*));
    new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
    for (i=1; i<nclusters; i++)
        new_centers[i] = new_centers[i-1] + nfeatures;
 
    float * dists_device;
    float * dists_host;
    cudaMalloc(&dists_device, sizeof(float)*nclusters);
    dists_host = (float*) malloc(sizeof(float)*nclusters);
    int index;
    float min_dist=FLT_MAX;
    FILE* fp = fopen("./index_cuda.txt", "a");
    do {
        
        delta = 0.0;
        
        for (i=0; i<npoints; i++) {
            /* find the index of nestest cluster centers */
            min_dist=FLT_MAX;
	        find_nearest_point<<<nclusters, nfeatures>>>(feature[i], nfeatures, clusters, nclusters, dists_device);
            cudaDeviceSynchronize();
            cudaMemcpy(dists_host, dists_device, sizeof(float)*nclusters, cudaMemcpyDeviceToHost);
            for(int k=0;k<nclusters;k++)
            {
                if (dists_host[k] < min_dist) 
                {
                    min_dist = dists_host[k];
                    index    = k;
                    // printf("%f\n", dists_host[k]);
                }
            }
            fprintf(fp, "index = %d\n", index);
            /* if membership changes, increase delta by 1 */
	        if (membership[i] != index) delta += 1.0;

	        /* assign the membership to object i */
	        membership[i] = index;

	        /* update new cluster centers : sum of objects located within */
	        new_centers_len[index]++;
	        for (j=0; j<nfeatures; j++)          
				new_centers[index][j] += feature[i][j];
        }
      

	/* replace old cluster centers with new_centers */
        for (i=0; i<nclusters; i++) {
            for (j=0; j<nfeatures; j++) {
                if (new_centers_len[i] > 0)
					clusters[i][j] = new_centers[i][j] / new_centers_len[i];
				new_centers[i][j] = 0.0;   /* set back to 0 */
			}
			new_centers_len[i] = 0;   /* set back to 0 */
		}
            
        //delta /= npoints;
    } while (delta > threshold);
    fclose(fp);
    cudaFree(dists_device);
    free(dists_host);
    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}

