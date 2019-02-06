#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Do not modify this function. A constraint of this exercise is
 * that it remain a host function.
 */

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

/*
 * This function calculates the gravitational impact of all bodies in the system
 * on all others, but does not update their positions.
 */
__global__
void bodyForce_sum0(Body *p,float * px,float* py,float* pz, float dt, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int idx = index; idx < n*n; idx += stride)
  {
		int i=idx/n;
		int j=idx%n;

    float dx = p[j].x - p[i].x;
    float dy = p[j].y - p[i].y;
    float dz = p[j].z - p[i].z;
    float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
    float invDist = rsqrtf(distSqr);
    float invDist3 = invDist * invDist * invDist;
		px[idx]=dx*invDist3;
		py[idx]=dy*invDist3;
		pz[idx]=dz*invDist3;
	}
}

__global__
void bodyForce_sum1(Body *p,float * px,float* py,float* pz, float dt, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for(int idx = index; idx < n; idx += stride) {
		for(int i = idx*n+1;i < (idx+1)*n;i++) {
			px[idx*n]+=px[i];
			py[idx*n]+=py[i];
			pz[idx*n]+=pz[i];
		}
	}
}

__global__
void bodyForce_sum2(Body *p,float * px,float* py,float* pz, float dt, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for(int idx = index; idx < n; idx += stride) {
		int i=idx;
		p[i].vx += dt*px[i*n];
		p[i].vy += dt*py[i*n];
		p[i].vz += dt*pz[i*n];
	}

}

__global__
void bodyForce_sum3(Body *p,float * px,float* py,float* pz, float dt, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for(int idx = index; idx < n; idx += stride) {
		int i=idx;
		p[i].x += p[i].vx*dt;
		p[i].y += p[i].vy*dt;
		p[i].z += p[i].vz*dt;
	}

}

void bodyForce(Body *p,float * px,float *py,float *pz, float dt, int n,int numberOfBlocks,int threadsPerBlock) {
    bodyForce_sum0<<<numberOfBlocks,threadsPerBlock>>>(p,px,py,pz, dt, n); // compute interbody forces
//  for (int i = 0; i < n; ++i) {
//    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
//
//    for (int j = 0; j < n; j++) {
//      float dx = p[j].x - p[i].x;
//      float dy = p[j].y - p[i].y;
//      float dz = p[j].z - p[i].z;
//      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
//      float invDist = rsqrtf(distSqr);
//      float invDist3 = invDist * invDist * invDist;
//
//      px[i*n+j]= dx * invDist3; py[i*n+j] = dy * invDist3; pz[i*n+j] = dz * invDist3;
//		}
//	}

		bodyForce_sum1<<<numberOfBlocks,threadsPerBlock>>>(p,px,py,pz, dt, n); // compute interbody forces

//	for (int i = 0; i < n; ++i) {
//		for (int j = 1; j < n; j++) {
//      px[i*n] += px[i*n+j]; py[i*n] += py[i*n+j]; pz[i*n] += pz[i*n+j];
//    }
//	}



		bodyForce_sum2<<<numberOfBlocks,threadsPerBlock>>>(p,px,py,pz, dt, n); // compute interbody forces
//	for (int i = 0; i < n; ++i) {
//    p[i].vx += dt*px[i*n]; p[i].vy += dt*py[i*n]; p[i].vz += dt*pz[i*n];
//  }


//  for (int i = 0; i < n; ++i) {
//    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
//
//    for (int j = 0; j < n; j++) {
//      float dx = p[j].x - p[i].x;
//      float dy = p[j].y - p[i].y;
//      float dz = p[j].z - p[i].z;
//      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
//      float invDist = rsqrtf(distSqr);
//      float invDist3 = invDist * invDist * invDist;
//
//      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
//    }
//
//    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
//  }
}

int main(const int argc, const char** argv) {

  /*
   * Do not change the value for `nBodies` here. If you would like to modify it,
   * pass values into the command line.
   */

  int nBodies = 2<<11;
  int salt = 0;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  /*
   * This salt is for assessment reasons. Tampering with it will result in automatic failure.
   */

  if (argc > 2) salt = atoi(argv[2]);




	// find out the GPU setting
	  int deviceId;
	  int numberOfSMs;
		cudaGetDevice(&deviceId);
		cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	  int threadsPerBlock = 256;
	  int numberOfBlocks = 32 * numberOfSMs;


  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;
	float *px;
	float *py;
	float *pz;

//  buf = (float *)malloc(bytes);
	cudaMallocManaged(&buf,bytes);
	cudaMallocManaged(&px,nBodies*nBodies*sizeof(float));
	cudaMallocManaged(&py,nBodies*nBodies*sizeof(float));
	cudaMallocManaged(&pz,nBodies*nBodies*sizeof(float));

  Body *p = (Body*)buf;

  /*
   * As a constraint of this exercise, `randomizeBodies` must remain a host function.
   */

  randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

  double totalTime = 0.0;

  /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

  /*******************************************************************/
  // Do not modify these 2 lines of code.
  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();
  /*******************************************************************/

  /*
   * You will likely wish to refactor the work being done in `bodyForce`,
   * as well as the work to integrate the positions.
   */

    bodyForce(p,px,py,pz, dt, nBodies,numberOfBlocks,threadsPerBlock); // compute interbody forces
		bodyForce_sum3<<<numberOfBlocks,threadsPerBlock>>>(p,px,py,pz, dt, nBodies); // compute interbody forces
	//cudaDeviceSynchronize();
  //  for (int i = 0 ; i < nBodies; i++) { // integrate position
  //    p[i].x += p[i].vx*dt;
  //    p[i].y += p[i].vy*dt;
  //    p[i].z += p[i].vz*dt;
  //  }

  /*
   * This position integration cannot occur until this round of `bodyForce` has completed.
   * Also, the next round of `bodyForce` cannot begin until the integration is complete.
   */

  /*******************************************************************/
  // Do not modify the code in this section.
    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }
  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
	cudaDeviceSynchronize();

#ifdef ASSESS
  checkPerformance(buf, billionsOfOpsPerSecond, salt);
#else
  checkAccuracy(buf, nBodies);
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
  salt += 1;
#endif
  /*******************************************************************/

  /*
   * Feel free to modify code below.
   */

//  free(buf);
	cudaFree(buf);
	cudaFree(px);
	cudaFree(py);
	cudaFree(pz);
}
