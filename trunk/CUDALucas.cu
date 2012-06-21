char program[] = "CUDALucas v2.04 Beta";
/* CUDALucas.c
   Shoichiro Yamada Oct. 2010 

   This is an adaptation of Richard Crandall lucdwt.c, John Sweeney MacLucasUNIX.c,
   and Guillermo Ballester Valor MacLucasFFTW.c code.
   Improvement From Prime95.
   
   It also contains mfaktc code by Oliver Weihe and Eric Christenson
   adapted for CUDALucas use. Such code is under the GPL, and is noted as such.
*/

/* Include Files */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <signal.h>
#ifndef _MSC_VER
#include <sys/types.h>
#include <sys/stat.h>
#else
#include <direct.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "cuda_safecalls.h"
#include "parse.h"

/* In order to have the gettimeofday() function, you need these includes on Linux:
#include <sys/time.h>
#include <unistd.h>
On Windows, you need 
#include <winsock2.h> and a definition for
int gettimeofday (struct timeval *tv, struct timezone *) {}
Both platforms are taken care of in parse.h and parse.c. */

/************************ definitions ************************************/
/* global variables needed */
double *two_to_phi, *two_to_minusphi;
double *g_ttp, *g_ttmp;
char *g_numbits;
int *g_mask;
float *g_inv2, *g_inv3;
double *g_ttp2, *g_ttmp2, *g_ttp3, *g_ttmp3;
double high, low, highinv, lowinv;
double Gsmall, Gbig, Hsmall, Hbig;
cufftHandle plan_fw, plan_bw, plan;
double *g_x, *g_save;
int j_save;
float *g_err;
int *g_carry;
int *ip, quitting, checkpoint_iter, b, c, fftlen, s_f, t_f, r_f, d_f, k_f;
int threads, polite, polite_f, bad_selftest=0;
char folder[132];
char input_filename[132], RESULTSFILE[132];
char INIFILE[132] = "CUDALucas.ini";
char AID[132]; // Assignment key
char s_residue[32];

/* http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html
   base code from Takuya OOURA.  */
__global__ void
rftfsub_kernel (int n, double *a)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  double wkr, wki, xr, xi, yr, yi, cc, d, aj, aj1, ak, ak1, *c;
  double new_aj, new_aj1, new_ak, new_ak1;
  const int m = n >> 1;
  const int nc = n >> 2;
  const int j = threadID << 1;
  const int j2 = threadID;
  c = &a[n];
  if (threadID)
    {
      int nminusj = n - j;

      wkr = 0.5 - c[nc - j2];
      wki = c[j2];
      aj = a[j];
      aj1 = a[1 + j];
      ak = a[nminusj];
      ak1 = a[1 + nminusj];
      xr = aj - ak;
      xi = aj1 + ak1;
      yr = wkr * xr - wki * xi;
      yi = wkr * xi + wki * xr;
      aj -= yr;
      aj1 -= yi;
      ak += yr;
      ak1 -= yi;

      new_aj1 = 2.0 * aj * aj1;
      new_aj = (aj - aj1) * (aj + aj1);

      new_ak1 = 2.0 * ak * ak1;
      new_ak = (ak - ak1) * (ak + ak1);

      xr = new_aj - new_ak;
      xi = new_aj1 + new_ak1;
      yr = wkr * xr + wki * xi;
      yi = wkr * xi - wki * xr;

      a[j] = new_aj - yr;
      a[1 + j] = yi - new_aj1;
      a[nminusj] = new_ak + yr;
      a[1 + nminusj] = yi - new_ak1;
    }
  else
    {
      xi = a[0] - a[1];
      a[0] += a[1];
      a[1] = xi;
      a[0] *= a[0];
      a[1] *= a[1];
      a[1] = 0.5 * (a[0] - a[1]);
      a[0] -= a[1];
      a[1] = -a[1];
      cc = a[0 + m];
      d = -a[1 + m];
      a[1 + m] = -2.0 * cc * d;
      a[0 + m] = (cc + d) * (cc - d);
      a[1 + m] = -a[1 + m];
    }
}

__global__ void
copy_kernel (double *save, double *x)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  save[threadID] = x[threadID];
}

void
rdft (int n, int isgn, double *a, int *ip)
{
  void makect (int nc, int *ip, double *c);
  const int nc = n >> 2;
  int nw = ip[0];
  if (nw == 0)
    {
      makect (nc, ip, &a[n]);
      cutilSafeCall (cudaMemcpy
		     (g_x, a, sizeof (double) * (n / 4 * 5),
		      cudaMemcpyHostToDevice));
      if (t_f)
	copy_kernel <<< n / 128, 128 >>> (g_save, g_x);
    }
  cufftSafeCall (cufftExecZ2Z
		 (plan, (cufftDoubleComplex *) g_x,
		  (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
  rftfsub_kernel <<< n / 512, 128 >>> (n, g_x);
  cufftSafeCall (cufftExecZ2Z
		 (plan, (cufftDoubleComplex *) g_x,
		  (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
  return;
}

/* -------- initializing routines -------- */
void
makect (int nc, int *ip, double *c)
{
  int j;
  const int nch = nc >> 1;
  double delta;
  ip[0] = 1;
  ip[1] = nc;
  if (nc > 1)
    {
      delta = atan (1.0) / nch;
      c[0] = cos (delta * nch);
      c[nch] = 0.5 * c[0];
      for (j = 1; j < nch; j++)
	{
	  c[j] = 0.5 * cos (delta * j);
	  c[nc - j] = 0.5 * sin (delta * j);
	}
    }
}

/**************************************************************
 *
 *      FFT and other related Functions
 *
 **************************************************************/
/* rint is not ANSI compatible, so we need a definition for 
 * WIN32 and other platforms with rint.
 * Also we use that to write the trick to rint()
 */
# define RINT_x86(x) (floor(x+0.5))
# define RINT(x)  __rintd(x)
__device__ static double
__rintd (double x)
{
  double y;
asm ("cvt.rni.f64.f64 %0, %1;": "=d" (y):"d" (x));
  return (y);
}

#ifdef _MSC_VER
long long int __double2ll (double);
#endif
__device__ static long long int
__double2ll (double x)
{
  long long int y;
asm ("cvt.rni.s64.f64 %0, %1;": "=l" (y):"d" (x));
  return (y);
}

/****************************************************************************
 *           Lucas Test - specific routines                                 *
 ***************************************************************************/
void
init_lucas (double *x, int q, int n)
{
  int j, qn, a, i, done;
  int size0, bj;
  double log2 = log (2.0);
  double ttp, ttmp;
  double *s_ttp, *s_ttmp;
  int *s_mask;
  float *s_inv;
  float *s_inv2;
  float *s_inv3;
  double *s_ttp2, *s_ttmp2;
  double *s_ttp3, *s_ttmp3;
  char *s_numbits;
  float *s_ttmpp;
  two_to_phi = (double *) malloc (sizeof (double) * (n / 2));
  two_to_minusphi = (double *) malloc (sizeof (double) * (n / 2));
  s_mask = (int *) malloc (sizeof (int) * 32);
  s_inv = (float *) malloc (sizeof (float) * (n));
  s_ttp = (double *) malloc (sizeof (double) * (n));
  s_ttmp = (double *) malloc (sizeof (double) * (n));
  s_ttmpp = (float *) malloc (sizeof (float) * (n));
  s_numbits = (char *) malloc (sizeof (char) * (n));
  s_inv2 = (float *) malloc (sizeof (float) * (n / threads));
  s_ttp2 = (double *) malloc (sizeof (double) * (n / threads));
  s_ttmp2 = (double *) malloc (sizeof (double) * (n / threads));
  s_inv3 = (float *) malloc (sizeof (float) * (n / threads));
  s_ttp3 = (double *) malloc (sizeof (double) * (n / threads));
  s_ttmp3 = (double *) malloc (sizeof (double) * (n / threads));
  cufftSafeCall (cufftPlan1d (&plan, n / 2, CUFFT_Z2Z, 1));
  cutilSafeCall (cudaMalloc ((void **) &g_x, sizeof (double) * (n / 4 * 5)));
  if (t_f)
    cutilSafeCall (cudaMalloc ((void **) &g_save, sizeof (double) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_err, sizeof (float)));
  cutilSafeCall (cudaMalloc ((void **) &g_carry, sizeof (int) * n / threads));
  cutilSafeCall (cudaMalloc ((void **) &g_mask, sizeof (int) * 32));
  cutilSafeCall (cudaMalloc ((void **) &g_ttp, sizeof (double) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_ttmp, sizeof (double) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_numbits, sizeof (char) * n));
  cutilSafeCall (cudaMalloc
		 ((void **) &g_inv2, sizeof (float) * n / threads));
  cutilSafeCall (cudaMalloc
		 ((void **) &g_ttp2, sizeof (double) * n / threads));
  cutilSafeCall (cudaMalloc
		 ((void **) &g_ttmp2, sizeof (double) * n / threads));
  cutilSafeCall (cudaMalloc
		 ((void **) &g_inv3, sizeof (float) * n / threads));
  cutilSafeCall (cudaMalloc
		 ((void **) &g_ttp3, sizeof (double) * n / threads));
  cutilSafeCall (cudaMalloc
		 ((void **) &g_ttmp3, sizeof (double) * n / threads));
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
  low = floor ((exp (floor ((double) q / n) * log2)) + 0.5);
  high = low + low;
  lowinv = 1.0 / low;
  highinv = 1.0 / high;
  b = q % n;
  c = n - b;
  two_to_phi[0] = 1.0;
  two_to_minusphi[0] = 1.0 / (double) (n);
  qn = (b * 2) % n;
  for (i = 1, j = 2; j < n; j += 2, i++)
    {
      a = n - qn;
      two_to_phi[i] = exp (a * log2 / n);
      two_to_minusphi[i] = 1.0 / (two_to_phi[i] * n);
      qn += b * 2;
      qn %= n;
    }
  Hbig = exp (c * log2 / n);
  Gbig = 1 / Hbig;
  done = 0;
  j = 0;
  while (!done)
    {
      if (!((j * b) % n >= c || j == 0))
	{
	  a = n - ((j + 1) * b) % n;
	  i = n - (j * b) % n;
	  Hsmall = exp (a * log2 / n) / exp (i * log2 / n);
	  Gsmall = 1 / Hsmall;
	  done = 1;
	}
      j++;
    }
  bj = n;
  size0 = 1;
  bj = n - 1 * b;
  for (j = 0, i = 0; j < n; j = j + 2, i++)
    {
      ttmp = two_to_minusphi[i];
      ttp = two_to_phi[i];
      bj += b;
      bj = bj % n;
      size0 = (bj >= c);
      if (j == 0)
	size0 = 1;
      s_ttmp[j] = ttmp * 2.0;
      s_ttmpp[j] = (float) ttmp * n;
      if (size0)
	{
	  s_inv[j] = (float) highinv;
	  ttmp *= Gbig;
	  s_ttp[j] = ttp * high;
	  ttp *= Hbig;
	}
      else
	{
	  s_inv[j] = (float) lowinv;
	  ttmp *= Gsmall;
	  s_ttp[j] = ttp * low;
	  ttp *= Hsmall;
	}
      s_ttmpp[j] *= (float) s_ttp[j];
      bj += b;
      bj = bj % n;
      size0 = (bj >= c);
      if (j == (n - 2))
	size0 = 0;
      s_ttmp[j + 1] = ttmp * -2.0;
      s_ttmpp[j + 1] = (float) ttmp * n;
      if (size0)
	{
	  s_inv[j + 1] = (float) highinv;
	  s_ttp[j + 1] = ttp * high;
	}
      else
	{
	  s_inv[j + 1] = (float) lowinv;
	  s_ttp[j + 1] = ttp * low;
	}
      s_ttmpp[j + 1] *= (float) s_ttp[j + 1];
    }
  for (i = 0; i < n; i++)
    {
      s_ttmpp[i] = (float) ((long) (s_ttmpp[i] + 0.5));
      if (s_ttmpp[i] == s_ttmpp[0])
	s_numbits[i] = q / n + 1;
      else
	s_numbits[i] = q / n;
    }
  {
    for (i = 0; i < 32; i++)
      s_mask[i] = -1 << i;
    cudaMemcpy (g_mask, s_mask, sizeof (int) * 32, cudaMemcpyHostToDevice);
    cudaMemcpy (g_ttmp, s_ttmp, sizeof (double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy (g_numbits, s_numbits, sizeof (char) * n,
		cudaMemcpyHostToDevice);
  }
  for (i = 0, j = 0; i < n; i++)
    {
      if ((i % threads) == 0)
	{
	  s_inv2[j] = s_inv[i];
	  s_ttp2[j] = s_ttp[i];
	  s_ttmp2[j] = s_ttmp[i] * 0.5 * n;
	  s_inv3[j] = s_inv[i + 1];
	  s_ttp3[j] = s_ttp[i + 1];
	  s_ttmp3[j] = s_ttmp[i + 1] * (-0.5) * n;
	  j++;
	}
    }
  for (i = 0, j = 0; i < n; i++)
    s_ttp[i] *= s_inv[i];
  cudaMemcpy (g_ttp, s_ttp, sizeof (double) * n, cudaMemcpyHostToDevice);
  cudaMemcpy (g_inv2, s_inv2, sizeof (float) * n / threads,
	      cudaMemcpyHostToDevice);
  cudaMemcpy (g_ttp2, s_ttp2, sizeof (double) * n / threads,
	      cudaMemcpyHostToDevice);
  cudaMemcpy (g_ttmp2, s_ttmp2, sizeof (double) * n / threads,
	      cudaMemcpyHostToDevice);
  cudaMemcpy (g_inv3, s_inv3, sizeof (float) * n / threads,
	      cudaMemcpyHostToDevice);
  cudaMemcpy (g_ttp3, s_ttp3, sizeof (double) * n / threads,
	      cudaMemcpyHostToDevice);
  cudaMemcpy (g_ttmp3, s_ttmp3, sizeof (double) * n / threads,
	      cudaMemcpyHostToDevice);

  free ((char *) s_inv);
  free ((char *) s_ttp);
  free ((char *) s_mask);
  free ((char *) s_ttmp);
  free ((char *) s_ttmpp);
  free ((char *) s_inv2);
  free ((char *) s_ttp2);
  free ((char *) s_ttmp2);
  free ((char *) s_inv3);
  free ((char *) s_ttp3);
  free ((char *) s_ttmp3);
  free ((char *) s_numbits);

  ip = (int *) malloc (((size_t) (2 + sqrt ((float) n / 2)) * sizeof (int)));
  ip[0] = 0;
}

void
close_lucas (double *x)
{
  free ((char *) x);
  free ((char *) two_to_phi);
  free ((char *) two_to_minusphi);
  free ((char *) ip);
  cutilSafeCall (cudaFree ((char *) g_x));
  if (t_f)
    cutilSafeCall (cudaFree ((char *) g_save));
  cutilSafeCall (cudaFree ((char *) g_err));
  cutilSafeCall (cudaFree ((char *) g_carry));
  cutilSafeCall (cudaFree ((char *) g_mask));
  cutilSafeCall (cudaFree ((char *) g_ttp));
  cutilSafeCall (cudaFree ((char *) g_ttmp));
  cutilSafeCall (cudaFree ((char *) g_numbits));
  cutilSafeCall (cudaFree ((char *) g_inv2));
  cutilSafeCall (cudaFree ((char *) g_ttp2));
  cutilSafeCall (cudaFree ((char *) g_ttmp2));
  cutilSafeCall (cudaFree ((char *) g_inv3));
  cutilSafeCall (cudaFree ((char *) g_ttp3));
  cutilSafeCall (cudaFree ((char *) g_ttmp3));
  cufftSafeCall (cufftDestroy (plan));
}

void set_err_to_0(float* maxerr)
{
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
  *maxerr = 0.0;
}

template < int g_err_flag > __global__ void
normalize_kernel (double *g_x, int threads,
		  volatile float *g_err, int *g_carry, int *g_mask,
		  double *g_ttp, double *g_ttmp, char *g_numbits,
		  float maxerr)
{
  long long int bigint;
  int val, numbits, mask, shifted_carry;
  __shared__ int carry[1024 + 1];
  // read the matrix tile into shared memory
  unsigned int index = blockIdx.x * threads + threadIdx.x;
  if (g_err_flag)
    {
      double tval, trint;
      float ferr;
//0
      tval = g_x[index] * g_ttmp[index];
      trint = RINT (tval);
      ferr = tval - trint;
      ferr = fabs (ferr);

      bigint = trint;

      if (ferr > maxerr) 
	   atomicMax((int*)g_err, __float_as_int(ferr));
    }
  else
    {
//0
      bigint = __double2ll (g_x[index] * g_ttmp[index]);
    }

  numbits = g_numbits[index];
  carry[threadIdx.x + 1] = (int) (bigint >> numbits);
  mask = g_mask[numbits];
  val = ((int) bigint) & ~mask;

//1    
  __syncthreads ();
  if (threadIdx.x)
    val += carry[threadIdx.x];
  shifted_carry = val - g_mask[numbits - 1];
  carry[threadIdx.x] = shifted_carry >> numbits;
  val = val - (shifted_carry & mask);

  if (threadIdx.x == (threads - 1))
    g_carry[blockIdx.x] = carry[threadIdx.x + 1] + carry[threadIdx.x];

//2
  __syncthreads ();
  if (threadIdx.x)
    val += carry[threadIdx.x - 1];
  g_x[index] = (double) val *g_ttp[index];

}

__global__ void
normalize2_kernel (double *g_x, int threads, int *g_carry,
		   int g_N, float *g_inv2, double *g_ttp2, double *g_ttmp2,
		   float *g_inv3, double *g_ttp3, double *g_ttmp3)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = threads * threadID;
  double temp0, tempErr;
  double temp1, tempErr2;
  int carry;
  if (j < g_N)
    {
      if (threadID)
	carry = g_carry[threadID - 1];
      else
	carry = g_carry[g_N / threads - 1] - 2;	// The -2 is part of the LL test
      temp0 = g_x[j];
      temp1 = g_x[j + 1];
      tempErr = temp0 * g_ttmp2[threadID];
      tempErr2 = temp1 * g_ttmp3[threadID];
      temp0 = tempErr + carry;
      temp0 *= g_inv2[threadID];
      carry = RINT (temp0);
      temp1 = tempErr2 + carry;
      temp1 *= g_inv3[threadID];
      g_x[j] = (temp0 - carry) * g_ttp2[threadID];
      g_x[j + 1] = temp1 * g_ttp3[threadID];
    }
}

double
last_normalize (double *x, int N, int err_flag)
{
  int i, j, k, bj, size0;
  double hi = high, hiinv = highinv, lo = low, loinv = lowinv, temp0, tempErr;
  double err = 0.0, terr = 0.0, ttmpSmall = Gsmall, ttmpBig =
    Gbig, ttmp, carry;
  carry = -2.0;			/* this is the -2 of the LL x*x - 2 */
  bj = N;
  size0 = 1;
  for (j = 0, i = 0; j < N; j += 2, i++)
    {
      ttmp = two_to_minusphi[i];
      temp0 = x[j];
      temp0 *= 2.0;
      tempErr = RINT_x86 (temp0 * ttmp);
      if (err_flag)
	{
	  terr = fabs (temp0 * ttmp - tempErr);
	  if (terr > err)
	    err = terr;
	}
      temp0 = tempErr + carry;
      if (size0)
	{
	  temp0 *= hiinv;
	  carry = RINT_x86 (temp0);
	  bj += b;
	  ttmp *= ttmpBig;
	  if (bj >= N)
	    bj -= N;
	  x[j] = (temp0 - carry) * hi;
	  size0 = (bj >= c);
	}
      else
	{
	  temp0 *= loinv;
	  carry = RINT_x86 (temp0);
	  bj += b;
	  ttmp *= ttmpSmall;
	  if (bj >= N)
	    bj -= N;
	  x[j] = (temp0 - carry) * lo;
	  size0 = (bj >= c);
	}
      temp0 = x[j + 1];
      temp0 *= -2.0;

      if (j == N - 2)
	size0 = 0;
      tempErr = RINT_x86 (temp0 * ttmp);
      if (err_flag)
	{
	  terr = fabs (temp0 * ttmp - tempErr);
	  if (terr > err)
	    err = terr;
	}
      temp0 = tempErr + carry;
      if (size0)
	{
	  temp0 *= hiinv;
	  carry = RINT_x86 (temp0);
	  bj += b;
	  ttmp *= ttmpBig;
	  if (bj >= N)
	    bj -= N;
	  x[j + 1] = (temp0 - carry) * hi;
	  size0 = (bj >= c);
	}
      else
	{
	  temp0 *= loinv;
	  carry = RINT_x86 (temp0);
	  bj += b;
	  ttmp *= ttmpSmall;
	  if (bj >= N)
	    bj -= N;
	  x[j + 1] = (temp0 - carry) * lo;
	  size0 = (bj >= c);
	}
    }
  bj = N;
  k = 0;
  while (carry != 0)
    {
      size0 = (bj >= c);
      bj += b;
      temp0 = (x[k] + carry);
      if (bj >= N)
	bj -= N;
      if (size0)
	{
	  temp0 *= hiinv;
	  carry = RINT_x86 (temp0);
	  x[k] = (temp0 - carry) * hi;
	}
      else
	{
	  temp0 *= loinv;
	  carry = RINT_x86 (temp0);
	  x[k] = (temp0 - carry) * lo;
	}
      k++;
    }
  return (err);
}

double
lucas_square (double *x, int N, int iter, int last, float* maxerr,
	      int error_flag)
{
  double terr;
  rdft (N, 1, x, ip);
  if (iter == last)
    {
      cutilSafeCall (cudaMemcpy
		     (x, g_x, sizeof (double) * N, cudaMemcpyDeviceToHost));
      terr = last_normalize (x, N, error_flag);
    }
  else
    {

      if ((iter % checkpoint_iter) == 0)
	{
	  cutilSafeCall (cudaMemcpy
			 (x, g_x, sizeof (double) * N,
			  cudaMemcpyDeviceToHost));
	  terr = last_normalize (x, N, error_flag);
	}

      if (error_flag || t_f)
	{
	  normalize_kernel < 1 > <<<N / threads, threads >>> (g_x, threads,
							      g_err, g_carry,
							      g_mask, g_ttp,
							      g_ttmp,
							      g_numbits,
							      *maxerr);
	}
      else
	{
	  normalize_kernel < 0 > <<<N / threads, threads >>> (g_x, threads,
							      g_err, g_carry,
							      g_mask, g_ttp,
							      g_ttmp,
							      g_numbits,
							      *maxerr);
	}
      normalize2_kernel <<< ((N + threads - 1) / threads + 127) / 128,
	128 >>> (g_x, threads, g_carry, N, g_inv2, g_ttp2, g_ttmp2, g_inv3,
		 g_ttp3, g_ttmp3);
     
      terr = 0.0;
      if (error_flag || (polite_f && (iter % polite) == 0))
	{
	  float c_err;
	  cutilSafeCall (cudaMemcpy
			 (&c_err, g_err, sizeof (float),
			  cudaMemcpyDeviceToHost));
	  if(c_err > *maxerr)
	    *maxerr = c_err;
	  terr = c_err;
	}
    }
  return (terr);
}

/**************************************************************************
 *                                                                        *
 *       End LL/GPU Functions, Begin Utility/CPU Functions                *
 *                                                                        *
 **************************************************************************/

int
choose_fft_length (int q, int* index)
{  
/* In order to increase length if an exponent has a round off issue, we use an
extra paramter that we can adjust on the fly. In check(), index starts as -1,
the default. In that case, choose from the table. If index >= 0, we must assume
it's an override index and return the corresponding length. If index > table-count,
then we assume it's a manual fftlen and return the proper index. */
  #define COUNT 89
  int multipliers[COUNT] = {  6,     8,    12,    16,    18,    24, 
                             32,    40,    48,    64,    72,    80, 
                             96,   120,   128,   144,   160,   192, 
                            240,   256,   288,   320,   384,   480, 
                            512,   576,   640,   768,   864,   960, 
                           1024,  1152,  1280,  1440,  1536,  1600, 
                           1728,  1920,  2048,  2304,  2400,  2560,
                           2880,  3072,  3200,  3456,  3840,  4000, 
                           4096,  4608,  4800,  5120,  5760,  6144, 
                           6400,  6912,  7680,  8000,  8192,  9216,
                           9600, 10240, 11520, 12288, 12800, 13824, 
                          15360, 16000, 16384, 18432, 19200, 20480, 
                          23040, 24576, 25600, 27648, 30720, 32000, 
                          32768, 34992, 36864, 38400, 40960, 46080,
                          49152, 51200, 55296, 61440, 65536         };
  // Largely copied from Prime95's jump tables, up to 32M
  // Support up to 64M, the maximum length with threads == 1024
  if( 0 <= *index && *index < COUNT ) // override
    return 1024*multipliers[*index];  
  else if( *index >= COUNT || q == 0) 
  { /* override with manual fftlen passed as arg; set pointer to largest index <= fftlen */
    int len, i;
    for(i = COUNT - 1; i >= 0; i--) {
      len = 1024*multipliers[i];
      if( len <= *index )
      {
        *index = i;
        return len; /* not really necessary, but now we could decide to override 
        fftlen with this value here */
      }
    }
  }
  else  
  { // *index < 0, not override, choose length and set pointer to proper index
    int len, i, estimate = q/20;
    for(i = 0; i < COUNT; i++) {
      len = 1024*multipliers[i];
      if( len >= estimate ) 
      {
        *index = i;
        return len;
      }
    }
  }
  return 0;
}

int fft_from_str(const char* str)
/* This is really just strtoul with some extra magic to deal with K or M */
{
  char* endptr;
  const char* ptr = str;
  int len, mult = 0;
  while( *ptr ) {
    if( *ptr == 'k' || *ptr == 'K' ) {
      mult = 1024;
      break;
    }
    if( *ptr == 'm' || *ptr == 'M' ) {
      mult = 1024*1024;
      break;
    }
    ptr++;
  }
  if( !mult ) { // No K or M, treat as before    (PS The Python else clause on loops I mention in parse.c would be useful here :) )
    mult = 1;
  }
  len = (int) strtoul(str, &endptr, 10)*mult;
  if( endptr != ptr ) { // The K or M must directly follow the num (or the num must extend to the end of the str)
    fprintf (stderr, "can't parse fft length \"%s\"\n\n", str);
    exit (2);
  }
  return len;
}

//From apsen
void
print_time_from_seconds (int sec)
{
  if (sec > 3600)
    {
      printf ("%d", sec / 3600);
      sec %= 3600;
      printf (":%02d", sec / 60);
    }
  else
    printf ("%d", sec / 60);
  sec %= 60;
  printf (":%02d", sec);
}

void
init_device (int device_number)
{
  int device_count = 0;
  struct cudaDeviceProp properties;
  cudaGetDeviceCount (&device_count);
  if (device_number >= device_count)
    {
      printf ("device_number >=  device_count ... exiting\n");
      printf ("(This is probably a driver problem)\n\n");
      exit (2);
    }
  if (d_f)
    {
      cudaDeviceProp dev;
      cudaGetDeviceProperties (&dev, device_number);
      printf ("------- DEVICE %d -------\n", device_number);
      printf ("name                %s\n", dev.name);
      printf ("totalGlobalMem      %d\n", (int) dev.totalGlobalMem);
      printf ("sharedMemPerBlock   %d\n", (int) dev.sharedMemPerBlock);
      printf ("regsPerBlock        %d\n", (int) dev.regsPerBlock);
      printf ("warpSize            %d\n", (int) dev.warpSize);
      printf ("memPitch            %d\n", (int) dev.memPitch);
      printf ("maxThreadsPerBlock  %d\n", (int) dev.maxThreadsPerBlock);
      printf ("maxThreadsDim[3]    %d,%d,%d\n",
	 dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
      printf ("maxGridSize[3]      %d,%d,%d\n", dev.maxGridSize[0],
	 dev.maxGridSize[1], dev.maxGridSize[2]);
      printf ("totalConstMem       %d\n", (int) dev.totalConstMem);
      printf ("Compatibility       %d.%d\n", dev.major, dev.minor);
      printf ("clockRate (MHz)     %d\n", dev.clockRate/1000);
      printf ("textureAlignment    %d\n", (int) dev.textureAlignment);
      printf ("deviceOverlap       %d\n", dev.deviceOverlap);
      printf ("multiProcessorCount %d\n\n", dev.multiProcessorCount);
    }
  cudaSetDeviceFlags (cudaDeviceBlockingSync);
  cudaSetDevice (device_number);
// From Iain
  cudaGetDeviceProperties (&properties, device_number);

  if (properties.major == 1 && properties.minor < 3)
    {
      printf
	("A GPU with compute capability >= 1.3 is required for double precision arithmetic\n\n");
      exit (2);
    }
}

int
is_big2 (int j, int bigx, int smallx, int n)
{
  return ((((bigx * j) % n) >= smallx) || j == 0);
}

void
balancedtostdrep (double *x, int n, int b, int c, double hi, double lo,
		  int mask, int shift)
{
  int sudden_death = 0, j = 0, NminusOne = n - 1, k, k1;
  while (1)
    {
      k = j + ((j & mask) >> shift);
      if (x[k] < 0.0)
	{
	  k1 = (j + 1) % n;
	  k1 += (k1 & mask) >> shift;
	  --x[k1];
	  if (j == 0 || (j != NminusOne && is_big2 (j, b, c, n)))
	    x[k] += hi;
	  else
	    x[k] += lo;
	}
      else if (sudden_death)
	break;
      if (++j == n)
	{
	  sudden_death = 1;
	  j = 0;
	}
    }
}

int
is_zero (double *x, int n, int mask, int shift)
{
  int j, offset;
  for (j = 0; j < n; ++j)
    {
      offset = j + ((j & mask) >> shift);
      if (rint (x[offset]))
	return (0);
    }
  return (1);
}

int
printbits (double *x, int q, int N,
	   int b, int c, double high, double low, int totalbits,
	   FILE* fp, char *expectedResidue)
{
  char *bits = (char *) malloc ((int) totalbits);
  char residue[32];
  char temp[32];
  int j, k, i, word;
  
  if (is_zero (x, N, 0, 0))
    {
      printf ("M( %d )P, n = %dK, %s", q, N/1024, program);
      if (fp)
	{
	  fprintf (fp, "M( %d )P, n = %dK, %s", q, N/1024, program);
	}
    }
  else
    {
      double *x_tmp;
      x_tmp = (double *) malloc (sizeof (double) * N);
      for (i = 0; i < N; i++)
	x_tmp[i] = x[i];
      balancedtostdrep (x_tmp, N, b, c, high, low, 0, 0);
      printf ("M( %d )C, 0x", q);
      if (fp)
	fprintf (fp, "M( %d )C, 0x", q);
      j = 0;
      i = 0;
      do
	{
	  k = (int) (ceil ((double) q * (j + 1) / N) -
		     ceil ((double) q * j / N));
	  if (k > totalbits)
	    k = totalbits;
	  totalbits -= k;
	  word = (int) x_tmp[j++];
	  while (k--)
	    {
	      bits[i++] = (char) ('0' + (word & 1));
	      word >>= 1;
	    }
	}
      while (totalbits);
      residue[0] = 0;
      while (i)
	{
	  k = 0;
	  for (j = 0; j < 4; j++)
	    {
	      i--;
	      k <<= 1;
	      if (bits[i] == '1')
		k++;
	    }
	  if (k > 9)
	    {
	      sprintf (temp, "%s", residue);
	      sprintf (residue, "%s%c", temp, (char) ('a' + k - 10));
	    }
	  else
	    {
	      sprintf (temp, "%s", residue);
	      sprintf (residue, "%s%c", temp, (char) ('0' + k));
	    }
	}
      free (x_tmp);
      printf ("%s", residue);
      printf (", n = %dK, %s", N/1024, program);
      if (fp)
	{
	  fprintf (fp, "%s", residue);
	  fprintf (fp, ", n = %dK, %s", N/1024, program);
	}
      if (expectedResidue && strcmp (residue, expectedResidue))
      {
	   bad_selftest++;
	   return 1;
      }
      else if(expectedResidue) 
	{
	   return 0;
	}
    } /* end else res not 0 */
  sprintf (s_residue, "%s", residue);
  free (bits);
  return 0;
}

void
rm_checkpoint (int q)
{
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  sprintf (chkpnt_cfn, "c" "%d", q);
  sprintf (chkpnt_tfn, "t" "%d", q);
  (void) unlink (chkpnt_cfn);
  (void) unlink (chkpnt_tfn);
}

double *
read_checkpoint (int q, int *n, int *j, long* total_time)
{
/* If we get file reading errors, then this doesn't try the backup t-file. Should we add that? */
  FILE *fPtr;
  int q_r, n_r, j_r;
  long time_r;
  double *x;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  sprintf (chkpnt_cfn, "c%d", q);
  sprintf (chkpnt_tfn, "t%d", q);
  fPtr = fopen (chkpnt_cfn, "rb");
  if (!fPtr)
    {
      fPtr = fopen (chkpnt_tfn, "rb");
      if (!fPtr)
	return NULL;
    }
  // check parameters
  if (   fread (&q_r, 1, sizeof (q_r), fPtr) != sizeof (q_r) 
      || fread (&n_r, 1, sizeof (n_r), fPtr) != sizeof (n_r) 
      || fread (&j_r, 1, sizeof (j_r), fPtr) != sizeof (j_r) )
    {
      fprintf (stderr,
	       "\nThe checkpoint appears to be corrupt.  Current test will be restarted.\n");
      fclose (fPtr);
      return NULL;
    }
  if (q != q_r)
    {
      fprintf
	(stderr,
	 "\nThe checkpoint doesn't match current test.  Current test will be restarted.\n");
      fclose (fPtr);
      return NULL;
    }
  // check for successful read of z, delayed until here since zSize can vary
  x = (double *) malloc (sizeof (double) * (n_r + n_r));
  if (fread (x, 1, sizeof (double) * (n_r), fPtr) !=
      (sizeof (double) * (n_r)))
    {
      fprintf (stderr,
	       "\nThe checkpoint appears to be corrupt.   Current test will be restarted.\n");
      fclose (fPtr);
      free (x);
      return NULL;
    }
  /* Attempt to read total time, ignoring errors for compatibilty with old checkpoints (2.00-2.03). */
  if ( fread (&time_r, 1, sizeof(time_r), fPtr) != sizeof (time_r) ) {
    *total_time = -1;
  } else {
    *total_time = time_r;
    #ifdef EBUG
    printf("Total time read from save file: ");
    print_time_from_seconds(time_r);
    printf("\n");
    #endif
  }
  
  // have good stuff, do checkpoint
  *n = n_r;
  *j = j_r;
  fclose (fPtr);
  return x;
}

void
write_checkpoint (double *x, int q, int n, int j, long total_time)
{
  FILE *fPtr;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  sprintf (chkpnt_cfn, "c%d", q);
  sprintf (chkpnt_tfn, "t%d", q);
  (void) unlink (chkpnt_tfn);
  (void) rename (chkpnt_cfn, chkpnt_tfn);
  fPtr = fopen (chkpnt_cfn, "wb");
  if (!fPtr) {
    fprintf(stderr, "Couldn't write checkpoint.\n");
    return;
  }
  fwrite (&q, 1, sizeof (q), fPtr);
  fwrite (&n, 1, sizeof (n), fPtr);
  fwrite (&j, 1, sizeof (j), fPtr);
  fwrite (x, 1, sizeof (double) * n, fPtr);
  fwrite (&total_time, 1, sizeof(total_time), fPtr);
  #ifdef EBUG
  printf("Total time is %ld\n", total_time);
  printf("Total time again: ");
  print_time_from_seconds(total_time);
  printf("\n");
  #endif
  fclose (fPtr);
  if (s_f > 0)			// save all checkpoint files
    {
      char chkpnt_sfn[64];
#ifndef _MSC_VER
      sprintf (chkpnt_sfn, "%s/s" "%d.%d.%s", folder, q, j, s_residue);
#else
      sprintf (chkpnt_sfn, "%s\\s" "%d.%d.%s.txt", folder, q, j, s_residue);
#endif
      fPtr = fopen (chkpnt_sfn, "wb");
      if (!fPtr)
	return;
      fwrite (&q, 1, sizeof (q), fPtr);
      fwrite (&n, 1, sizeof (n), fPtr);
      fwrite (&j, 1, sizeof (j), fPtr);
      fwrite (x, 1, sizeof (double) * n, fPtr);
      fwrite (&total_time, 1, sizeof(total_time), fPtr);
      fclose (fPtr);
    }
}

void
cufftbench (int cufftbench_s, int cufftbench_e, int cufftbench_d)
{
  cudaEvent_t start, stop;
  double *x;
  float outerTime;
  int i, j;
  printf ("CUFFT bench start = %d end = %d distance = %d\n", cufftbench_s,
	  cufftbench_e, cufftbench_d);

  cutilSafeCall (cudaMalloc ((void **) &g_x, sizeof (double) * cufftbench_e));
  x = ((double *) malloc (sizeof (double) * cufftbench_e + 1));
  for (i = 0; i <= cufftbench_e; i++)
    x[i] = 0;
  cutilSafeCall (cudaMemcpy
		 (g_x, x, sizeof (double) * cufftbench_e,
		  cudaMemcpyHostToDevice));
  cutilSafeCall (cudaEventCreate (&start));
  cutilSafeCall (cudaEventCreate (&stop));
  for (j = cufftbench_s; j <= cufftbench_e; j += cufftbench_d)
    {
      cufftSafeCall (cufftPlan1d (&plan, j / 2, CUFFT_Z2Z, 1));
      cufftSafeCall (cufftExecZ2Z
		     (plan, (cufftDoubleComplex *) g_x,
		      (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
      cutilSafeCall (cudaEventRecord (start, 0));
      for (i = 0; i < 100; i++)
	cufftSafeCall (cufftExecZ2Z
		       (plan, (cufftDoubleComplex *) g_x,
			(cufftDoubleComplex *) g_x, CUFFT_INVERSE));
      cutilSafeCall (cudaEventRecord (stop, 0));
      cutilSafeCall (cudaEventSynchronize (stop));
      cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
      printf ("CUFFT_Z2Z size= %d time= %f msec\n", j, outerTime / 100);
      cufftSafeCall (cufftDestroy (plan));
    }
  cutilSafeCall (cudaFree ((char *) g_x));
  cutilSafeCall (cudaEventDestroy (start));
  cutilSafeCall (cudaEventDestroy (stop));
  free ((char *) x);
}

void
SetQuitting (int sig)
{
  quitting = 1;
 sig==SIGINT ? printf( "\tSIGINT") : (sig==SIGTERM ? printf( "\tSIGTERM") : printf( "\tUnknown signal")) ;
 printf( " caught, writing checkpoint.");
}

#ifndef _MSC_VER
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
int
_kbhit (void)
{
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr (STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr (STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl (STDIN_FILENO, F_GETFL, 0);
  fcntl (STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

  ch = getchar ();

  tcsetattr (STDIN_FILENO, TCSANOW, &oldt);
  fcntl (STDIN_FILENO, F_SETFL, oldf);

  if (ch != EOF)
    {
      ungetc (ch, stdin);
      return 1;
    }

  return 0;
}
#else
#include <conio.h>
#endif
void interact(void); // defined below everything else
/**************************************************************
 *
 *      Main Function
 *
 **************************************************************/
int
check (int q, char *expectedResidue)
{
  int n, j = 1L, last = 2L, error_flag;
  int t_ff = t_f; t_f = 1; /* Override for round off test, deactivate before actual test */
  int fft_index = -1; // See the comments of choose_length() for an explanation
  size_t k;
  double terr, *x = NULL, totalerr, avgerr;
  float maxerr;
  int restarting = 0, continuing = 0;
  timeval time0, time1;
  long total_time = 0, start_time;
  // use start_total because every time we write a checkpoint, total_time is increased, but we don't quit everytime we write a checkpoint
  if (!expectedResidue)
    {
      // We log to file in most cases anyway.
      signal (SIGTERM, SetQuitting);
      signal (SIGINT, SetQuitting);
    }
  do
    {				/* while (restarting) */
      totalerr = maxerr = 0.0; j = 1L;
      
      if (fftlen) {
	n = fftlen;
	choose_fft_length(0, &fftlen); // set fftindex to the largest index < fftlen,
	fft_index = fftlen;            // so that fft_index++ works even for manual lengths
      } else        
	n = choose_fft_length(q, &fft_index);

      if ((n / threads) > 65535)
	{
	  fprintf (stderr, "over specifications Grid = %d\n", (int) n / threads);
	  fprintf (stderr, "try increasing threads (%d) or decreasing FFT length (%dK)\n\n",  threads, n/1024);
	  exit (2);
	}
      if (!expectedResidue && !restarting
	  && (x = read_checkpoint (q, &n, &j, &total_time)) != NULL) {
	 printf
	  ("Continuing work from a partial result of M%d fft length = %dK iteration = %d\n",
	   q, n/1024, j);
	 continuing = 1;
	}
      else
	{
	  printf ("Starting M%d fft length = %dK\n", q, n/1024);
	  x = (double *) malloc (sizeof (double) * (n + n));
	  for (k = 1; k < (unsigned int)n; k++)
	    x[k] = 0.0;
	  x[0] = 4.0;
	  j = 1;
	  j_save = 0; // Only if(t_f) in old code
	  total_time = 0;	  
	}
      fflush (stdout);
      restarting = 0;
      init_lucas (x, q, n);
      gettimeofday (&time0, NULL);
      start_time = time0.tv_sec;
      last = q - 2;		/* the last iteration done in the primary loop */
      
      if( !continuing ) {
        printf("Running careful round off test for 1000 iterations. If average error >= 0.25, the test will restart with a larger FFT length.\n");
        /* This isn't actually any *safer* than the previous code which had error_flag = 1 if j < 1000, 
        but there would have been a *lot* of if statements to get the extra messages working, and there 
        were lots of ifs already. Now there's a lot less ifs in the main code, and now we can check the 
        average. We also reset the maxerr a lot more often (which doesn't make it *safer*, but makes the
        average more accurate). */
        double max_err = 0.0;
        
        for (; !restarting && j < 1000; j++) // Initial LL loop; no -k, no checkpoints, quitting doesn't do anything
	  {
	    terr = lucas_square (x, n, j, last, &maxerr, 1);
	    totalerr += terr; // only works if error_flag is 1 for every iteration
	    if(terr > max_err)
	      max_err = terr; 
	    // ^ This is necessary because we want to print the max_err every 100 iters, but we reset maxerr every 16 iters.
	    if( !(j&15) )
	    {
	      set_err_to_0(&maxerr);
	      if(terr >= 0.35)
	        {
	          /* n is not big enough; increase it and start over */
		   printf
	  	    ("Iteration = %d < 1000 && err = %5.5f >= 0.35, increasing n from %dK\n",
		    j, terr, (int) n/1024);
		   fft_index++;
		   restarting = 1;
		   fftlen = 0; // Whatever the user entered isn't good enough, so override
	        }
	    }
	    if( j % 100 == 0 ) {
	      printf(
	        "Iteration  %d, average error = %5.5f, max error = %5.5f\n",
	        j, totalerr/(double)j, max_err);
	      max_err = 0.0;
	    }
	} // End special/initial LL for loop; now we determine if the error is acceptable
	  
	if( !restarting ) {
          avgerr = totalerr/(j-1); // j should be 1000, but we only did 999 iters
          if( avgerr >= 0.25 ) {
            printf("Iteration 1000, average error = %5.5f >= 0.25 (max error = %5.5f), increasing FFT length and restarting.\n", avgerr, max_err);
            fftlen = 0; // Whatever the user entered isn't good enough, so override
            fft_index++;
            restarting = 1;
          } else if( avgerr < 0 ) {
            fprintf(stderr, "Something's gone terribly wrong! Avgerr = %5.5f < 0 !\n", avgerr);
          } else {
            printf("Iteration 1000, average error = %5.5f < 0.25 (max error = %5.5f), continuing test.\n", avgerr, max_err);
            continuing = 1; // I don't think this is necessary, but just in case.
            set_err_to_0(&maxerr);
          }
        }
      } // End special/initial testing      
      
      if( !t_ff ) t_f = 0; // Undo t_f override from beginning of check()
      
      for (; !restarting && j <= last; j++) // Main LL loop, j should be >= 1000 unless user killed a test with j<1000
	{
	  if ((j % 100) == 1)
	    error_flag = 1;
	  else
	    error_flag = 0;

	  terr = lucas_square (x, n, j, last, &maxerr, error_flag);
		
	  if (error_flag || (polite_f && !(j%polite)))
	    {
	      if (terr >= 0.35)
		    {
		      if (t_f)
			{
			  gettimeofday(&time1, NULL);
			  printf
			    ("Iteration = %d >= 1000 && err = %5.5g >= 0.35, fft length = %dK, writing checkpoint file (because -t is enabled) and exiting.\n\n",
			     j, (double) terr, (int) n/1024);
			  cutilSafeCall (cudaMemcpy
					 (x, g_save, sizeof (double) * n,
					  cudaMemcpyDeviceToHost));
			  if( total_time >= 0) {total_time += (time1.tv_sec - start_time);}
			  write_checkpoint (x, q, n, j_save + 1, total_time);
			  exit (2);
			}
		      else
			{
			  printf
			    ("Iteration = %d >= 1000 && err = %5.5g >= 0.35, fft length = %dK, not writing checkpoint file (because -t is disabled) and exiting.\n\n",
			     j, (double) terr, (int) n/1024);
			  exit (2);
			}
		    }
		else		// error_flag && terr < 0.35
		    {
		      if (t_f)
			{
			  copy_kernel <<< n / 128, 128 >>> (g_save, g_x);
			  j_save = j;
			}
		    }
	    }
	  if ((j % checkpoint_iter) == 0)
	    {
	      gettimeofday (&time1, NULL);
	      printf ("Iteration %d ", j);
	      int ret = printbits (x, q, n, b, c, high, low, 64, 0, expectedResidue);
	      long diff = time1.tv_sec - time0.tv_sec;
	      long diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
	      printf (" err = %4.4f (", maxerr);
	      print_time_from_seconds (diff);
	      printf (" real, %4.4f ms/iter, ETA ",
		      diff1 / 1000.0 / checkpoint_iter);
	      diff = (long) ((last - j) / checkpoint_iter * (diff1 / 1e6));
	      print_time_from_seconds (diff);
	      printf (")\n");
	      fflush (stdout);
	      set_err_to_0(&maxerr); // Instead of tracking maxerr over whole run, reset it at each checkpoint.
	      gettimeofday (&time0, NULL);
	      if (expectedResidue) 
	      {
		j = last + 1;
		if (ret)
		  printf
		  ("Expected residue [%s] does not match actual residue [%s]\n\n",
	          expectedResidue, s_residue);
	        else printf("This residue is correct.\n\n");
	      }
	    }

	  if (((j % checkpoint_iter) == 0 || quitting == 1)
	      && !expectedResidue)
	    {
	      gettimeofday (&time1, NULL);
	      cutilSafeCall (cudaMemcpy
			     (x, g_x, sizeof (double) * n,
			      cudaMemcpyDeviceToHost));
	      if( total_time >= 0 ) {
	        total_time += (time1.tv_sec - start_time);
	        start_time = time1.tv_sec;
	      }
	      write_checkpoint (x, q, n, j + 1, total_time);
	      if (quitting == 1) {
	        if(total_time >= 0) {
	          printf(" Estimated time spent so far: ");
	          print_time_from_seconds(total_time);
	        }
	        printf("\n\n");
		j = last + 1;
	      }
	    }

	  if ( k_f && !quitting && !expectedResidue && (!(j & 15)) && _kbhit() )
	    interact(); // abstracted to clean up check()
	  
	  fflush (stdout);	    
	} /* end main LL for-loop */
	
      if (!restarting && !expectedResidue && !quitting)
	{
	  gettimeofday (&time1, NULL);
	  FILE* fp = fopen_and_lock(RESULTSFILE, "a");
	  if(!fp) {
	    fprintf (stderr, "Cannot write results to %s\n\n", RESULTSFILE);
	    exit (1);
	  }
	  printbits (x, q, n, b, c, high, low, 64, fp, 0);
	  if( total_time >= 0 ) { /* Only print time if we don't have an old checkpoint file */
	    total_time += (time1.tv_sec - start_time);
	    printf (", estimated total time = ");
	    print_time_from_seconds(total_time);
	  }
	  
	  if( AID[0] && strncasecmp(AID, "N/A", 3) ) { // If (AID is not null), AND (AID is NOT "N/A") (case insensitive)
	    fprintf(fp, ", AID: %s\n", AID);
	  } else {
	    fprintf(fp, "\n");
	  }
	  unlock_and_fclose(fp);
	  fflush (stdout);
	  rm_checkpoint (q);
	  printf("\n\n");
	}
      close_lucas (x);
    }
  while (restarting);
  return (0);
}

void parse_args(int argc, char *argv[], int* q, int* device_numer, 
		int* cufftbench_s, int* cufftbench_e, int* cufftbench_d);
		/* The rest of the opts are global */
int main (int argc, char *argv[])
{ 
  printf("\n");
  quitting = 0;
#define THREADS_DFLT 256
#define CHECKPOINT_ITER_DFLT 10000
#define SAVE_FOLDER_DFLT "savefiles"
#define S_F_DFLT 0
#define T_F_DFLT 0
#define K_F_DFLT 0
#define D_F_DFLT 0
#define POLITE_DFLT 1
#define WORKFILE_DFLT "worktodo.txt"
#define RESULTSFILE_DFLT "results.txt"
  
  /* "Production" opts to be read in from command line or ini file */
  int q = -1;
  int device_number = -1, f_f = 0;
  checkpoint_iter = -1;
  threads = -1;
  fftlen = -1;
  s_f = t_f = d_f = k_f = -1;
  polite_f = polite = -1;
  input_filename[0] = RESULTSFILE[0] = 0; /* First character is null terminator */
  char fft_str[132] = "\0";
  
  /* Non-"production" opts */
  r_f = 0;
  int cufftbench_s, cufftbench_e, cufftbench_d;  
  cufftbench_s = cufftbench_e = cufftbench_d = 0;

  parse_args(argc, argv, &q, &device_number, &cufftbench_s, &cufftbench_e, &cufftbench_d);
  /* The rest of the args are globals */
  
  if (file_exists(INIFILE))
  {  
   if( checkpoint_iter < 1 && 		!IniGetInt(INIFILE, "CheckpointIterations", &checkpoint_iter, CHECKPOINT_ITER_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option CheckpointIterations; using default: %d\n", CHECKPOINT_ITER_DFLT)*/;
   if( threads < 1 && 			!IniGetInt(INIFILE, "Threads", &threads, THREADS_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option Threads; using default: %d\n", THREADS_DFLT);
   if( s_f < 0 && 			!IniGetInt(INIFILE, "SaveAllCheckpoints", &s_f, S_F_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option SaveAllCheckpoints; using default: off\n")*/;
   if( 		     	     s_f > 0 && !IniGetStr(INIFILE, "SaveFolder", folder, SAVE_FOLDER_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option SaveFolder; using default: \"%s\"\n", SAVE_FOLDER_DFLT)*/;
   if( t_f < 0 && 			!IniGetInt(INIFILE, "CheckRoundoffAllIterations", &t_f, 0) )
    fprintf(stderr, "Warning: Couldn't parse ini file option CheckRoundoffAllIterations; using default: off\n");
   if( polite < 0 && 			!IniGetInt(INIFILE, "Polite", &polite, POLITE_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option Polite; using default: %d\n", POLITE_DFLT);
   if( k_f < 0 && 			!IniGetInt(INIFILE, "Interactive", &k_f, 0) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option Interactive; using default: off\n")*/;
   if( device_number < 0 &&		!IniGetInt(INIFILE, "DeviceNumber", &device_number, 0) )
    fprintf(stderr, "Warning: Couldn't parse ini file option DeviceNumber; using default: 0\n");
   if( d_f < 0 &&			!IniGetInt(INIFILE, "PrintDeviceInfo", &d_f, D_F_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option PrintDeviceInfo; using default: off\n")*/;
   if( !input_filename[0] &&		!IniGetStr(INIFILE, "WorkFile", input_filename, WORKFILE_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option WorkFile; using default \"%s\"\n", WORKFILE_DFLT);
    /* I've readded the warnings about worktodo and results due to the multiple-instances-in-one-dir feature. */
   if( !RESULTSFILE[0] && 		!IniGetStr(INIFILE, "ResultsFile", RESULTSFILE, RESULTSFILE_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option ResultsFile; using default \"%s\"\n", RESULTSFILE_DFLT);
   if( fftlen < 0 && 			!IniGetStr(INIFILE, "FFTLength", fft_str, "\0") )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option FFTLength; using autoselect.\n")*/;
  }
  else // no ini file
    {
      fprintf(stderr, "Warning: Couldn't find .ini file. Using defaults for non-specified options.\n");
      if( checkpoint_iter < 1 ) checkpoint_iter = CHECKPOINT_ITER_DFLT;
      if( threads < 1 ) threads = THREADS_DFLT;
      if( fftlen < 0 ) fftlen = 0;
      if( s_f < 0 ) s_f = S_F_DFLT;
      if( t_f < 0 ) t_f = T_F_DFLT;
      if( k_f < 0 ) k_f = K_F_DFLT;
      if( device_number < 0 ) device_number = 0;
      if( d_f < 0 ) d_f = D_F_DFLT;
      if( polite < 0 ) polite = POLITE_DFLT;
      if( !input_filename[0] ) sprintf(input_filename, WORKFILE_DFLT);
      if( !RESULTSFILE[0] ) sprintf(RESULTSFILE, RESULTSFILE_DFLT);
  }
  
  if( fftlen < 0 ) { // possible if -f not on command line
      fftlen = fft_from_str(fft_str);
  }
  if (polite == 0) {
    polite_f = 0;
    polite = 1;
  } else {
    polite_f = 1;
  }
  if (threads != 32 && threads != 64 && threads != 128
	      && threads != 256 && threads != 512 && threads != 1024)
  {
    fprintf(stderr, "Error: thread count is invalid.\n");
    fprintf(stderr, "Threads must be 2^k, 5 <= k <= 10.\n\n");
    exit(2);
  }
  f_f = fftlen; // if the user has given an override... then note this length must be kept between tests
    
  
  init_device (device_number);

  if (r_f)
    {
      fftlen = 0;
      checkpoint_iter = 10000;
      t_f = 1;
      check (86243, "23992ccd735a03d9");
      check (132049, "4c52a92b54635f9e");
      check (216091, "30247786758b8792");
      check (756839, "5d2cbe7cb24a109a");
      check (859433, "3c4ad525c2d0aed0");
      check (1257787, "3f45bf9bea7213ea");
      check (1398269, "a4a6d2f0e34629db");
      check (2976221, "2a7111b7f70fea2f");
      check (3021377, "6387a70a85d46baf");
      check (6972593, "88f1d2640adb89e1");
      check (13466917, "9fdc1f4092b15d69");
      check (20996011, "5fc58920a821da11");
      check (24036583, "cbdef38a0bdc4f00");
      check (25964951, "62eb3ff0a5f6237c");
      check (30402457, "0b8600ef47e69d27");
      check (32582657, "02751b7fcec76bb1");
      check (37156667, "67ad7646a1fad514");
      check (42643801, "8f90d78d5007bba7");
      check (43112609, "e86891ebf6cd70c4");
      if (bad_selftest)
      {
        fprintf(stderr, "Error: There ");
        bad_selftest > 1 ? fprintf(stderr, "were %d bad selftests!\n",bad_selftest) 
        		 : fprintf(stderr, "was a bad selftest!\n");
      }
    }
  else if (cufftbench_d)
    cufftbench (cufftbench_s, cufftbench_e, cufftbench_d);
  else
    {
      if (s_f)
	{
#ifndef _MSC_VER
	  mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
	  if (mkdir (folder, mode) != 0)
	    fprintf (stderr,
		     "mkdir: cannot create directory `%s': File exists\n",
		     folder);
#else
	  if (_mkdir (folder) != 0)
	    fprintf (stderr,
		     "mkdir: cannot create directory `%s': File exists\n",
		     folder);
#endif
	}
      if (q <= 0)
      {
        int error;
	
	#ifdef EBUG
	printf("Processed INI file and console arguments correctly; about to call get_next_assignment().\n");
	#endif
	do { // while(!quitting)
	
	                 
	  fftlen = f_f; // fftlen can be changed for each test, so be sure to reset it
	                   
  	  error = get_next_assignment(input_filename, &q, &fftlen, &AID);
  	  /* Guaranteed to write to fftlen ONLY if specified on workfile line, so that if unspecified, the pre-set
  	  default is kept. */
	  if( error ) exit (2); // get_next_assignment prints warning message	  
	  #ifdef EBUG
	  printf("Gotten assignment, about to call check().\n");
	  #endif
	  
	  check (q, 0);
	  
	  if(!quitting) // Only clear assignment if not killed by user, i.e. test finished 
	    {
	      error = clear_assignment(input_filename, q);
	      if(error) exit (2); // prints its own warnings
	    }
	    
	} while(!quitting);  
      } else // Exponent passed in as argument
	{
	  if (!valid_assignment(q, fftlen)) {printf("\n");} //! v_a prints warning
	  else
	    check (q, 0);
	}
    } // end if(-r) else if(-cufft) else(workfile)
} // end main()

void parse_args(int argc, char *argv[], int* q, int* device_number, 
		int* cufftbench_s, int* cufftbench_e, int* cufftbench_d)
{
while (argc > 1)
    {
      if (strcmp (argv[1], "-t") == 0)
	{
	  t_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-h") == 0)
        {
      	  fprintf (stderr,
	       "$ CUDALucas -h|-v\n\n");
      	  fprintf (stderr,
	       "$ CUDALucas [-d device_number] [-info] [-i inifile] [-threads 32|64|128|256|512|1024] [-c checkpoint_iteration] [-f fft_length] [-s folder] [-t] [-polite iteration] [-k] exponent|input_filename\n\n");
      	  fprintf (stderr,
	       "$ CUDALucas [-d device_number] [-info] [-i inifile] [-threads 32|64|128|256|512|1024] [-polite iteration] -r\n\n");
      	  fprintf (stderr,
	       "$ CUDALucas [-d device_number] [-info] -cufftbench start end distance\n\n");
	  fprintf (stderr,
	       "                       -h          print this help message\n");
	  fprintf (stderr,
	       "                       -v          print version number\n");
	  fprintf (stderr,
	       "                       -info       print device information\n");
	  fprintf (stderr,
	       "                       -i          set .ini file name (default = \"CUDALucas.ini\")\n");
      	  fprintf (stderr,
	       "                       -threads    set threads number (default = 256)\n");
      	  fprintf (stderr,
	       "                       -f          set fft length (if round off error then exit)\n");
      	  fprintf (stderr,
	       "                       -s          save all checkpoint files\n");
      	  fprintf (stderr,
	       "                       -t          check round off error all iterations\n");
      	  fprintf (stderr,
	       "                       -polite     GPU is polite every n iterations (default -polite 1) (-polite 0 = GPU aggressive)\n");
      	  fprintf (stderr,
	       "                       -cufftbench exec CUFFT benchmark (Ex. $ ./CUDALucas -d 1 -cufftbench 1179648 6291456 32768 )\n");
      	  fprintf (stderr, 
      	       "                       -r          exec residue test.\n");
      	  fprintf (stderr,
	       "                       -k          enable keys (p change -polite, t disable -t, s change -s)\n\n");
      	  exit (2);          
      	}
      else if (strcmp (argv[1], "-v") == 0)
        {  
          printf("%s\n\n", program);
          exit (2);
        }
      else if (strcmp (argv[1], "-polite") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -polite option\n\n");
	      exit (2);
	    }
	  polite = atoi (argv[2]);
	  if (polite == 0)
	    {
	      polite_f = 0;
	      polite = 1;
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-r") == 0)
	{
	  r_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-k") == 0)
	{
	  k_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-d") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -d option\n\n");
	      exit (2);
	    }
	  *device_number = atoi (argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-i") == 0)
	{
	  if(argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -i option\n\n");
	      exit (2);
	    }
	  sprintf (INIFILE, "%s", argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-info") == 0)
        {
          d_f = 1;
          argv++;
          argc--;
        }
      else if (strcmp (argv[1], "-cufftbench") == 0)
	{
	  if (argc < 5 || argv[2][0] == '-' || argv[3][0] == '-' || argv[4][0] == '-')
	    {
	      fprintf (stderr, "can't parse -cufftbench option\n\n");
	      exit (2);
	    }
	  *cufftbench_s = atoi (argv[2]);
	  *cufftbench_e = atoi (argv[3]);
	  *cufftbench_d = atoi (argv[4]);
	  argv += 4;
	  argc -= 4;
	}
      else if (strcmp (argv[1], "-threads") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -threads option\n\n");
	      exit (2);
	    }
	  threads = atoi (argv[2]);
	  if (threads != 32 && threads != 64 && threads != 128
	      && threads != 256 && threads != 512 && threads != 1024)
	    {
	      fprintf(stderr, "Error: thread count is invalid.\n");
	      fprintf(stderr, "Threads must be 2^k, 5 <= k <= 10.\n\n");
	      exit (2);
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-c") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -c option\n\n");
	      exit (2);
	    }
	  checkpoint_iter = atoi (argv[2]);
	  if (checkpoint_iter == 0)
	    {
	      fprintf (stderr, "can't parse -c option\n\n");
	      exit (2);
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-f") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -f option\n\n");
	      exit (2);
	    }
	  fftlen = fft_from_str(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-s") == 0)
	{
	  s_f = 1;
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -s option\n\n");
	      exit (2);
	    }
	  sprintf (folder, "%s", argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else
	{
	  if (*q != -1 || strcmp (input_filename, "") != 0 )
	    {
	      fprintf (stderr, "can't parse options\n\n");
	      exit (2);
	    }
	  int derp = atoi (argv[1]);
	  if (derp == 0) {
	    sprintf (input_filename, "%s", argv[1]);
	  } else { *q = derp; }
	  argv++;
	  argc--;
	}
    }
}

void interact(void)
{
  int c = getchar ();
  if (c == 'p')
    if (polite_f)
	  {
	    polite_f = 0;
	    printf ("   -polite 0\n");
	  }
    else
     {
      polite_f = 1;
      printf ("   -polite %d\n", polite);
     }
  else if (c == 't')
    {
      t_f = 0;
      printf ("   disabling -t\n");
    }
  else if (c == 's')
    if (s_f == 1)
      {
        s_f = 2;
        printf ("   disabling -s\n");
      }
    else if (s_f == 2)
      {
        s_f = 1;
        printf ("   enabling -s\n");
      }
  fflush (stdin);
}
