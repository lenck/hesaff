/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#include <vector>
#include "siftdesc.h"

using namespace std;
using namespace cv;

// The SIFT descriptor is subject to US Patent 6,711,293

void SIFTDescriptor::precomputeBinsAndWeights()
{
   int halfSize = par.patchSize>>1;
   float step = float(par.spatialBins+1)/(2*halfSize);
   
   // allocate maps at the same location
   precomp_bins.resize(2*par.patchSize);
   precomp_weights.resize(2*par.patchSize);
   bin1 = bin0 = &precomp_bins.front(); bin1 += par.patchSize;
   w1   =   w0 = &precomp_weights.front(); w1 += par.patchSize;

   // maps every pixel in the patch 0..patch_size-1 to appropriate spatial bin and weight
   for (int i = 0; i < par.patchSize; i++)
   {
      float x = step*i;      // x goes from <-1 ... spatial_bins> + 1
      int  xi = (int)(x);
      // bin indices
      bin0[i] = xi-1; // get real xi
      bin1[i] = xi;
      // weights
      w1[i]   = x - xi;
      w0[i]   = 1.0f - w1[i];
      // truncate weights and bins in case they reach outside of valid range
      if (bin0[i] <          0) { bin0[i] = 0;           w0[i] = 0; }
      if (bin0[i] >= par.spatialBins) { bin0[i] = par.spatialBins-1; w0[i] = 0; }
      if (bin1[i] <          0) { bin1[i] = 0;           w1[i] = 0; }
      if (bin1[i] >= par.spatialBins) { bin1[i] = par.spatialBins-1; w1[i] = 0; }
      // adjust for orientation bin skip
      bin0[i] *= par.orientationBins;
      bin1[i] *= par.orientationBins;
   }
}

void SIFTDescriptor::samplePatch(const Patch &patch)
{  
   for (int r = 0; r < par.patchSize; ++r)
   {
      const int br0 = par.spatialBins * bin0[r]; const float wr0 = w0[r];
      const int br1 = par.spatialBins * bin1[r]; const float wr1 = w1[r];
      // TODO rewrite access to pointers
      for (int c = 0; c < par.patchSize; ++c)
      {
         float val = patch.mask.at<float>(r,c) * patch.gradMag.at<float>(r,c);
         
         const int bc0 = bin0[c]; const float wc0 = w0[c]*val;
         const int bc1 = bin1[c]; const float wc1 = w1[c]*val;
         
         // ori from atan2 is in range <-pi,pi> so add 2*pi to be surely above zero         
         //const float o = float(par.orientationBins)*(patch.gradAngle.at<float>(r,c) + M_PI)/(2.f*M_PI);
         const float o = float(par.orientationBins)*(patch.gradAngle.at<float>(r,c) + 2*M_PI)/(2*M_PI);

         int   bo0 = (int)o;
         const float wo1 =  o - bo0; 
         bo0 %= par.orientationBins;
         
         int   bo1 = (bo0+1) % par.orientationBins; 
         const float wo0 = 1.0f - wo1;
         
         // add to corresponding 8 vec...
         val = wr0*wc0; if (val>0) { vec[br0+bc0+bo0] += val * wo0; vec[br0+bc0+bo1] += val * wo1; }
         val = wr0*wc1; if (val>0) { vec[br0+bc1+bo0] += val * wo0; vec[br0+bc1+bo1] += val * wo1; }
         val = wr1*wc0; if (val>0) { vec[br1+bc0+bo0] += val * wo0; vec[br1+bc0+bo1] += val * wo1; }
         val = wr1*wc1; if (val>0) { vec[br1+bc1+bo0] += val * wo0; vec[br1+bc1+bo1] += val * wo1; }
      }
   }
}

float SIFTDescriptor::normalize()
{
   float vectlen = 0.0f;
   for (size_t i = 0; i < vec.size(); i++) 
   {
      const float val = vec[i];
      vectlen += val * val;
   }
   vectlen = sqrt(vectlen);
   
   const float fac = float(1.0f / vectlen);
   for (size_t i = 0; i < vec.size(); i++) vec[i] *= fac;
   return vectlen;
}

void SIFTDescriptor::sample(const Patch &patch)
{
   for (size_t i = 0; i < vec.size(); i++) vec[i]=0;
   // accumulate histograms
   samplePatch(patch);
   normalize();
   // check if there are some values above threshold
   bool changed = false; 
   for (size_t i = 0; i < vec.size(); i++) if (vec[i] > par.maxBinValue) { vec[i] = par.maxBinValue; changed = true; }
   if (changed) normalize();

   for (size_t i = 0; i < vec.size(); i++) 
   {
      int b  = min((int)(512.0f * vec[i]), 255);
      vec[i] = float(b);
   }
}

void SIFTDescriptor::computeSiftDescriptor(Patch &patch)
{
   patch.computeGradientPolar();
   // compute SIFT vector
   sample(patch);
}
