/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

// The SIFT descriptor is subject to US Patent 6,711,293

#ifndef __SIFTDESC_H__
#define __SIFTDESC_H__

#include <vector>
#include <cv.h>
#include "helpers.h"

struct SIFTDescriptorParams
{
   int spatialBins;
   int orientationBins;
   float maxBinValue;
   int patchSize;
   SIFTDescriptorParams()
      {
         spatialBins = 4;
         orientationBins = 8;
         maxBinValue = 0.2f;
         patchSize = 41;
      } 
};


struct SIFTDescriptor
{

public:
   // top level interface
   SIFTDescriptor(const SIFTDescriptorParams &par)
      {
         this->par = par;
         vec.resize(par.spatialBins * par.spatialBins * par.orientationBins);
         precomputeBinsAndWeights();
      }
   
   void computeSiftDescriptor(Patch &patch);

public:
   std::vector<float> vec;

private:
   // helper functions
   
float normalize();
   void sample(const Patch &patch);
   void samplePatch(const Patch &patch);
   void precomputeBinsAndWeights();

private:
   SIFTDescriptorParams par;
   std::vector<int> precomp_bins;
   std::vector<float> precomp_weights;
   int *bin0, *bin1;
   float *w0, *w1;
};

#endif //__SIFTDESC_H__
