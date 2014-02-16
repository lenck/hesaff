/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#ifndef __SIM_H__
#define __SIM_H__

#include <vector>
#include <cv.h>
#include "helpers.h"

struct SimShapeParams
{
   // width and height of the patch
   int patchSize;

   // amount of smoothing applied to the initial level of first octave
   float initialSigma;

   // size of the measurement region (as multiple of the feature scale)
   float mrSize;

   SimShapeParams()
      {
         initialSigma = 1.6f;
         patchSize = 41;
         mrSize = 3.0f*sqrt(3.0f);
      }
};

struct SimShapeCallback
{
   virtual void onSimShapeFound(
      const cv::Mat &blur,     // corresponding scale level
      float x, float y,     // subpixel, image coordinates
      float s,              // scale
      float pixelDistance,  // distance between pixels in provided blured image
      int type, float response, int iters) = 0;
};

struct SimShape
{
public:   
   SimShape(const SimShapeParams &par) :
      patch(par.patchSize)
      {                     
         this->par = par;
         simShapeCallback = 0;
      }
   
   ~SimShape()
      {
      } 

   // fills patch with sim normalized neighbourhood around point in the img, enlarged mrSize times
   bool normalize(const cv::Mat &img, float x, float y, float s, float angle);

   void setSimShapeCallback(SimShapeCallback *callback)
      {
         simShapeCallback = callback;
      }

public:
   Patch patch;

protected:
   SimShapeParams par;

private:
   SimShapeCallback *simShapeCallback;
};

#endif // __SIM_H__
