/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#include "sim.h"

using namespace cv;

bool SimShape::normalize(const Mat &img, float x, float y, float s, float angle)
{
    // TODO test how much this works on similarity invariant images
    // test it...
    // eventually find a way how to solve it for extracting the patches from the scale space..
   // determinant == 1 assumed (i.e. isotropic scaling should be separated in mrScale
   float mrScale = ceil(s * par.mrSize); // half patch size in pixels of image

   int   patchImageSize = 2*int(mrScale)+1; // odd size
   float imageToPatchScale = float(patchImageSize) / float(par.patchSize);  // patch size in the image / patch size -> amount of down/up sampling
   
   float m11,m12,m21,m22;
   m11 = m22 = imageToPatchScale;
   m12 = m21 = 0;
   if (angle != 0.) {
      float lecos=cos(angle), lesin=sin(angle);
      m11 = mi11*lecos;
      m12 = mi11*lesin;
      m21 = -mi22*lesin;
      m22 = mi22*lecos;
   }

   // ok, do the interpolation
   bool touchesBoundary = interpolate(img, x, y, m11, m12, m21, m22, patch.data);
   return touchesBoundary;
}
