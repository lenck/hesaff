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

bool SimShape::normalize(const Mat &img, float x, float y, float s)
{
   // determinant == 1 assumed (i.e. isotropic scaling should be separated in mrScale
   float mrScale = ceil(s * par.mrSize); // half patch size in pixels of image

   int   patchImageSize = 2*int(mrScale)+1; // odd size
   float imageToPatchScale = float(patchImageSize) / float(par.patchSize);  // patch size in the image / patch size -> amount of down/up sampling
   
   // ok, do the interpolation
   bool touchesBoundary = interpolate(img, x, y, imageToPatchScale, 0., 0., imageToPatchScale, patch);
   return touchesBoundary;
}
