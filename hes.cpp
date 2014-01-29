/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#include <iostream>
#include <fstream>

#include "pyramid.h"
#include "helpers.h"
#include "siftdesc.h"
#include "sim.h"

using namespace cv;
using namespace std;

struct HessianSimParams
{
   float threshold;
   float desc_factor;
   int   patch_size;
   bool  verbose;
   HessianSimParams()
      {
         threshold = 16.0f/3.0f;
         desc_factor = 3.0f*sqrt(3.0f);
         patch_size = 41;
         verbose = false;
      }
};

int g_numberOfPoints = 0;
int g_numberOfDescriptors = 0;

struct Keypoint
{
   float x, y, s;
   float response;
   int type;
   unsigned char desc[128];
};

// Not really similarity invariant, missing dominant orientation
struct SimHessianDetector : public HessianDetector, SimShape, HessianKeypointCallback, SimShapeCallback
{
   const Mat image;
   SIFTDescriptor sift;
   vector<Keypoint> keys;
public:
   SimHessianDetector(const Mat &image, const PyramidParams &par, const SimShapeParams &ap, const SIFTDescriptorParams &sp) :
      HessianDetector(par),
      SimShape(ap),
      image(image),
      sift(sp)
      {
         this->setHessianKeypointCallback(this);
         this->setSimShapeCallback(this);
      }

   void onSimShapeFound(
      const cv::Mat &blur,     // corresponding scale level
      float x, float y,     // subpixel, image coordinates
      float s,              // scale
      float pixelDistance,  // distance between pixels in provided blured image
      int type, float response, int iters) {

   }

   void onHessianKeypointDetected(const Mat &blur, float x, float y, float s, float pixelDistance, int type, float response)
      {
         g_numberOfPoints++;
         // ideally call find rotation...
         //findAffineShape(blur, x, y, s, pixelDistance, type, response);
         // convert shape into a up is up frame

         // TODO possibility extract keypoint from the blur...
         // TODO dominant orientation...

         // now sample the patch
         if (!normalize(image, x, y, s))
         {
            // compute SIFT
            sift.computeSiftDescriptor(this->patch);
            // store the keypoint
            keys.push_back(Keypoint());
            Keypoint &k = keys.back();
            k.x = x; k.y = y; k.s = s; k.response = response; k.type = type;
            for (int i=0; i<128; i++)
               k.desc[i] = (unsigned char)sift.vec[i];
            // debugging stuff
            if (0)
            {
               cout << "x: " << x << ", y: " << y
                    << ", s: " << s << ", pd: " << pixelDistance
                    << ", t: " << type << ", r: " << response << endl;
               for (size_t i=0; i<sift.vec.size(); i++)
                  cout << " " << sift.vec[i];
               cout << endl;
            }
            g_numberOfDescriptors++;
         }
      }

   void exportKeypoints(ostream &out)
      {
         out << 128 << endl;
         out << keys.size() << endl;
         for (size_t i=0; i<keys.size(); i++)
         {
            Keypoint &k = keys[i];

            float sc = SimShape::par.mrSize * k.s;
            float sc_f = 1./(sc*sc);

            out << k.x << " " << k.y << " " <<sc_f << " " << 0. << " " << sc_f;
            for (size_t i=0; i<128; i++)
               out << " " << int(k.desc[i]);
            out << endl;
         }
      }
};




int main(int argc, char **argv)
{
   if (argc>1)
   {
      Mat tmp = imread(argv[1]);
      Mat image(tmp.rows, tmp.cols, CV_32FC1, Scalar(0));
      
      float *out = image.ptr<float>(0);
      unsigned char *in  = tmp.ptr<unsigned char>(0); 

      for (size_t i=tmp.rows*tmp.cols; i > 0; i--)
      {
         *out = (float(in[0]) + in[1] + in[2])/3.0f;
         out++;
         in+=3;
      }
      
      HessianSimParams par;
      double t1 = 0;
      {
         // copy params 
         PyramidParams p;
         p.threshold = par.threshold;
         
         SimShapeParams ap;
         ap.mrSize = par.desc_factor;
         
         SIFTDescriptorParams sp;
         sp.patchSize = par.patch_size;
                
         SimHessianDetector detector(image, p, ap, sp);
         t1 = getTime(); g_numberOfPoints = 0;
         detector.detectPyramidKeypoints(image);
         cout << "Detected " << g_numberOfPoints << " keypoints and " << g_numberOfDescriptors << " affine shapes in " << getTime()-t1 << " sec." << endl;

         char suffix[] = ".hes.sift";
         int len = strlen(argv[1])+strlen(suffix)+1;
         char buf[len];
         snprintf(buf, len, "%s%s", argv[1], suffix); buf[len-1]=0;      
         ofstream out(buf);
         detector.exportKeypoints(out);
      }
   } else {
      printf("\nUsage: hesaff image_name.ppm\nDetects Hessian Affine points and describes them using SIFT descriptor.\nThe detector assumes that the vertical orientation is preserved.\n\n");
   }
}
