/*
 * Copyright (C) 2008-12 Michal Perdoch, 2014 Karel Lenc
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/program_options.hpp>

#include "pyramid.h"
#include "helpers.h"
#include "affine.h"
#include "siftdesc.h"

using namespace cv;
using namespace std;

#define RET_ERROR 1
#define RET_SUCCESS 0

#define GET_OPTION(VM,OPT,NAME,TYPE) do { \
   if (VM.count(#NAME)) OPT.NAME = VM[#NAME].as<TYPE>();\
   } while (0)

enum OutputFormat {
   OF_OXFORD = 1,
   OF_DENAFF
};

struct HessianAffineParams
{
   float threshold;
   int   max_iter;
   float desc_factor;
   int   patch_size;
   bool  rot_invariant;
   bool  aff_invariant;
   bool  fast_norm;
   bool  verbose;
   OutputFormat keypoint_format;
   HessianAffineParams()
   {
      threshold = 10;//16.0f/3.0f;
      max_iter = 16;
      desc_factor = 3.0f*sqrt(3.0f);
      patch_size = 41;
      rot_invariant = true;
      aff_invariant = false;
      fast_norm = false;
      verbose = false;
      keypoint_format = OF_OXFORD;
   }
};

int g_numberOfPoints = 0;
int g_numberOfDescriptors = 0;

struct Keypoint
{
   int drid;
   float x, y, s, angle;
   float a11,a12,a21,a22;
   float response;
   int type;
   unsigned char desc[128];
};

struct AffineHessianDetector : public HessianDetector, AffineShape, HessianKeypointCallback, AffineShapeCallback
{
   const Mat image;
   SIFTDescriptor sift;
   vector<Keypoint> keys;
   bool aff_invariant;
   bool rot_invariant;
   bool fast_norm;
   vector<float> angles;
public:
   AffineHessianDetector(const Mat &image, const PyramidParams &par,
                         const AffineShapeParams &ap,
                         const SIFTDescriptorParams &sp,
                         bool aff_inv, bool rot_inv) :
      HessianDetector(par),
      AffineShape(ap),
      image(image),
      sift(sp),
      aff_invariant(aff_inv),
      rot_invariant(rot_inv)
   {
      this->setHessianKeypointCallback(this);
      this->setAffineShapeCallback(this);
      angles.reserve(10);
   }

   void onHessianKeypointDetected(const Mat &blur, float x, float y, float s,
                                  float pixelDistance, int type, float response)
   {
      g_numberOfPoints++;
      if (aff_invariant){
         findAffineShape(blur, x, y, s, pixelDistance, type, response);
      } else {
         onAffineShapeFound(blur, x, y, s, pixelDistance,
                            1., 0., 0., 1., type, response, 0);
      }
   }
   
   void onAffineShapeFound(
         const Mat &blur, float x, float y, float s, float pixelDistance,
         float a11, float a12,
         float a21, float a22,
         int type, float response, int iters)
   {

      angles.clear();
      if (rot_invariant) {
         if (!normalize(image, x, y, s, a11, a12, a21, a22)) {
            computeHistAngles(this->patch, angles);
         }
      } else {
         angles.push_back(0.);
      }


      for (unsigned int ai = 0; ai < angles.size(); ++ai) {
         float angle = angles.at(ai);
         float ra11, ra12, ra21, ra22;

         if (angle != 0){
            // Rotate the affine transformation accordingly
            float lecos=cos(-angle), lesin=sin(-angle);
            ra11 = (a11*lecos-a12*lesin);
            ra12 = (a11*lesin+a12*lecos);
            ra21 = (a21*lecos-a22*lesin);
            ra22 = (a21*lesin+a22*lecos);
         } else {
            ra11 = a11; ra12 = a12; ra21 = a21; ra22 = a22;
         }

         // now sample the patch
         if (!normalize(image, x, y, s, ra11, ra12, ra21, ra22))
         {
            // compute SIFT
            sift.computeSiftDescriptor(this->patch);
            // store the keypoint
            keys.push_back(Keypoint());
            Keypoint &k = keys.back();
            k.drid = g_numberOfPoints;
            k.x = x; k.y = y; k.s = s; k.angle = angle;
            k.a11 = ra11; k.a12 = ra12; k.a21 = ra21; k.a22 = ra22;
            k.response = response; k.type = type;
            for (int i=0; i<128; i++)
               k.desc[i] = (unsigned char)sift.vec[i];
            // debugging stuff
            if (0)
            {
               cout << "x: " << x << ", y: " << y
                    << ", s: " << s << ", pd: " << pixelDistance
                    << ", a11: " << ra11 << ", a12: " << ra12
                    << ", a21: " << ra21 << ", a22: " << ra22
                    << ", t: " << type << ", r: " << response << endl;
               for (size_t i=0; i<sift.vec.size(); i++)
                  cout << " " << sift.vec[i];
               cout << endl;
            }
            g_numberOfDescriptors++;
         }
      }
   }

   void exportKeypointsOxf(ostream &out)
   {
      out << 128 << endl;
      out << keys.size() << endl;
      for (size_t i=0; i<keys.size(); i++)
      {
         Keypoint &k = keys[i];
         
         float sc = AffineShape::par.mrSize * k.s;
         Mat A = (Mat_<float>(2,2) << k.a11, k.a12, k.a21, k.a22);
         SVD svd(A, SVD::FULL_UV);

         float *d = (float *)svd.w.data;
         d[0] = 1.0f/(d[0]*d[0]*sc*sc);
         d[1] = 1.0f/(d[1]*d[1]*sc*sc);

         A = svd.u * Mat::diag(svd.w) * svd.u.t();

         out << k.x << " " << k.y << " " << A.at<float>(0,0) << " " << A.at<float>(0,1) << " " << A.at<float>(1,1);
         for (size_t i=0; i<128; i++)
            out << " " << int(k.desc[i]);
         out << endl;
      }
   }
   void exportKeypointsDenormAff(ostream &out)
   {
      out << 128 << endl;
      out << keys.size() << endl;
      for (size_t i=0; i<keys.size(); i++)
      {
         Keypoint &k = keys[i];

         float sc = (AffineShape::par.mrSize * k.s);
         // Format: DRID X Y A11 A21 A12 A22 - matlab-like stacking
         out << k.drid << " " << k.x << " " << k.y << " " << k.a11*sc << " " << k.a21*sc << " " << k.a12*sc << " " << k.a22*sc;
         for (size_t i=0; i<128; i++)
            out << " " << int(k.desc[i]);
         out << endl;
      }
   }
};

int main(int argc, char **argv)
{
   string imageFileName;
   string outputFileName;
   HessianAffineParams par;

   namespace po = boost::program_options;
   po::options_description desc("Options");

   desc.add_options()
         ("image,i", po::value<string>(&imageFileName)->required(),
          "input image")
         ("output,o",  po::value<string>(&outputFileName),
          "output feature file [<input_image>.hesaff.sift]")
         ("keypoint_format,k",  po::value<int>(),
          "set output keypoint format (oxford=1/affdenorm=2) [1]")
         ("threshold,t", po::value<float>(&par.threshold),
          "min. response threshold [5.3]")
         ("rot_invariant,r",  po::value<bool>(&par.rot_invariant),
          "set rotation invariance [off]")
         ("aff_invariant,a",  po::value<bool>(&par.aff_invariant),
          "set affine invariance [on]")
         ("max_iter", po::value<int>(&par.max_iter),
          "max number of aff. adapt. iterations [16]")
         ("desc_factor,m", po::value<float>(&par.desc_factor),
          "magnification factor for measurement region [5.2]")
         ("patch_size", po::value<int>(&par.patch_size),
          "SIFT patch size [41]")
         ("fast_norm,f",  po::value<bool>(&par.fast_norm),
          "set fast patch normalisation [on]")
         ("help,h", "Print help messages")
         ("verbose,v", "set verbosity level")
         ;

   po::positional_options_description positionalOptions;
   positionalOptions.add("image", 1);

   po::variables_map vm;

   try
   {
      po::store(po::command_line_parser(argc, argv).options(desc)
                .positional(positionalOptions).run(),
                vm);

      if ( vm.count("help")  )
      {
         cout << "Hessian Affine Feature Detector" << std::endl
              << "Detect affine covariant feature points and describes " << std::endl
              << "them using SIFT descriptor. Available options:"
              << std::endl << std::endl
              << desc << std::endl
              << "Output formats:" << std::endl << std::endl
              << "OXFORD: X Y S11 S12 S22 <SIFT_DESC>\n"
                 "X, Y are coordiantes of the center. S11, S12, S22 are \n"
                 "the elements of a 2x2 covariance matrix S (a positive\n"
                 "semidefinite matrix) defining the ellipse shape. The ellipse\n"
                 "is the set of points {x + T: x' S x = 1}, where T is the center.\n\n"
              << "AFFNORM: DRID X Y A11 A21 A12 A22 <SIFT_DESC> - matlab-like stacking\n"
                 "The ellipse is obtaine by transforming a unit circle by A as the \n"
                 "set of points {A x + T : |x| = 1}, where T is the center.\n"
                 "DRID is the unique id of a distinguished region.\n"
                 "Ellipse and affine frame definitions from vl_plotframe\n"
                 "documentation of VLFeat library (vlfeat.org).\n"
              << std::endl;
         return RET_SUCCESS;
      }
      po::notify(vm);
   }
   catch(boost::program_options::required_option& e)
   {
      cerr << "ERROR: " << e.what() << endl << endl
           << desc << endl;
      return RET_ERROR;
   }
   catch(boost::program_options::error& e)
   {
      cerr << "ERROR: " << e.what() << endl << endl
           << desc << endl;
      return RET_ERROR;
   }

   if (vm.count("verbose")) par.verbose = true;
   if (vm.count("keypoint_format")) {
      par.keypoint_format = (OutputFormat)vm["keypoint_format"].as<int>();
   }
   if (!vm.count("output")) {
      stringstream ss;
      ss << imageFileName << ".hes.sift";
      outputFileName = ss.str();
   }

   if (par.verbose) {
      cout << "Option values:" << endl
           << "image\t\t: " << imageFileName << endl
           << "output\t\t: " << outputFileName << endl
           << "threshold\t: " << par.threshold << endl
           << "max_iter\t: " << par.max_iter << endl
           << "desc_factor\t: " << par.desc_factor << endl
           << "patch_size\t: " << par.patch_size << endl
           << "rot_invariant\t: " << par.rot_invariant << endl
           << "aff_invariant\t: " << par.aff_invariant << endl
           << "fast_norm\t: " << par.fast_norm << endl
           << "keypoint_format\t: " << par.keypoint_format << endl;
   }

   Mat tmp = imread(imageFileName);
   Mat image(tmp.rows, tmp.cols, CV_32FC1, Scalar(0));

   float *out = image.ptr<float>(0);
   unsigned char *in  = tmp.ptr<unsigned char>(0);

   for (size_t i=tmp.rows*tmp.cols; i > 0; i--)
   {
      *out = (float(in[0]) + in[1] + in[2])/3.0f;
      out++;
      in+=3;
   }

   double t1 = 0;
   {
      // copy params
      PyramidParams p;
      p.threshold = par.threshold;

      AffineShapeParams ap;
      ap.maxIterations = par.max_iter;
      ap.patchSize = par.patch_size;
      ap.mrSize = par.desc_factor;
      ap.fastNorm = par.fast_norm;

      SIFTDescriptorParams sp;
      sp.patchSize = par.patch_size;

      AffineHessianDetector detector(image, p, ap, sp, par.aff_invariant, par.rot_invariant);
      t1 = getTime(); g_numberOfPoints = 0;
      detector.detectPyramidKeypoints(image);
      cout << "Detected " << g_numberOfPoints << " keypoints and " << g_numberOfDescriptors << " affine shapes in " << getTime()-t1 << " sec." << endl;

      ofstream out(outputFileName.c_str());
      switch (par.keypoint_format){
      case OF_OXFORD:
         detector.exportKeypointsOxf(out);
         break;
      case OF_DENAFF:
         detector.exportKeypointsDenormAff(out);
         break;
      }
   }
}
