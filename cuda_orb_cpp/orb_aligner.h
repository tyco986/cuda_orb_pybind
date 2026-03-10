#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace orb {

struct OrbAlignerConfig {
    float ransac_thresh = 5.0f;
    int ransac_max_iters = 200;
    float ransac_confidence = 0.99f;
    float nndr = 0.75f;
    int max_pts = 10000;
    int noctaves = 5;
    int fast_threshold = 20;
};

/**
 * C++ OrbAligner: GPU (CUDA) or CPU (OpenCV) ORB detect+match + cv::findHomography (RANSAC).
 * Matches Python cuda_orb.OrbAligner behavior.
 * device >= 0: CUDA device ID; device < 0 (e.g. -1): use OpenCV ORB on CPU.
 */
class OrbAligner {
public:
    explicit OrbAligner(int device = 0, const OrbAlignerConfig& config = {});

    ~OrbAligner();

    OrbAligner(const OrbAligner&) = delete;
    OrbAligner& operator=(const OrbAligner&) = delete;

    /** Single pair: compute H (3x3) and motion (H[0][2], H[1][2]). Returns false if insufficient matches. */
    bool findTransform(const cv::Mat& template_img, const cv::Mat& image,
                       cv::Mat& H_out, cv::Vec2f& motion_out);

    /** Batch: process B pairs, append H and motion to output vectors. */
    void findTransformBatch(const std::vector<cv::Mat>& templates,
                            const std::vector<cv::Mat>& images,
                            std::vector<cv::Mat>& H_out,
                            std::vector<cv::Vec2f>& motion_out);

private:
    OrbAlignerConfig config_;
    int device_;
    struct Impl;
    Impl* impl_ = nullptr;

    void detectAndMatchCpu(const cv::Mat& img1, const cv::Mat& img2, float nndr,
                          std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2);
};

}  // namespace orb
