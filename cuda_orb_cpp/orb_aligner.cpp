#include "orb_aligner.h"
#include "orb.h"
#include "cuda_utils.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace orb {

struct OrbAligner::Impl {
    orb::Orbor orbor;
    orb::OrbData data1{}, data2{};
    unsigned char* d_img = nullptr;
    unsigned char* d_desc1 = nullptr;
    unsigned char* d_desc2 = nullptr;
    int alloc_pitch = 0;
    int alloc_height = 0;

    Impl(int max_pts, int noctaves, int edge_threshold, int fast_threshold,
         int patch_size, int wta_k) {
        orbor.initOrbData(data1, max_pts, true, true);
        orbor.initOrbData(data2, max_pts, true, true);
        orbor.init(noctaves, edge_threshold, wta_k, orb::HARRIS_SCORE,
                   patch_size, fast_threshold, -1, max_pts);
    }

    ~Impl() {
        if (d_img) cudaFree(d_img);
        if (d_desc1) cudaFree(d_desc1);
        if (d_desc2) cudaFree(d_desc2);
        orbor.freeOrbData(data1);
        orbor.freeOrbData(data2);
    }

    void ensureImageBuffer(int pitch, int h) {
        if (pitch > alloc_pitch || h > alloc_height) {
            if (d_img) cudaFree(d_img);
            alloc_pitch = std::max(pitch, alloc_pitch);
            alloc_height = std::max(h, alloc_height);
            CHECK(cudaMalloc((void**)&d_img,
                             static_cast<size_t>(alloc_pitch) * alloc_height));
        }
    }

    void detectAndMatch(const cv::Mat& img1, const cv::Mat& img2, float nndr,
                        std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2) {
        if (img1.empty() || img2.empty() || img1.type() != CV_8UC1 || img2.type() != CV_8UC1) {
            pts1.clear();
            pts2.clear();
            return;
        }
        int w1 = img1.cols, h1 = img1.rows;
        int w2 = img2.cols, h2 = img2.rows;
        int3 whp1{w1, h1, iAlignUp(w1, 128)};
        int3 whp2{w2, h2, iAlignUp(w2, 128)};

        ensureImageBuffer(std::max(whp1.z, whp2.z), std::max(h1, h2));

        CHECK(cudaMemcpy2D(d_img, whp1.z, img1.data, w1, w1, h1, cudaMemcpyHostToDevice));
        data1.num_pts = 0;
        orbor.detectAndCompute(d_img, data1, whp1, (void**)&d_desc1, true);

        CHECK(cudaMemcpy2D(d_img, whp2.z, img2.data, w2, w2, h2, cudaMemcpyHostToDevice));
        data2.num_pts = 0;
        orbor.detectAndCompute(d_img, data2, whp2, (void**)&d_desc2, true);

        if (data1.num_pts > 0 && data2.num_pts > 0 && d_desc1 && d_desc2)
            orbor.match(data1, data2, d_desc1, d_desc2, nndr);

        pts1.clear();
        pts2.clear();
        for (int i = 0; i < data1.num_pts; i++) {
            int k = data1.h_data[i].match;
            if (k >= 0) {
                pts1.push_back(cv::Point2f(static_cast<float>(data1.h_data[i].x),
                                          static_cast<float>(data1.h_data[i].y)));
                pts2.push_back(cv::Point2f(static_cast<float>(data2.h_data[k].x),
                                          static_cast<float>(data2.h_data[k].y)));
            }
        }
    }
};

OrbAligner::OrbAligner(int device, const OrbAlignerConfig& config)
    : config_(config), device_(device) {
    if (device >= 0) {
        initDevice(device);
        impl_ = new Impl(config_.max_pts, config_.noctaves, 31, config_.fast_threshold, 31, 2);
    } else {
        impl_ = nullptr;
    }
}

OrbAligner::~OrbAligner() {
    if (impl_) delete impl_;
}

void OrbAligner::detectAndMatchCpu(const cv::Mat& img1, const cv::Mat& img2, float nndr,
                                   std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2) {
    pts1.clear();
    pts2.clear();
    if (img1.empty() || img2.empty() || img1.type() != CV_8UC1 || img2.type() != CV_8UC1)
        return;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(config_.max_pts, 2.0f, config_.noctaves, 31, 0, 2,
                                           cv::ORB::HARRIS_SCORE, 31, config_.fast_threshold);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, desc2);
    if (desc1.empty() || desc2.empty() || kp1.empty() || kp2.empty())
        return;
    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    if (nndr >= 1.0f) {
        std::vector<cv::DMatch> matches;
        matcher.match(desc1, desc2, matches);
        for (const auto& m : matches) {
            pts1.push_back(kp1[m.queryIdx].pt);
            pts2.push_back(kp2[m.trainIdx].pt);
        }
    } else {
        std::vector<std::vector<cv::DMatch>> knn;
        matcher.knnMatch(desc1, desc2, knn, 2);
        for (const auto& m_n : knn) {
            if (m_n.size() == 2 && m_n[0].distance < nndr * m_n[1].distance) {
                pts1.push_back(kp1[m_n[0].queryIdx].pt);
                pts2.push_back(kp2[m_n[0].trainIdx].pt);
            } else if (m_n.size() == 1) {
                pts1.push_back(kp1[m_n[0].queryIdx].pt);
                pts2.push_back(kp2[m_n[0].trainIdx].pt);
            }
        }
    }
}

bool OrbAligner::findTransform(const cv::Mat& template_img, const cv::Mat& image,
                               cv::Mat& H_out, cv::Vec2f& motion_out) {
    cv::Mat tpl = template_img;
    cv::Mat img = image;
    if (template_img.channels() == 3)
        cv::cvtColor(template_img, tpl, cv::COLOR_BGR2GRAY);
    if (image.channels() == 3)
        cv::cvtColor(image, img, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> pts1, pts2;
    if (impl_) {
        impl_->detectAndMatch(tpl, img, config_.nndr, pts1, pts2);
    } else {
        detectAndMatchCpu(tpl, img, config_.nndr, pts1, pts2);
    }

    if (pts1.size() < 4) {
        H_out = cv::Mat::eye(3, 3, CV_32F);
        motion_out = cv::Vec2f(0, 0);
        return false;
    }

    cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, config_.ransac_thresh,
                                   cv::noArray(), config_.ransac_max_iters,
                                   config_.ransac_confidence);
    if (H.empty()) {
        H_out = cv::Mat::eye(3, 3, CV_32F);
        motion_out = cv::Vec2f(0, 0);
        return false;
    }
    H.convertTo(H_out, CV_32F);
    motion_out = cv::Vec2f(H_out.at<float>(0, 2), H_out.at<float>(1, 2));
    return true;
}

void OrbAligner::findTransformBatch(const std::vector<cv::Mat>& templates,
                                    const std::vector<cv::Mat>& images,
                                    std::vector<cv::Mat>& H_out,
                                    std::vector<cv::Vec2f>& motion_out) {
    const size_t B = std::min(templates.size(), images.size());
    H_out.reserve(H_out.size() + B);
    motion_out.reserve(motion_out.size() + B);
    for (size_t b = 0; b < B; b++) {
        cv::Mat H;
        cv::Vec2f mot;
        findTransform(templates[b], images[b], H, mot);
        H_out.push_back(H);
        motion_out.push_back(mot);
    }
}

}  // namespace orb
