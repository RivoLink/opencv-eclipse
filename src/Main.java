import java.util.LinkedList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Main {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat img1 = Imgcodecs.imread("./res/img_11.png");
        Mat img2 = Imgcodecs.imread("./res/img_12.png");

        stiching1(img1,img2);
    }

    public static void stiching1(Mat img1,Mat img2){

        Mat gray_image1 = new Mat();
        Mat gray_image2 = new Mat();

        Imgproc.cvtColor(img1, gray_image1, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.cvtColor(img2, gray_image2, Imgproc.COLOR_RGBA2GRAY);

        MatOfKeyPoint keyPoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keyPoints2 = new MatOfKeyPoint();

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.AKAZE);
        detector.detect(gray_image1, keyPoints1);
        detector.detect(gray_image2, keyPoints2);

        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();

        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.AKAZE);
        extractor.compute(gray_image1, keyPoints1, descriptors1);
        extractor.compute(gray_image2, keyPoints2, descriptors2);

        MatOfDMatch matches = new MatOfDMatch();

        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
        matcher.match(descriptors1, descriptors2, matches);

        double max_dist = 0; double min_dist = 200;
        List<DMatch> listMatches = matches.toList();

        for( int i = 0; i < listMatches.size(); i++ ) {
            double dist = listMatches.get(i).distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        System.out.println("Min: " + min_dist);
        System.out.println("Max: " + max_dist);

        LinkedList<DMatch> good_matches = new LinkedList<DMatch>();
        MatOfDMatch goodMatches = new MatOfDMatch();
        for(int i = 0; i < listMatches.size(); i++){
            if(listMatches.get(i).distance < 2*min_dist){
                good_matches.addLast(listMatches.get(i));
            }
        }

        goodMatches.fromList(good_matches);

        Mat matt = new Mat();
        Features2d.drawMatches(img1, keyPoints1, img2, keyPoints2, goodMatches, matt);
        Imgcodecs.imwrite("./out/matches.png", matt);

        Mat img_matches = new Mat(new Size(img1.cols()+img2.cols(),img1.rows()), CvType.CV_32FC2);

        LinkedList<Point> imgPoints1List = new LinkedList<Point>();
        LinkedList<Point> imgPoints2List = new LinkedList<Point>();
        List<KeyPoint> keypoints1List = keyPoints1.toList();
        List<KeyPoint> keypoints2List = keyPoints2.toList();

        for(int i = 0; i<good_matches.size(); i++){
            imgPoints1List.addLast(keypoints1List.get(good_matches.get(i).queryIdx).pt);
            imgPoints2List.addLast(keypoints2List.get(good_matches.get(i).trainIdx).pt);
        }

        MatOfPoint2f obj = new MatOfPoint2f();
        obj.fromList(imgPoints1List);
        MatOfPoint2f scene = new MatOfPoint2f();
        scene.fromList(imgPoints2List);

        Mat H = Calib3d.findHomography(obj, scene, Calib3d.RANSAC,3);

        Size s = new Size(img2.cols() + img1.cols(),img1.rows());

        Imgproc.warpPerspective(img1, img_matches, H, s);
        Mat m = new Mat(img_matches,new Rect(0,0,img2.cols(), img2.rows()));

        img2.copyTo(m);

        Imgcodecs.imwrite("./out/out.png", img_matches);
    }
}
