
// SIFTDlg.cpp : implementation file
//

#include "stdafx.h"
#include "SIFT.h"
#include "SIFTDlg.h"
#include "afxdialogex.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/objdetect/objdetect.hpp>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/face.hpp>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::face;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CSIFTDlg dialog



CSIFTDlg::CSIFTDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_SIFT_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CSIFTDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CSIFTDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CSIFTDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CSIFTDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CSIFTDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CSIFTDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CSIFTDlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON8, &CSIFTDlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON7, &CSIFTDlg::OnBnClickedButton7)
END_MESSAGE_MAP()


// CSIFTDlg message handlers

BOOL CSIFTDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here
	AllocConsole();
	FILE *stream;
	freopen_s(&stream, "CONOUT$", "w", stdout);
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CSIFTDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CSIFTDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CSIFTDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CSIFTDlg::OnBnClickedButton1()
{
	// TODO: Add your control notification handler code here

	//SIFT
	
	//to load image
	Mat img1 = imread("database/plane1.jpg", CV_LOAD_IMAGE_COLOR),
		img2 = imread("database/plane2.jpg", CV_LOAD_IMAGE_COLOR);

	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
	vector<KeyPoint> keyPoints1, keyPoints2;

	f2d->detect(img1, keyPoints1);
	f2d->detect(img2, keyPoints2);
	drawKeypoints(img1, keyPoints1, img1);
	drawKeypoints(img2, keyPoints2, img2);

	cout << "Total image 1 feature point: " << keyPoints1.size() << endl;
	cout << "Total image 2 feature point: " << keyPoints2.size() << endl;
	
	imshow("Image 1", img1);
	imshow("Image 2", img2);
	
	waitKey(0);
}

void CSIFTDlg::OnBnClickedButton2()
{
	// TODO: Add your control notification handler code here

	//Background Subtraction
	Mat frame, MOG2Frame;
	Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(500, 300);

	//to get video
	VideoCapture capture("database/bgSub_video.mp4");

	if (!capture.isOpened()) {
		//error in opening the video
		cerr << "Unable to open video" << endl;
	}

	while (true) {
		//to read the current frame
		if (!capture.read(frame)) {
			//videl endding or error in video
			break;
		}

		//to update the background model
		pMOG2->apply(frame, MOG2Frame);

		//to show frame and MOG2Frame
		imshow("Frame", frame);
		imshow("FG Mask MOG 2", MOG2Frame);

		waitKey(10);
	}

	capture.release();
	destroyAllWindows();
}

int pointNumber = 7, windowSize = 10, halfWindowSize = windowSize / 2;
Point2f half(halfWindowSize, halfWindowSize);
Scalar color(0, 0, 255);
vector<Point2f> initPoints;
Mat firstFrame;

//											flags param
static void onMouse(int event, int x, int y, int, void*) {
	if (event == EVENT_LBUTTONDOWN && initPoints.size() < pointNumber) {
		Point2f p((float)x, (float)y);
		initPoints.push_back(p);
		rectangle(firstFrame, p - half, p + half, color, CV_FILLED, 8);
		imshow("Preprocessing", firstFrame);
	}
}

void CSIFTDlg::OnBnClickedButton3()
{
	// TODO: Add your control notification handler code here

	//Optical Flow tracking Preprocessing

	//to get video
	VideoCapture capture("database/tracking_video.mp4");

	if (!capture.isOpened()) {
		//error in opening the video
		cerr << "Unable to open video" << endl;
	}

	if (!capture.read(firstFrame)) {
		//videl endding or error in video
		cerr << "cannot read frame" << endl;
	}

	//to read 7 points and window size
	imshow("Preprocessing", firstFrame);

	setMouseCallback("Preprocessing", onMouse, 0);

	while (initPoints.size() < 7) {
		if (waitKey(27) == 27) {
			break;
		}
	}

	fstream f;
	f.open("hw3_1.txt", ios::out);

	for (int i = 0; i < pointNumber; ++i) {
		f << "Point" << i + 1 << ":(" << initPoints[i].x << ',' << initPoints[i].y << ')' << endl;
	}

	f << "Window size:" << windowSize << endl;

	f.close();
	capture.release();
}

void WriteFramePoints(fstream& f, int& counter, vector<Point2f> points) {
	f << "frame " << counter++ << " : ";

	int temp = pointNumber - 1;
	for (int i = 0; i < temp; ++i) {
		f << '(' << (int)points[i].x << ',' << (int)points[i].y << "), ";
	}

	//last one
	f << '(' << (int)points[temp].x << ',' << (int)points[temp].y << ')' << endl;
}

void CSIFTDlg::OnBnClickedButton4()
{
	// TODO: Add your control notification handler code here

	//Optical Flow tracking Tracking whole video

	Mat frame, preFrame, grayFrame, grayPreFrame;
	vector<uchar> status;
	vector<float> err;

	//to get video
	VideoCapture capture("database/tracking_video.mp4");

	if (!capture.isOpened()) {
		//error in opening the video
		cerr << "Unable to open video" << endl;
	}

	//to read first frame
	if (!capture.read(preFrame)) {
		//videl endding or error in video
		cerr << "cannot read frame" << endl;
	}

	//to read first
	cvtColor(preFrame, grayPreFrame, COLOR_BGR2GRAY);

	//to read 7 points and window size
	vector<Point2f> currentPoints, futurePoints;
	vector<vector<Point2f>> pastPoints;
	fstream f;
	int windowSize, frameCounter = 1;

	f.open("hw3_1.txt", ios::in);

	if (initPoints.size() != pointNumber) {
		OnBnClickedButton3();
	}

	currentPoints = initPoints;
	pastPoints.push_back(currentPoints);

	/*
	for (int i = 0; i < pointNumber; ++i) {
		int x, y;
		f >> x >> y;
		Point2f p(x, y);
		currentPoints.push_back(p);
	}
	*/

	f.close();

	Scalar pastColor(0, 200, 255);

	//to write points into file
	f.open("hw3_2.txt", ios::out);
	WriteFramePoints(f, frameCounter, currentPoints);

	while (true) {
		if (!capture.read(frame)) {
			//videl endding or error in video
			break;
		}

		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		calcOpticalFlowPyrLK(grayPreFrame, grayFrame, currentPoints, futurePoints, status, err);

		grayFrame.copyTo(grayPreFrame);

		WriteFramePoints(f, frameCounter, futurePoints);
		pastPoints.push_back(futurePoints);
		currentPoints = futurePoints;
	}

	f.close();

	cout << "tracking endding" << endl;

	f.open("hw3_2.txt", ios::in);

	//to restart video
	capture.set(CV_CAP_PROP_POS_FRAMES, 0);
	frameCounter = 0;

	while (true) {
		if (!capture.read(frame)) {
			//videl endding or error in video
			break;
		}

		//to draw past moving trajectory
		int length = frameCounter - 1;
		for (int i = 0; i < length; ++i) {
			for (int j = 0; j < pointNumber; ++j) {
				line(frame, pastPoints[i][j], pastPoints[i + 1][j], pastColor);
			}
		}

		//to draw current points on frame
		for (int i = 0; i < pointNumber; ++i) {
			rectangle(frame, pastPoints[frameCounter][i] - half, pastPoints[frameCounter][i] + half, color, CV_FILLED, 8);
		}

		++frameCounter;

		imshow("Tracking whole video", frame);
		waitKey(10);
	}

	f.close();
	capture.release();
	destroyAllWindows();
}

string names[]{ "Harry Potter", "Hermione Granger", "Ron Weasley" };

void CSIFTDlg::OnBnClickedButton5()
{
	// TODO: Add your control notification handler code here

	//Face Recognition
	vector<Mat> imgs;
	vector<int> labels;
	
	imgs.push_back(imread("database/0.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	imgs.push_back(imread("database/1.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	imgs.push_back(imread("database/2.jpg", CV_LOAD_IMAGE_GRAYSCALE));

	labels.push_back(0);
	labels.push_back(1);
	labels.push_back(2);

	Ptr<BasicFaceRecognizer> model = createEigenFaceRecognizer(10);
	model->train(imgs, labels);

	Mat mean, dst, dst1;
	model->getMean().copyTo(mean);
	dst = mean.reshape(1, 120);

	normalize(dst, dst1, 0, 255, NORM_MINMAX, CV_8UC1);

	dst = imread("database/test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cout << "the image is : " << names[model->predict(dst)] << endl;
	
	imshow("Mean Face", dst1);
	imshow("Test Face", dst);
	waitKey(0);
}

void CSIFTDlg::OnBnClickedButton8()
{
	// TODO: Add your control notification handler code here

	//Face Detection
	Mat img = imread("database/face.jpg", CV_LOAD_IMAGE_COLOR), grayImg;
	cvtColor(img, grayImg, COLOR_BGR2GRAY);
	equalizeHist(grayImg, grayImg);
	
	vector<Rect> faces;
	CascadeClassifier classifier;
	
	if (!classifier.load("database/haarcascade_frontalface_alt.xml")) {
		cout << "loading classifier is failed" << endl;
	}

	classifier.detectMultiScale(grayImg, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(3, 3));

	int n = faces.size();
	Scalar color(0, 0, 255);
	
	for (int i = 0; i < n; ++i) {
		rectangle(img, faces[i].tl(), faces[i].br(), color, 3, 8);
	}

	cout << n << " faces detect" << endl;

	imshow("Detection", img);
	waitKey(0);
}

void CSIFTDlg::OnBnClickedButton7()
{
	// TODO: Add your control notification handler code here

	//Face Recognition & Detection

	//Face Recognition
	vector<Mat> imgs;
	vector<int> labels;

	imgs.push_back(imread("database/0.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	imgs.push_back(imread("database/1.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	imgs.push_back(imread("database/2.jpg", CV_LOAD_IMAGE_GRAYSCALE));

	labels.push_back(0);
	labels.push_back(1);
	labels.push_back(2);

	Ptr<FaceRecognizer> model = createEigenFaceRecognizer(10);
	model->train(imgs, labels);

	//Face Detection
	Mat img = imread("database/face.jpg", CV_LOAD_IMAGE_COLOR), grayImg, grayImg1;

	cvtColor(img, grayImg, COLOR_BGR2GRAY);
	grayImg.copyTo(grayImg1);
	equalizeHist(grayImg, grayImg);

	vector<Rect> faces;
	CascadeClassifier classifier;

	if (!classifier.load("database/haarcascade_frontalface_alt.xml")) {
		cout << "loading classifier is failed" << endl;
	}

	classifier.detectMultiScale(grayImg, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(3, 3));

	int n = faces.size();
	Size faceSize(100, 120);
	Scalar color(0, 0, 255);
	Point2i bias1(-20, -30), bias2(-20, -10);
	
	for (int i = 0; i < n; ++i) {
		Rect rect(faces[i].tl() + bias1, faceSize);
		Mat face;
		grayImg(rect).copyTo(face);
		
		int x = model->predict(face);
		//cout << "x = " << x << endl;

		rectangle(img, faces[i].tl(), faces[i].br(), color, 3, 8);
		putText(img, names[x], faces[i].tl() + bias2, CV_FONT_HERSHEY_PLAIN, 1, color, 2);
	}
	
	imshow("Face Recognition & Detection", img);
	waitKey(0);
}
