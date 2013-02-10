#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vtkSmartPointer.h>
#include <vtkStructuredPoints.h>
#include <vtkPointData.h>
#include <vtkPLYWriter.h>
#include <vtkFloatArray.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkMarchingCubes.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>

const int IMG_WIDTH = 720;
const int IMG_HEIGHT = 576;
const int VOXEL_DIM = 256;
const int VOXEL_SIZE = VOXEL_DIM*VOXEL_DIM*VOXEL_DIM;
const int VOXEL_SLICE = VOXEL_DIM*VOXEL_DIM;

struct voxel {
    float xpos;
    float ypos;
    float zpos;
    float res;
    float value;
};

struct coord {
    int x;
    int y;
};

struct startParams {
    float startX;
    float startY;
    float startZ;
    float voxelWidth;
    float voxelHeight;
    float voxelDepth;
};

struct camera {
    cv::Mat Image;
    cv::Mat P;
    cv::Mat K;
    cv::Mat R;
    cv::Mat t;
    cv::Mat Silhouette;
};

void exportModel(char *filename, vtkPolyData *polyData) {
    
    /* exports 3d model in ply format */
    vtkSmartPointer<vtkPLYWriter> plyExporter = vtkSmartPointer<vtkPLYWriter>::New();
    plyExporter->SetFileName(filename);
    plyExporter->SetInput(polyData);
    plyExporter->Update();
    plyExporter->Write();
}

void renderModel(float fArray[], startParams params) {
    
    /* create vtk visualization pipeline from voxel grid (float array) */
    vtkSmartPointer<vtkStructuredPoints> sPoints = vtkSmartPointer<vtkStructuredPoints>::New();
    sPoints->SetDimensions(VOXEL_DIM, VOXEL_DIM, VOXEL_DIM);
    sPoints->SetSpacing(params.voxelWidth, params.voxelHeight, params.voxelDepth);
    sPoints->SetOrigin(params.startX, params.startY, params.startZ);
    sPoints->SetScalarTypeToFloat();
    
    vtkSmartPointer<vtkFloatArray> vtkFArray = vtkSmartPointer<vtkFloatArray>::New();
    vtkFArray->SetNumberOfValues(VOXEL_SIZE);
    vtkFArray->SetArray(fArray, VOXEL_SIZE, 1);
    
    sPoints->GetPointData()->SetScalars(vtkFArray);
    sPoints->Update();
    
    /* create iso surface with marching cubes algorithm */
    vtkSmartPointer<vtkMarchingCubes> mcSource = vtkSmartPointer<vtkMarchingCubes>::New();
    mcSource->SetInputConnection(sPoints->GetProducerPort());
    mcSource->SetComputeNormals(1);
    mcSource->SetNumberOfContours(1);
    mcSource->SetValue(0,0.5);
    mcSource->Update();
    
    /* usual render stuff */
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->SetBackground(.45, .45, .9);
    renderer->SetBackground2(.0, .0, .0);
    renderer->GradientBackgroundOn();
    
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(renderWindow);
    
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(mcSource->GetOutputPort());
    
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    
    /* visible light properties */
    actor->GetProperty()->SetSpecular(0.25);
    actor->GetProperty()->SetAmbient(0.25);
    actor->GetProperty()->SetDiffuse(0.25);
    actor->GetProperty()->SetInterpolationToPhong();
    renderer->AddActor(actor);
    
    renderWindow->Render();
    interactor->Start();
}

coord project(camera cam, voxel v) {
    
    coord im;
    
    /* project voxel into camera image coords */
    float z =   cam.P.at<float>(2, 0) * v.xpos +
                cam.P.at<float>(2, 1) * v.ypos +
                cam.P.at<float>(2, 2) * v.zpos +
                cam.P.at<float>(2, 3);
        
    im.y =    (cam.P.at<float>(1, 0) * v.xpos +
               cam.P.at<float>(1, 1) * v.ypos +
               cam.P.at<float>(1, 2) * v.zpos +
               cam.P.at<float>(1, 3)) / z;

    im.x =    (cam.P.at<float>(0, 0) * v.xpos +
               cam.P.at<float>(0, 1) * v.ypos +
               cam.P.at<float>(0, 2) * v.zpos +
               cam.P.at<float>(0, 3)) / z;
    
    return im;
}

void carve(float fArray[], startParams params, camera cam) {
        
    for (int i=0; i<VOXEL_DIM; i++) {
        for (int j=0; j<VOXEL_DIM; j++) {
            for (int k=0; k<VOXEL_DIM; k++) {
                
                /* calc voxel position inside camera view frustum */
                voxel v;
                v.xpos = params.startX + i * params.voxelWidth;
                v.ypos = params.startY + j * params.voxelHeight;
                v.zpos = params.startZ + k * params.voxelDepth;
                v.value = 1.0f;
                
                coord im = project(cam, v);
                                
                /* test if projected voxel is within image coords */
                if (im.x > 0 && im.y > 0 && im.x < IMG_WIDTH && im.y < IMG_HEIGHT) {
                    
                    /* clear any that are not inside the silhouette (color == 0) */
                    if (cam.Silhouette.at<uchar>(im.y, im.x) == 0) {
                        fArray[i*VOXEL_SLICE+j*VOXEL_DIM+k] += v.value;
                    } 
                }
                
            }
        }
    }
    
}

int main(int argc, char* argv[]) {
    
    /* acquire camera images, silhouettes and camera matrix */
    std::vector<camera> cameras;
    cv::FileStorage fs("../../assets/viff.xml", cv::FileStorage::READ);
    for (int i=0; i<36; i++) {
        
        /* camera image */
        std::stringstream simg;
        simg << "../../assets/viff." << std::setfill('0') << std::setw(3) << i << ".ppm";
        cv::Mat img = cv::imread(simg.str());
        
        /* silhouette */
        cv::Mat silhouette;
        cv::cvtColor(img, silhouette, CV_BGR2HSV);
        cv::inRange(silhouette, cv::Scalar(100, 20, 65), cv::Scalar(160,255,255), silhouette);
        
        /* camera matrix */
        std::stringstream smat;
        smat << "viff" << std::setfill('0') << std::setw(3) << i << "_matrix";
        cv::Mat P;
        fs[smat.str()] >> P;
        
        /* decompose proj matrix to cam- and rot matrix and trans vect */
        cv::Mat K, R, t;
        cv::decomposeProjectionMatrix(P, K, R, t);
        
        camera c;
        c.Image = img;
        c.P = P;
        c.K = K;
        c.R = R;
        c.t = t;
        c.Silhouette = silhouette;
                
        cameras.push_back(c);
    }
    
    /* bounding box dimensions of dinosaur */
    float xmin = -0.13066, ymin = -0.13066, zmin = -0.74832;
    float xmax = 0.11556, ymax = 0.11556, zmax = -0.50210;
            
    float bbwidth = std::abs(xmax-xmin);
    float bbheight = std::abs(ymax-ymin);
    float bbdepth = std::abs(zmax-zmin);
    
    startParams params;
    params.startX = xmin - bbwidth*0.005;
    params.startY = ymin - bbheight*0.005;
    params.startZ = zmin - bbdepth*0.005;
    params.voxelWidth = bbwidth*1.01/VOXEL_DIM;
    params.voxelHeight = bbheight*1.01/VOXEL_DIM;
    params.voxelDepth = bbdepth*1.01/VOXEL_DIM;
    
    /* 3 dimensional voxel grid */
    float *fArray = new float[VOXEL_SIZE];
    std::fill_n(fArray, VOXEL_SIZE, 0.0f);
    
    /* carving model for every given camera image */
    for (int i=0; i<36; i++) {
        carve(fArray, params, cameras.at(i));
    }
    
    /* remove anything outside the 3d model */
    for (int i = 0; i < VOXEL_SIZE; i++) {
        fArray[i] = 1.0f ? fArray[i] == 36.0f : 0.0f;
    }

    cv::imshow("Dino" , cameras.at(1).Image);
    cv::imshow("Dino Silhouette", cameras.at(1).Silhouette);
    renderModel(fArray, params);
    
    return 0;
}
