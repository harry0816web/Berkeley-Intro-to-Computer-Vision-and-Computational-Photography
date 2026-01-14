import React from 'react';
import AssignmentHeader from '../components/AssignmentHeader';
import Method from '../components/Method';
import Footer from '../components/Footer';

const Assignment4Page = ({ 
  name = "Your Name",
  githubLink = "https://github.com/yourusername",
  email = "your.email@example.com",
  linkedinLink = "https://linkedin.com/in/yourprofile"
}) => {
  // Assignment 4 - Object Detection & Recognition data
  const assignmentData = {
    title: "Object Detection & Recognition",
    mainImage: "/object_detection_result.jpg",
    abstract: "This project implements object detection and recognition using deep learning approaches. The system can identify and localize multiple objects within images, providing bounding box coordinates and classification confidence scores for various object categories.",
    methodExplanation: (
      <div>
        <h3>Overview</h3>
        <p>
          Object detection and recognition is a computer vision task that involves identifying objects 
          in images and determining their locations through bounding box coordinates and class labels.
        </p>
        
        <h3>Methodology</h3>
        <p>
          The object detection pipeline includes:
        </p>
        <ol>
          <li><strong>Data Preprocessing:</strong> Image normalization and augmentation for training data.</li>
          <li><strong>Model Architecture:</strong> Implementation of YOLO (You Only Look Once) architecture.</li>
          <li><strong>Training:</strong> End-to-end training with multi-task loss function.</li>
          <li><strong>Inference:</strong> Real-time object detection with confidence thresholding.</li>
          <li><strong>Post-processing:</strong> Non-maximum suppression for duplicate detection removal.</li>
        </ol>
        
        <h3>Technical Implementation</h3>
        <p>
          The system uses a single-stage detection approach that predicts bounding boxes and class 
          probabilities directly from input images, enabling real-time performance on various hardware platforms.
        </p>
      </div>
    ),
    results: [
      {
        title: "COCO Dataset Results",
        image: "/coco_detection.jpg",
        description: "Object detection results on COCO validation set showing accurate localization and classification.",
        metrics: {
          "mAP@0.5": "0.68",
          "mAP@0.5:0.95": "0.42",
          "FPS": "45"
        }
      },
      {
        title: "Real-time Detection", 
        image: "/realtime_detection.jpg",
        description: "Real-time object detection on video stream with multiple object categories identified.",
        metrics: {
          "Processing Time": "22ms",
          "Objects Detected": "7",
          "Confidence": "0.89"
        }
      },
      {
        title: "Custom Dataset",
        image: "/custom_detection.jpg", 
        description: "Detection results on custom dataset with fine-tuned model for specific object categories.",
        metrics: {
          "Custom Classes": "15",
          "Accuracy": "92.3%",
          "Precision": "0.91"
        }
      }
    ]
  };

  return (
    <>
      <AssignmentHeader
        assignmentNumber="Assignment 4"
        title={assignmentData.title}
        mainImage={assignmentData.mainImage}
        abstract={assignmentData.abstract}
      />
      
      <Method
        methodTitle="Methodology & Results"
        methodExplanation={assignmentData.methodExplanation}
        results={assignmentData.results}
      />
      
      <Footer
        name={name}
        githubLink={githubLink}
        email={email}
        linkedinLink={linkedinLink}
      />
    </>
  );
};

export default Assignment4Page;
