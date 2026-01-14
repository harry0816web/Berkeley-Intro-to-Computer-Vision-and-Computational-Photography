import React from 'react';
import AssignmentHeader from '../components/AssignmentHeader';
import Method from '../components/Method';
import Footer from '../components/Footer';

const Assignment3Page = ({ 
  name = "Your Name",
  githubLink = "https://github.com/yourusername",
  email = "your.email@example.com",
  linkedinLink = "https://linkedin.com/in/yourprofile"
}) => {
  // Assignment 3 - Image Stitching & Panoramas data
  const assignmentData = {
    title: "Image Stitching & Panoramas",
    mainImage: "/panorama_result.jpg",
    abstract: "This project implements automatic image stitching to create panoramic images from multiple overlapping photographs. The system uses feature detection, homography estimation, and advanced blending techniques to seamlessly combine images into high-quality panoramas.",
    methodExplanation: (
      <div>
        <h3>Overview</h3>
        <p>
          Image stitching combines multiple overlapping images to create a single, wider panoramic view. 
          This technique is essential for creating wide-angle views that exceed the field of view of a single camera.
        </p>
        
        <h3>Methodology</h3>
        <p>
          The panoramic stitching pipeline includes:
        </p>
        <ol>
          <li><strong>Feature Detection:</strong> Extract SIFT features from all input images.</li>
          <li><strong>Feature Matching:</strong> Find correspondences between overlapping image pairs.</li>
          <li><strong>Homography Estimation:</strong> Compute transformation matrices using RANSAC.</li>
          <li><strong>Image Warping:</strong> Transform images to a common coordinate system.</li>
          <li><strong>Blending:</strong> Seamlessly blend overlapping regions using multi-band blending.</li>
        </ol>
        
        <h3>Technical Implementation</h3>
        <p>
          The system handles various challenges including exposure differences, lens distortion, 
          and parallax effects to produce high-quality panoramic images with minimal artifacts.
        </p>
      </div>
    ),
    results: [
      {
        title: "Campus Panorama",
        image: "/campus_panorama.jpg",
        description: "360-degree panoramic view of the university campus created from 8 overlapping images.",
        metrics: {
          "Input Images": "8",
          "Stitching Time": "3.2s",
          "Final Resolution": "8192x2048"
        }
      },
      {
        title: "Mountain Landscape", 
        image: "/mountain_panorama.jpg",
        description: "Wide mountain landscape panorama with seamless blending and natural color transitions.",
        metrics: {
          "Input Images": "6",
          "Overlap": "30%",
          "Quality Score": "9.2/10"
        }
      },
      {
        title: "Urban Cityscape",
        image: "/city_panorama.jpg", 
        description: "Urban cityscape panorama showing detailed architectural features and proper perspective correction.",
        metrics: {
          "Input Images": "12",
          "Processing Time": "5.8s",
          "Feature Matches": "1,247"
        }
      }
    ]
  };

  return (
    <>
      <AssignmentHeader
        assignmentNumber="Assignment 3"
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

export default Assignment3Page;
