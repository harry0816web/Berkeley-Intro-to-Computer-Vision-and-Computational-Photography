import React from 'react';
import AssignmentHeader from '../components/AssignmentHeader';
import Method from '../components/Method';
import Footer from '../components/Footer';

const Assignment1Page = ({ 
  name = "Your Name",
  githubLink = "https://github.com/yourusername",
  email = "your.email@example.com",
  linkedinLink = "https://linkedin.com/in/yourprofile"
}) => {
  // Assignment 1 - Prokudin-Gorskii Colorization data
  const assignmentData = {
    title: "Prokudin-Gorskii Photo Colorization",
    mainImage: "/harvesters.tif_aligned.jpg",
    abstract: "This project implements automatic colorization of Prokudin-Gorskii glass plate photographs using image alignment techniques. The algorithm separates the three color channels (red, green, blue) from the stacked grayscale image and aligns them using both single-scale and multi-scale pyramid approaches with Sobel edge detection for improved accuracy.",
    methodExplanation: (
      <div>
        <h3>Overview</h3>
        <p>
          Prokudin-Gorskii photographs were captured using three separate color filters (red, green, blue) 
          on glass plates. To create a color image, we need to align these three channels precisely.
        </p>
        
        <h3>Methodology</h3>
        <p>
          The alignment process involves several key steps:
        </p>
        <ol>
          <li><strong>Image Preprocessing:</strong> Load the stacked grayscale image and split it into three equal parts representing the color channels.</li>
          <li><strong>Edge Detection:</strong> Apply Sobel edge detection to enhance structural features and improve alignment robustness.</li>
          <li><strong>Alignment Algorithms:</strong>
            <ul>
              <li>Single-scale alignment: Exhaustive search over a specified range</li>
              <li>Multi-scale pyramid alignment: Coarse-to-fine approach for efficiency</li>
            </ul>
          </li>
          <li><strong>Similarity Metrics:</strong> Use L2 distance, Normalized Cross Correlation (NCC), or Structural Similarity Index (SSIM) to evaluate alignment quality.</li>
          <li><strong>Final Composition:</strong> Apply the computed offsets to the original channels and combine them into a color image.</li>
        </ol>
        
        <h3>Technical Implementation</h3>
        <p>
          The pyramid approach builds multiple resolution levels, starting with coarse alignment at the smallest scale 
          and progressively refining the alignment at higher resolutions. This method is both computationally efficient 
          and robust to large displacements.
        </p>
      </div>
    ),
    results: [
      {
        title: "Cathedral Alignment",
        image: "/cathedral_result.jpg",
        description: "Successful alignment of cathedral image showing clear architectural details and proper color registration.",
        metrics: {
          "Green Offset": "(2, 5)",
          "Red Offset": "(3, 12)",
          "SSIM Score": "0.89"
        }
      },
      {
        title: "Monastery Alignment", 
        image: "/monastery_result.jpg",
        description: "Monastery image with well-aligned color channels, preserving fine details in the architectural elements.",
        metrics: {
          "Green Offset": "(-1, 3)",
          "Red Offset": "(2, 8)",
          "SSIM Score": "0.92"
        }
      },
      {
        title: "Tobolsk Alignment",
        image: "/tobolsk_result.jpg", 
        description: "Tobolsk image demonstrating the effectiveness of edge-based alignment on complex scenes.",
        metrics: {
          "Green Offset": "(1, 4)",
          "Red Offset": "(4, 10)",
          "SSIM Score": "0.87"
        }
      }
    ]
  };

  return (
    <>
      <AssignmentHeader
        assignmentNumber="Assignment 1"
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

export default Assignment1Page;
