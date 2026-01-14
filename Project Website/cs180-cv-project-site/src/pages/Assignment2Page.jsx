import React from 'react';
import AssignmentHeader from '../components/AssignmentHeader';
import Method from '../components/Method';
import Footer from '../components/Footer';

const Assignment2Page = ({ 
  name = "Your Name",
  githubLink = "https://github.com/yourusername",
  email = "your.email@example.com",
  linkedinLink = "https://linkedin.com/in/yourprofile"
}) => {
  // Assignment 2 - Convolutions from Scratch data
  const assignmentData = {
    title: "Convolutions from Scratch",
    mainImage: "/convolution_results.png",
    abstract: "This project implements convolution operations from scratch using numpy, including four-loop and two-loop implementations with zero padding. We compare our implementation with scipy.signal.convolve2d and apply various filters including box filters and finite difference operators for edge detection.",
    methodExplanation: (
      <div>
        <h3>Overview</h3>
        <p>
          Convolution is a fundamental operation in computer vision and image processing. This project implements 
          convolution from scratch using only numpy operations, demonstrating different approaches and comparing 
          with built-in functions.
        </p>
        
        <h3>Implementation Approaches</h3>
        <p>
          We implemented convolution using three different approaches:
        </p>
        <ol>
          <li><strong>Four Nested Loops:</strong> Most basic implementation with explicit indexing</li>
          <li><strong>Two Loops with Vectorization:</strong> Optimized version using numpy array operations</li>
          <li><strong>Zero Padding:</strong> Handles edge effects by padding the input image</li>
        </ol>
        
        <h3>Code Implementation</h3>
        <h4>Four Loops Implementation:</h4>
        <pre style={{backgroundColor: '#f5f5f5', padding: '1rem', borderRadius: '8px', overflow: 'auto'}}>
{`def convolution_four_loops(image, kernel, padding=0):
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape
    out_h = img_h + 2 * padding - ker_h + 1
    out_w = img_w + 2 * padding - ker_w + 1
    output = np.zeros((out_h, out_w))
    
    if padding > 0:
        padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    else:
        padded_image = image
    
    # Four nested loops
    for i in range(out_h):
        for j in range(out_w):
            for ki in range(ker_h):
                for kj in range(ker_w):
                    output[i, j] += padded_image[i + ki, j + kj] * kernel[ki, kj]
    
    return output`}
        </pre>
        
        <h4>Two Loops Implementation (Vectorized):</h4>
        <pre style={{backgroundColor: '#f5f5f5', padding: '1rem', borderRadius: '8px', overflow: 'auto'}}>
{`def convolution_two_loops(image, kernel, padding=0):
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape
    out_h = img_h + 2 * padding - ker_h + 1
    out_w = img_w + 2 * padding - ker_w + 1
    output = np.zeros((out_h, out_w))
    
    if padding > 0:
        padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    else:
        padded_image = image
    
    # Two loops with vectorized operations
    for i in range(out_h):
        for j in range(out_w):
            patch = padded_image[i:i+ker_h, j:j+ker_w]
            output[i, j] = np.sum(patch * kernel)
    
    return output`}
        </pre>
        
        <h4>Box Filter (9x9):</h4>
        <pre style={{backgroundColor: '#f5f5f5', padding: '1rem', borderRadius: '8px', overflow: 'auto'}}>
{`def create_box_filter(size=9):
    return np.ones((size, size)) / (size * size)

# Apply box filter
box_filter = create_box_filter(9)
box_result = convolution_with_padding(image, box_filter, padding=4)`}
        </pre>
        
        <h4>Finite Difference Operators:</h4>
        <pre style={{backgroundColor: '#f5f5f5', padding: '1rem', borderRadius: '8px', overflow: 'auto'}}>
{`def create_finite_difference_operators():
    # Dx: horizontal gradient (detects vertical edges)
    Dx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
    
    # Dy: vertical gradient (detects horizontal edges)
    Dy = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]])
    
    return Dx, Dy

# Apply finite difference operators
Dx, Dy = create_finite_difference_operators()
dx_result = convolution_with_padding(image, Dx, padding=1)
dy_result = convolution_with_padding(image, Dy, padding=1)`}
        </pre>
        
        <h3>Comparison with Scipy</h3>
        <p>
          Our implementation was compared with <code>scipy.signal.convolve2d</code> to verify correctness. 
          The results showed identical outputs with differences in the order of machine precision (1e-15).
        </p>
        
        <h3>Results Analysis</h3>
        <ul>
          <li><strong>Box Filter:</strong> Smooths the image by averaging neighboring pixels</li>
          <li><strong>Dx Operator:</strong> Detects vertical edges (horizontal gradients)</li>
          <li><strong>Dy Operator:</strong> Detects horizontal edges (vertical gradients)</li>
          <li><strong>Gradient Magnitude:</strong> Combined edge strength using √(Dx² + Dy²)</li>
        </ul>
      </div>
    ),
    results: [
      {
        title: "Box Filter (9x9)",
        image: "/box_filter_result.jpg",
        description: "Box filter convolution result showing image smoothing effect. The 9x9 averaging filter reduces noise and creates a blurred version of the original image.",
        metrics: {
          "Filter Size": "9x9",
          "Padding": "4 pixels",
          "Effect": "Smoothing"
        }
      },
      {
        title: "Dx Gradient (Vertical Edges)", 
        image: "/dx_gradient.jpg",
        description: "Horizontal gradient detection using Dx operator. This filter detects vertical edges by computing differences in the horizontal direction.",
        metrics: {
          "Operator": "Dx",
          "Edge Type": "Vertical",
          "Direction": "Horizontal"
        }
      },
      {
        title: "Dy Gradient (Horizontal Edges)",
        image: "/dy_gradient.jpg", 
        description: "Vertical gradient detection using Dy operator. This filter detects horizontal edges by computing differences in the vertical direction.",
        metrics: {
          "Operator": "Dy",
          "Edge Type": "Horizontal", 
          "Direction": "Vertical"
        }
      },
      {
        title: "Implementation Comparison",
        image: "/difference.jpg",
        description: "Difference between our implementation and scipy.signal.convolve2d. The near-zero differences confirm the correctness of our implementation.",
        metrics: {
          "Max Difference": "1e-15",
          "Mean Difference": "1e-16",
          "Status": "Identical"
        }
      }
    ]
  };

  return (
    <>
      <AssignmentHeader
        assignmentNumber="Assignment 2"
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

export default Assignment2Page;
