import React from 'react';
import './AssignmentHeader.css';

const AssignmentHeader = ({ 
  title, 
  mainImage, 
  abstract, 
  assignmentNumber = "Assignment 1" 
}) => {
  return (
    <header className="assignment-header">
      <div className="header-content">
        <div className="assignment-info">
          <h1 className="assignment-title">{assignmentNumber}</h1>
          <h2 className="project-title">{title}</h2>
          <p className="abstract">{abstract}</p>
        </div>
        <div className="main-image-container">
          <img 
            src={mainImage} 
            alt={`${title} result`}
            className="main-image"
          />
        </div>
      </div>
    </header>
  );
};

export default AssignmentHeader;
