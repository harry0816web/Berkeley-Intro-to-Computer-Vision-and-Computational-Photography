import React, { useState } from 'react';
import './App.css';
import Navigation from './components/Navigation';
import { 
  Assignment1Page, 
  Assignment2Page, 
  Assignment3Page, 
  Assignment4Page 
} from './pages';

function App() {
  const [currentAssignment, setCurrentAssignment] = useState(1);

  // Sample data for multiple assignments
  const assignments = [
    {
      number: 1,
      title: "Prokudin-Gorskii Photo Colorization"
    },
    {
      number: 2,
      title: "Convolutions from Scratch"
    },
    {
      number: 3,
      title: "Image Stitching & Panoramas"
    },
    {
      number: 4,
      title: "Object Detection & Recognition"
    }
  ];


  const handleAssignmentChange = (assignmentNumber) => {
    setCurrentAssignment(assignmentNumber);
  };

  // Personal information - update these with your details
  const personalInfo = {
    name: "Hung-I Yang",
    githubLink: "https://github.com/harry0816web",
    email: "harry940816@gmail.com",
    linkedinLink: "https://www.linkedin.com/in/harry-yang-073132219/"
  };

  // Render the appropriate page based on current assignment
  const renderCurrentPage = () => {
    switch (currentAssignment) {
      case 1:
        return <Assignment1Page {...personalInfo} />;
      case 2:
        return <Assignment2Page {...personalInfo} />;
      case 3:
        return <Assignment3Page {...personalInfo} />;
      case 4:
        return <Assignment4Page {...personalInfo} />;
      default:
        return <Assignment1Page {...personalInfo} />;
    }
  };

  return (
    <div className="App">
      <Navigation 
        assignments={assignments}
        currentAssignment={currentAssignment}
        onAssignmentChange={handleAssignmentChange}
      />
      
      {renderCurrentPage()}
    </div>
  );
}

export default App;
