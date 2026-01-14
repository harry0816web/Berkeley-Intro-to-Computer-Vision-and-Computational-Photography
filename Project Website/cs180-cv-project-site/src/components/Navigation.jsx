import React, { useState } from 'react';
import './Navigation.css';

const Navigation = ({ 
  assignments = [], 
  currentAssignment = 1,
  onAssignmentChange 
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  const handleAssignmentSelect = (assignmentNumber) => {
    onAssignmentChange(assignmentNumber);
    setIsOpen(false);
  };

  return (
    <nav className="navigation">
      <div className="nav-container">
        <button 
          className={`hamburger ${isOpen ? 'active' : ''}`}
          onClick={toggleMenu}
          aria-label="Toggle navigation menu"
        >
          <span className="hamburger-line"></span>
          <span className="hamburger-line"></span>
          <span className="hamburger-line"></span>
        </button>
        
        <div className={`nav-menu ${isOpen ? 'active' : ''}`}>
          <div className="nav-menu-content">
            <h3 className="nav-title">Assignments</h3>
            <ul className="nav-list">
              {assignments.map((assignment) => (
                <li key={assignment.number} className="nav-item">
                  <button
                    className={`nav-link ${currentAssignment === assignment.number ? 'active' : ''}`}
                    onClick={() => handleAssignmentSelect(assignment.number)}
                  >
                    <span className="nav-number">Assignment {assignment.number}</span>
                    <span className="nav-name">{assignment.title}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
