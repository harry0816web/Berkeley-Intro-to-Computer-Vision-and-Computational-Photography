import React, { useState } from 'react';
import './Method.css';

const Method = ({ 
  methodExplanation, 
  results = [], 
  methodTitle = "Methodology" 
}) => {
  const [activeTab, setActiveTab] = useState('explanation');

  return (
    <section className="method-section">
      <div className="method-container">
        <h2 className="method-title">{methodTitle}</h2>
        
        <div className="method-tabs">
          <button 
            className={`tab-button ${activeTab === 'explanation' ? 'active' : ''}`}
            onClick={() => setActiveTab('explanation')}
          >
            Method Explanation
          </button>
          <button 
            className={`tab-button ${activeTab === 'results' ? 'active' : ''}`}
            onClick={() => setActiveTab('results')}
          >
            Results
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'explanation' && (
            <div className="explanation-content">
              <div className="explanation-text">
                {methodExplanation}
              </div>
            </div>
          )}

          {activeTab === 'results' && (
            <div className="results-content">
              <div className="results-grid">
                {results.map((result, index) => (
                  <div key={index} className="result-item">
                    <div className="result-image-container">
                      <img 
                        src={result.image} 
                        alt={result.title}
                        className="result-image"
                      />
                    </div>
                    <div className="result-info">
                      <h3 className="result-title">{result.title}</h3>
                      <p className="result-description">{result.description}</p>
                      {result.metrics && (
                        <div className="result-metrics">
                          {Object.entries(result.metrics).map(([key, value]) => (
                            <div key={key} className="metric">
                              <span className="metric-label">{key}:</span>
                              <span className="metric-value">{value}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default Method;
