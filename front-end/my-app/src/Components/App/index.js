import React from 'react';

import './App.css';
import Gallery from '../Gallery';

const App = () => (
  <div className="App">
    <header className="App-header">
      <h1 className="App-title">Near Duplicate Image Detector</h1>
      <small className="App-small">Click on an image to sort according to similarity</small>
    </header>
    <Gallery />
  </div>
);

export default App;
