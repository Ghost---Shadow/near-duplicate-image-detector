import React, { Component } from 'react';
import './App.css';

import Gallery from './Gallery.js';

class App extends Component {
  render() {
    return (
      <div className="App">
        <header className="App-header">
          <h1 className="App-title">Near Duplicate Image Detector</h1>
        </header>
        <Gallery />
      </div>
    );
  }
}

export default App;
