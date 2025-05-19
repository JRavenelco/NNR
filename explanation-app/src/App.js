import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import Header from './components/Header';
import Navbar from './components/Navbar'; 
import OverviewPage from './pages/OverviewPage';
import DataLoaderPage from './pages/DataLoaderPage'; 
import ModelPage from './pages/ModelPage';
import TrainPage from './pages/TrainPage';
import PredictCameraPage from './pages/PredictCameraPage';
import TensorBoardGuidePage from './pages/TensorBoardGuidePage';

function App() {
  return (
    <Router>
      <div className="App">
        <Header />
        <Navbar /> 
        <main className="container">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/data-loader" element={<DataLoaderPage />} />
            <Route path="/model" element={<ModelPage />} />
            <Route path="/training" element={<TrainPage />} />
            <Route path="/predict-camera" element={<PredictCameraPage />} />
            <Route path="/tensorboard" element={<TensorBoardGuidePage />} />
          </Routes>
        </main>
        <footer className="footer">
          <p>&copy; 2025 Explicación del Proyecto CIFAR-100</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
