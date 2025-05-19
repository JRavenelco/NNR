import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

function Navbar() {
  return (
    <nav className="navbar">
      <ul className="navbar-nav">
        <li className="nav-item">
          <Link to="/" className="nav-link">Visión General</Link>
        </li>
        <li className="nav-item">
          <Link to="/data-loader" className="nav-link">Carga de Datos (<code>data_loader.py</code>)</Link>
        </li>
        <li className="nav-item">
          <Link to="/model" className="nav-link">Modelo CNN (<code>model.py</code>)</Link>
        </li>
        <li className="nav-item">
          <Link to="/training" className="nav-link">Entrenamiento (<code>train.py</code>)</Link>
        </li>
        <li className="nav-item">
          <Link to="/predict-camera" className="nav-link">Predicción con Cámara (<code>predict_camera.py</code>)</Link>
        </li>
        <li className="nav-item">
          <Link to="/tensorboard" className="nav-link">Guía de TensorBoard</Link>
        </li>
      </ul>
    </nav>
  );
}

export default Navbar;
