import React from 'react'
import '../../history_demo/main-nav-entry.js'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import ThemeProvider from './components/ThemeProvider.jsx'
import { AdminModeProvider } from './components/AdminMode.jsx'
import { BackendFeaturesProvider } from './components/BackendFeatures.jsx'
import './styles/app.css'
import './styles/workspace.css'
import './styles/theme.css'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BackendFeaturesProvider><ThemeProvider><AdminModeProvider><App /></AdminModeProvider></ThemeProvider></BackendFeaturesProvider>
  </React.StrictMode>
)
