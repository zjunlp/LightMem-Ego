import React from 'react'
import { createRoot } from 'react-dom/client'
import MemoryWorkspace from './components/MemoryWorkspace.jsx'
import ThemeProvider from './components/ThemeProvider.jsx'
import { AdminModeProvider } from './components/AdminMode.jsx'
import { BackendFeaturesProvider } from './components/BackendFeatures.jsx'
import './styles/app.css'
import './styles/workspace.css'
import './styles/theme.css'
import '../../history_demo/main-nav-entry.js'

createRoot(document.getElementById('root')).render(
  <React.StrictMode><BackendFeaturesProvider><ThemeProvider><AdminModeProvider><MemoryWorkspace /></AdminModeProvider></ThemeProvider></BackendFeaturesProvider></React.StrictMode>
)
