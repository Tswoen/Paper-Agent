import { useEffect, useState } from 'react'
import { Navigate, NavLink, Route, Routes, useNavigate } from 'react-router-dom'
import { ConfigurationPage } from './features/config/ConfigurationPage'
import { HistoryPage } from './features/history/HistoryPage'
import { ResearchPage } from './features/research/ResearchPage'

type Theme = 'light' | 'dark'

const navItems = [
  { title: '报告生成', path: '/', icon: 'R' },
  { title: '系统配置', path: '/configuration', icon: 'C' }
]

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [theme, setTheme] = useState<Theme>('light')
  const navigate = useNavigate()

  useEffect(() => {
    const savedTheme = localStorage.getItem('paper-agent-theme')
    if (savedTheme === 'light' || savedTheme === 'dark') {
      setTheme(savedTheme)
    }
  }, [])

  const toggleTheme = () => {
    setTheme((current) => {
      const next = current === 'light' ? 'dark' : 'light'
      localStorage.setItem('paper-agent-theme', next)
      return next
    })
  }

  return (
    <div className="main-layout" data-theme={theme}>
      <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        <header className="sidebar-header">
          <button className="brand-button" type="button" title="报告生成" onClick={() => navigate('/')}>
            <span className="brand-mark">PA</span>
            {!sidebarCollapsed && (
              <span className="brand-copy">
                <strong>Paper Agent</strong>
                <small>Academic Research</small>
              </span>
            )}
          </button>
          <button
            className="collapse-button"
            type="button"
            title={sidebarCollapsed ? '展开导航' : '收起导航'}
            onClick={() => setSidebarCollapsed((value) => !value)}
          >
            {sidebarCollapsed ? '›' : '‹'}
          </button>
        </header>

        <nav className="nav-list" aria-label="主导航">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
              title={sidebarCollapsed ? item.title : ''}
            >
              <span className="nav-icon">{item.icon}</span>
              {!sidebarCollapsed && <span className="nav-text">{item.title}</span>}
            </NavLink>
          ))}
        </nav>

        <footer className="sidebar-footer">
          <button
            className="theme-toggle"
            type="button"
            title={theme === 'light' ? '切换到 Dark' : '切换到 Light'}
            onClick={toggleTheme}
          >
            <span className="theme-dot">{theme === 'light' ? 'L' : 'D'}</span>
            {!sidebarCollapsed && (
              <span className="theme-label">
                <strong>{theme === 'light' ? 'Light' : 'Dark'}</strong>
                <small>全局主题</small>
              </span>
            )}
          </button>
        </footer>
      </aside>

      <main className="main-content">
        <section className="content-wrapper">
          <Routes>
            <Route path="/" element={<ResearchPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/configuration" element={<ConfigurationPage />} />
            <Route path="/knowledge" element={<Navigate to="/" replace />} />
          </Routes>
        </section>
      </main>
    </div>
  )
}
