import { createRouter, createWebHistory } from 'vue-router'
import MainLayout from '../layouts/MainLayout.vue'
import Home from '../App.vue'
import History from '../views/History.vue'
import Configuration from '../views/Configuration.vue'

const routes = [
  {
    path: '/',
    component: MainLayout,
    children: [
      {
        path: '',
        name: 'Home',
        component: Home,
        meta: {
          title: '报告生成',
          icon: 'R'
        }
      },
      {
        path: 'history',
        name: 'History',
        component: History,
        meta: {
          title: '历史报告',
          icon: 'H'
        }
      },
      {
        path: 'configuration',
        name: 'Configuration',
        component: Configuration,
        meta: {
          title: '系统配置',
          icon: 'C'
        }
      },
      {
        path: 'knowledge',
        redirect: '/'
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
