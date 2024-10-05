import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'

import $ from 'jquery'
import './assets/iconfont/iconfont.css'
import 'bootstrap'
import "bootstrap/dist/css/bootstrap.min.css";

import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'

// import 'jquery-ui-dist/jquery-ui'
// import 'jquery-ui-dist/jquery-ui.css'

// import '../static/jquery.range.js'
// import '../static/jquery.range.css'

import ViewUIPlus from 'view-ui-plus'
import 'view-ui-plus/dist/styles/viewuiplus.css'

import Popper from "vue3-popper";
import './assets/tooltip.css'

const app = createApp(App)

app.use(createPinia())
app.use(ElementPlus)
app.use(ViewUIPlus)
app.use(router)

app.component("Popper", Popper)

app.mount('#app')
